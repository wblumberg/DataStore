"""Profile-driven orchestration for operational and local datastore workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from ..catalog.contract import assert_report_ok, validate_observation_db_contract, validate_zarr_contract
from ..catalog.manifest import ProductRecord, append_product_record, build_store_metadata
from ..catalog.paths import ProductLayout
from ..observations.obs2sql import parse_input_file
from ..observations.sql_store import ObservationSQLStore, parse_time_like, to_iso_utc
from ..pipelines.lagged_ensemble import build_lagged_ensemble
from ..pipelines.postprocess_ensemble import postprocess_member_stores
from ..sources.tropical.atcf import sync_aid_public
from ..sources.tropical.tracks import build_ecmwf_tracks


@dataclass(frozen=True)
class WorkflowStep:
    """One actionable orchestration step in a profile."""

    name: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class WorkflowProfile:
    """Top-level workflow profile model."""

    name: str
    description: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    steps: tuple[WorkflowStep, ...] = ()


@dataclass
class StepResult:
    """Execution result for one step."""

    step: str
    action: str
    status: str
    started_at: str
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


def load_profile(path: Path | str) -> WorkflowProfile:
    """Load a JSON workflow profile from disk."""
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))

    name = str(payload.get("name") or profile_path.stem)
    description = str(payload.get("description") or "")
    variables = payload.get("variables") or {}

    if not isinstance(variables, dict):
        raise ValueError("profile.variables must be a JSON object")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("profile.steps must be a non-empty list")

    steps: list[WorkflowStep] = []
    for idx, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"profile.steps[{idx}] must be a JSON object")

        step_name = str(raw.get("name") or f"step_{idx}")
        action = str(raw.get("action") or "").strip()
        if not action:
            raise ValueError(f"profile.steps[{idx}] missing required action")

        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"profile.steps[{idx}].params must be a JSON object")

        enabled = bool(raw.get("enabled", True))
        steps.append(WorkflowStep(name=step_name, action=action, params=params, enabled=enabled))

    return WorkflowProfile(
        name=name,
        description=description,
        variables=dict(variables),
        steps=tuple(steps),
    )


def run_profile(
    profile: WorkflowProfile,
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
    only_steps: Sequence[str] | None = None,
    variable_overrides: Mapping[str, Any] | None = None,
) -> list[StepResult]:
    """Execute a workflow profile and return per-step results."""
    context = dict(profile.variables)
    if variable_overrides:
        context.update(variable_overrides)

    selected = set(only_steps or [])
    results: list[StepResult] = []

    for step in profile.steps:
        if not step.enabled:
            continue
        if selected and step.name not in selected:
            continue

        started = datetime.now(tz=timezone.utc)
        t0 = time.perf_counter()

        resolved_params = _resolve_templates(step.params, context)

        if dry_run:
            preview = _preview_step(step.action, resolved_params)
            results.append(
                StepResult(
                    step=step.name,
                    action=step.action,
                    status="dry-run",
                    started_at=to_iso_utc(started),
                    duration_seconds=0.0,
                    details={"preview": preview, "params": resolved_params},
                )
            )
            continue

        try:
            details = _execute_step(step.action, resolved_params)
            duration = time.perf_counter() - t0
            results.append(
                StepResult(
                    step=step.name,
                    action=step.action,
                    status="ok",
                    started_at=to_iso_utc(started),
                    duration_seconds=duration,
                    details=details,
                )
            )
        except Exception as exc:
            duration = time.perf_counter() - t0
            error_result = StepResult(
                step=step.name,
                action=step.action,
                status="error",
                started_at=to_iso_utc(started),
                duration_seconds=duration,
                details={"error": str(exc), "params": resolved_params},
            )
            results.append(error_result)
            if not continue_on_error:
                break

    return results


def _execute_step(action: str, params: dict[str, Any]) -> dict[str, Any]:
    if action == "download_models":
        return _step_download_models(params)
    if action == "build_lagged_ensemble":
        return _step_build_lagged_ensemble(params)
    if action == "postprocess_ensemble":
        return _step_postprocess_ensemble(params)
    if action == "obs_init":
        return _step_obs_init(params)
    if action == "obs_ingest":
        return _step_obs_ingest(params)
    if action == "obs_query":
        return _step_obs_query(params)
    if action == "sync_atcf":
        return _step_sync_atcf(params)
    if action == "build_tracks":
        return _step_build_tracks(params)
    if action == "write_manifest":
        return _step_write_manifest(params)
    if action == "validate_contract":
        return _step_validate_contract(params)
    if action == "catalog_paths":
        return _step_catalog_paths(params)
    if action == "shell":
        return _step_shell(params)
    if action == "slurm_submit":
        return _step_slurm_submit(params)

    raise ValueError(f"Unsupported action '{action}'")


def _preview_step(action: str, params: dict[str, Any]) -> str:
    if action == "shell":
        return str(params.get("command", ""))
    if action == "slurm_submit":
        script = params.get("script", "")
        args = [str(v) for v in params.get("args", [])]
        return " ".join(["sbatch", str(script)] + [shlex.quote(arg) for arg in args])
    return f"{action} {json.dumps(params, default=str)}"


def _step_download_models(params: dict[str, Any]) -> dict[str, Any]:
    from ..sources.remote.download_hrefnssl_models import MODELS, process_model

    token = str(params.get("model_token") or params.get("model") or "").strip()
    if not token:
        raise ValueError("download_models requires model_token")

    max_workers = int(params.get("max_workers", 4))
    matched = []

    for cfg in MODELS:
        if token == cfg.get("model") or token in str(cfg.get("gempak_pattern", "")):
            process_model(cfg, max_workers=max_workers)
            matched.append(cfg.get("gempak_pattern", cfg.get("model", "unknown")))

    if not matched:
        raise ValueError(f"No model configuration matched {token!r}")

    return {"matched": matched, "max_workers": max_workers}


def _step_build_lagged_ensemble(params: dict[str, Any]) -> dict[str, Any]:
    input_root = params.get("input_root")
    output_zarr = params.get("output_zarr")
    if not input_root or not output_zarr:
        raise ValueError("build_lagged_ensemble requires input_root and output_zarr")

    output_path = build_lagged_ensemble(
        input_root=Path(str(input_root)),
        output_zarr=Path(str(output_zarr)),
        max_lags=int(params.get("max_lags", 2)),
        exclude_dirs=[str(item) for item in params.get("exclude_dirs", [])],
        patterns=[str(item) for item in params.get("patterns", ["*.grib2"])],
    )
    return {"output": str(output_path)}


def _step_postprocess_ensemble(params: dict[str, Any]) -> dict[str, Any]:
    member_zarrs = params.get("member_zarrs")
    variable = params.get("variable")
    output_zarr = params.get("output_zarr")

    if not member_zarrs or not variable or not output_zarr:
        raise ValueError("postprocess_ensemble requires member_zarrs, variable, and output_zarr")

    diag = postprocess_member_stores(
        member_zarrs=[Path(str(path)) for path in member_zarrs],
        variable=str(variable),
        output_zarr=Path(str(output_zarr)),
        thresholds=params.get("thresholds"),
        percentile_probs=params.get("percentile_probs"),
        include_pmm=bool(params.get("include_pmm", True)),
        include_lpmm=bool(params.get("include_lpmm", True)),
        lpmm_radius_x=int(params.get("lpmm_radius_x", 10)),
        lpmm_radius_y=int(params.get("lpmm_radius_y", 10)),
    )
    return {"output": str(output_zarr), "variables": list(diag.data_vars)}


def _step_obs_init(params: dict[str, Any]) -> dict[str, Any]:
    db_path = Path(str(params.get("db") or "observations.sqlite")).expanduser().resolve()
    store = ObservationSQLStore(db_path)
    store.initialize()
    return {"db": str(db_path)}


def _step_obs_ingest(params: dict[str, Any]) -> dict[str, Any]:
    db_path = Path(str(params.get("db") or "observations.sqlite")).expanduser().resolve()
    obs_type = str(params.get("obs_type") or "UNKNOWN").strip().upper()
    file_format = str(params.get("format", "auto"))

    raw_inputs = params.get("input")
    if not raw_inputs or not isinstance(raw_inputs, list):
        raise ValueError("obs_ingest requires input list")

    assume_time = parse_time_like(params.get("assume_time")) if params.get("assume_time") else None
    source_name_override = params.get("source_name")

    store = ObservationSQLStore(db_path)
    store.initialize()

    rows = []
    per_file_counts: dict[str, int] = {}

    for raw in raw_inputs:
        path = Path(str(raw)).expanduser().resolve()
        parsed = parse_input_file(
            path=path,
            obs_type=obs_type,
            source_name=str(source_name_override) if source_name_override else path.name,
            assume_time=assume_time,
            file_format=file_format,
            gempak_country=params.get("gempak_country"),
            gempak_date_time=params.get("gempak_date_time"),
        )
        rows.extend(parsed)
        per_file_counts[str(path)] = len(parsed)

    result = store.insert_many(rows)
    return {
        "db": str(db_path),
        "obs_type": obs_type,
        "files": per_file_counts,
        **result,
    }


def _step_obs_query(params: dict[str, Any]) -> dict[str, Any]:
    db_path = Path(str(params.get("db") or "observations.sqlite")).expanduser().resolve()
    store = ObservationSQLStore(db_path)
    store.initialize()

    obs_types = params.get("obs_types")
    if obs_types is not None and not isinstance(obs_types, list):
        raise ValueError("obs_query.obs_types must be a list when provided")

    parameter_names = params.get("parameter_names")
    if parameter_names is not None and not isinstance(parameter_names, list):
        raise ValueError("obs_query.parameter_names must be a list when provided")

    result = store.query_observations(
        obs_types=obs_types,
        center_time=parse_time_like(params.get("center_time")) if params.get("center_time") else None,
        minutes_before=int(params.get("minutes_before", 0)),
        minutes_after=int(params.get("minutes_after", 0)),
        latest_only=bool(params.get("latest_only", False)),
        prefer_most_data=bool(params.get("prefer_most_data", True)),
        parameter_names=parameter_names,
        bin_minutes=int(params.get("bin_minutes", 0)),
        max_rows=int(params.get("max_rows", 50000)),
    )
    result["db"] = str(db_path)
    return result


def _step_sync_atcf(params: dict[str, Any]) -> dict[str, Any]:
    rc = sync_aid_public(
        output_dir=Path(str(params.get("output_dir") or "/data/gempak/atcf/")),
        download_dir=Path(str(params.get("download_dir") or ".")),
        min_age_hours=float(params.get("min_age_hours", 0.0)),
    )
    return {"status_code": rc}


def _step_build_tracks(params: dict[str, Any]) -> dict[str, Any]:
    out_path = build_ecmwf_tracks(out_dir=Path(str(params.get("out_dir") or "/data/gempak/storm/enstrack/")))
    return {"output": str(out_path)}


def _step_write_manifest(params: dict[str, Any]) -> dict[str, Any]:
    """Write or append a product record to a run manifest JSON."""
    manifest_path = params.get("manifest_path")
    if not manifest_path:
        raise ValueError("write_manifest requires manifest_path")

    product_type = str(params.get("product_type") or "unknown")
    path = str(params.get("path") or "")
    system = str(params.get("system") or "") or None
    run_id = str(params.get("run_id") or "") or None
    workflow = str(params.get("workflow") or "") or None
    profile = str(params.get("profile") or "") or None
    extra_metadata = params.get("metadata") or {}

    record = ProductRecord(
        product_type=product_type,
        path=path,
        system=system,
        run_id=run_id,
        metadata=dict(extra_metadata),
    )
    out = append_product_record(
        manifest_path=manifest_path,
        record=record,
        workflow=workflow,
        profile=profile,
        system=system,
        run_id=run_id,
    )
    return {"manifest_path": str(out), "product_type": product_type, "path": path}


def _step_validate_contract(params: dict[str, Any]) -> dict[str, Any]:
    """Validate a data product against its schema/metadata contract."""
    target = params.get("target") or params.get("path")
    if not target:
        raise ValueError("validate_contract requires target (or path)")

    target_path = Path(str(target)).expanduser().resolve()
    product_type = str(params.get("product_type") or "")
    raise_on_error = bool(params.get("raise_on_error", True))

    # Detect store type from params or file extension
    store_type = str(params.get("store_type") or "").lower()
    if not store_type:
        if target_path.suffix == ".sqlite":
            store_type = "sqlite"
        else:
            store_type = "zarr"

    if store_type == "sqlite":
        report = validate_observation_db_contract(target_path)
    else:
        report = validate_zarr_contract(target_path, product_type=product_type or None)

    if raise_on_error:
        assert_report_ok(report)

    return {
        "target": str(target_path),
        "product_type": report.product_type,
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
    }


def _step_catalog_paths(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve canonical product paths using ProductLayout and return them."""
    base_dir = params.get("base_dir")
    if not base_dir:
        raise ValueError("catalog_paths requires base_dir")

    layout = ProductLayout(base_dir=Path(str(base_dir)).expanduser().resolve())
    system = str(params.get("system") or "")
    run_id = str(params.get("run_id") or "")

    resolved: dict[str, str] = {}

    if members := params.get("members"):
        for member in members:
            key = f"member_{member}"
            resolved[key] = str(layout.member_store(system=system, run_id=run_id, member=str(member)))

    if ensemble_name := params.get("ensemble_name"):
        resolved["ensemble"] = str(layout.ensemble_store(system=system, run_id=run_id, name=str(ensemble_name)))

    if product := params.get("diagnostic_product"):
        resolved["diagnostic"] = str(layout.diagnostic_store(system=system, run_id=run_id, product=str(product)))

    if params.get("manifest"):
        resolved["manifest"] = str(layout.manifest_path(system=system, run_id=run_id))

    if obs_dataset := params.get("obs_dataset"):
        resolved["observations"] = str(layout.observations_db(dataset=str(obs_dataset), date_key=run_id or None))

    return {"base_dir": str(layout.base_dir), "system": system, "run_id": run_id, "paths": resolved}


def _step_shell(params: dict[str, Any]) -> dict[str, Any]:
    command = str(params.get("command") or "").strip()
    if not command:
        raise ValueError("shell step requires command")

    cwd = params.get("cwd")
    cwd_path = Path(str(cwd)).expanduser().resolve() if cwd else None

    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd_path) if cwd_path else None,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"shell command failed ({proc.returncode}): {proc.stderr.strip()}")

    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _step_slurm_submit(params: dict[str, Any]) -> dict[str, Any]:
    script = params.get("script")
    if not script:
        raise ValueError("slurm_submit requires script")

    args = [str(item) for item in params.get("args", [])]
    command = ["sbatch", str(script), *args]

    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed ({proc.returncode}): {proc.stderr.strip()}")

    return {
        "command": " ".join(shlex.quote(item) for item in command),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def _resolve_templates(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(_SafeFormatMap(context))
        except Exception:
            return value

    if isinstance(value, list):
        return [_resolve_templates(item, context) for item in value]

    if isinstance(value, dict):
        return {str(key): _resolve_templates(item, context) for key, item in value.items()}

    return value


class _SafeFormatMap(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def summarize_results(results: Sequence[StepResult]) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for result in results if result.status == "ok")
    dry = sum(1 for result in results if result.status == "dry-run")
    errors = [result for result in results if result.status == "error"]

    return {
        "total": total,
        "ok": ok,
        "dry_run": dry,
        "errors": len(errors),
        "failed_steps": [result.step for result in errors],
    }


def _parse_overrides(values: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --set value {raw!r}; expected key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value {raw!r}; key cannot be empty")
        out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run datastore workflow profiles")
    parser.add_argument("--profile", required=True, help="Path to JSON workflow profile")
    parser.add_argument("--dry-run", action="store_true", help="Render steps without executing")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps when one step fails",
    )
    parser.add_argument(
        "--only-step",
        action="append",
        default=[],
        help="Run only named step(s); can be repeated",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override profile variable value using key=value; can be repeated",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    overrides = _parse_overrides(args.set)

    results = run_profile(
        profile,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        only_steps=args.only_step,
        variable_overrides=overrides,
    )

    summary = summarize_results(results)

    if args.json:
        payload = {
            "profile": profile.name,
            "summary": summary,
            "results": [
                {
                    "step": result.step,
                    "action": result.action,
                    "status": result.status,
                    "started_at": result.started_at,
                    "duration_seconds": result.duration_seconds,
                    "details": result.details,
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(f"Profile: {profile.name}")
        if profile.description:
            print(profile.description)
        print(f"Summary: {summary}")
        for result in results:
            print(
                f"[{result.status}] {result.step} ({result.action}) "
                f"{result.duration_seconds:.3f}s"
            )

    return 0 if summary["errors"] == 0 else 1


__all__ = [
    "StepResult",
    "WorkflowProfile",
    "WorkflowStep",
    "build_parser",
    "load_profile",
    "main",
    "run_profile",
    "summarize_results",
]


if __name__ == "__main__":
    raise SystemExit(main())

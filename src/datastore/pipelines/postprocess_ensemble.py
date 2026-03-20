"""Post-process member Zarr stores into derived ensemble diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from ..catalog.contract import assert_report_ok, validate_zarr_contract
from ..catalog.manifest import ProductRecord, append_product_record, build_store_metadata
from ..diagnostics.ensemble import process_ensemble_to_zarr


def load_member_ensemble(member_zarrs: Sequence[Path | str], chunks: Mapping | None = None) -> xr.Dataset:
    """Open member stores and concatenate over the member dimension."""
    if not member_zarrs:
        raise ValueError("member_zarrs is required")

    datasets: list[xr.Dataset] = []
    for raw in member_zarrs:
        path = Path(raw)
        ds = xr.open_zarr(str(path), chunks=chunks)

        member_name = path.name
        if member_name.endswith(".zarr"):
            member_name = member_name[:-5]

        if "member" not in ds.dims:
            ds = ds.expand_dims(member=[member_name])
        elif ds.sizes.get("member", 0) == 1:
            ds = ds.assign_coords(member=[member_name])

        datasets.append(ds)

    return xr.concat(datasets, dim="member")


def _update_store_attrs(*, output_zarr: Path | str, attrs: Mapping[str, Any]) -> None:
    import zarr

    root = zarr.open_group(str(output_zarr), mode="a")
    root.attrs.update(dict(attrs))


def postprocess_member_stores(
    *,
    member_zarrs: Sequence[Path | str],
    variable: str,
    output_zarr: Path | str,
    thresholds: Mapping[str, float] | None = None,
    percentile_probs: Sequence[float] | None = None,
    include_pmm: bool = True,
    include_lpmm: bool = True,
    lpmm_radius_x: int = 10,
    lpmm_radius_y: int = 10,
    chunks: Mapping | None = None,
    store_metadata: Mapping[str, Any] | None = None,
    manifest_path: Path | str | None = None,
    validate_contract: bool = False,
) -> xr.Dataset:
    """Load member stores, compute diagnostics, and write an output Zarr."""
    ensemble = load_member_ensemble(member_zarrs=member_zarrs, chunks=chunks)

    default_run_id = str(ensemble.attrs.get("run_id") or "")
    default_system = str(ensemble.attrs.get("system") or "")

    metadata = build_store_metadata(
        product_type="diagnostics",
        system=default_system or None,
        run_id=default_run_id or None,
        source="datastore.pipelines.postprocess_ensemble",
        extra={"variable": variable},
    )
    if store_metadata:
        metadata.update(dict(store_metadata))

    diagnostics_ds = process_ensemble_to_zarr(
        ds=ensemble,
        variable=variable,
        output_zarr_path=output_zarr,
        thresholds=thresholds,
        percentile_probs=percentile_probs,
        include_pmm=include_pmm,
        include_lpmm=include_lpmm,
        lpmm_radius_x=lpmm_radius_x,
        lpmm_radius_y=lpmm_radius_y,
    )

    _update_store_attrs(output_zarr=output_zarr, attrs=metadata)

    if manifest_path:
        append_product_record(
            manifest_path=manifest_path,
            record=ProductRecord(
                product_type="diagnostics",
                path=str(Path(output_zarr).expanduser().resolve()),
                system=metadata.get("system"),
                run_id=metadata.get("run_id"),
                metadata=metadata,
            ),
            workflow="postprocess_ensemble",
            profile=str(metadata.get("profile", "")),
            system=metadata.get("system"),
            run_id=metadata.get("run_id"),
        )

    if validate_contract:
        report = validate_zarr_contract(output_zarr, product_type="diagnostics")
        assert_report_ok(report)

    diagnostics_ds.attrs.update(metadata)
    return diagnostics_ds


def _parse_thresholds(values: Sequence[str] | None) -> dict[str, float] | None:
    if not values:
        return None

    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid threshold '{value}'. Expected name=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid threshold '{value}'. Name cannot be empty.")
        out[key] = float(raw)

    return out or None


def _parse_percentiles(values: Sequence[str] | None) -> list[float] | None:
    if not values:
        return None
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process member stores into ensemble diagnostics")
    parser.add_argument("--member-zarrs", nargs="+", required=True, help="Input member Zarr paths")
    parser.add_argument("--variable", required=True, help="Variable name to process")
    parser.add_argument("--output-zarr", required=True, help="Output diagnostics Zarr path")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold in name=value format (can be repeated)",
    )
    parser.add_argument(
        "--percentile",
        action="append",
        default=[],
        help="Percentile probability between 0 and 1 (can be repeated)",
    )
    parser.add_argument("--include-pmm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lpmm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpmm-radius-x", type=int, default=10)
    parser.add_argument("--lpmm-radius-y", type=int, default=10)
    parser.add_argument("--manifest-path", help="Optional manifest JSON path to append output record")
    parser.add_argument("--system", help="Optional system label for provenance metadata")
    parser.add_argument("--run-id", help="Optional run identifier for provenance metadata")
    parser.add_argument("--profile", help="Optional workflow profile name for provenance metadata")
    parser.add_argument("--validate-contract", action="store_true", help="Validate output contract after write")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    thresholds = _parse_thresholds(args.threshold)
    percentile_probs = _parse_percentiles(args.percentile)

    metadata = build_store_metadata(
        product_type="diagnostics",
        system=args.system,
        run_id=args.run_id,
        profile=args.profile,
        workflow="postprocess_ensemble",
        source="datastore.pipelines.postprocess_ensemble",
    )

    postprocess_member_stores(
        member_zarrs=args.member_zarrs,
        variable=args.variable,
        output_zarr=args.output_zarr,
        thresholds=thresholds,
        percentile_probs=percentile_probs,
        include_pmm=args.include_pmm,
        include_lpmm=args.include_lpmm,
        lpmm_radius_x=args.lpmm_radius_x,
        lpmm_radius_y=args.lpmm_radius_y,
        store_metadata=metadata,
        manifest_path=args.manifest_path,
        validate_contract=args.validate_contract,
    )
    return 0


__all__ = [
    "build_parser",
    "load_member_ensemble",
    "main",
    "postprocess_member_stores",
]


if __name__ == "__main__":
    raise SystemExit(main())

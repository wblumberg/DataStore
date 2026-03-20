"""Build streamlined HREF-derived diagnostics from member Zarr stores.

This script is designed for the common workflow:
1) open all member-level Zarr stores,
2) compute selected derived products,
3) write a single diagnostics Zarr store.

Derived products written by default:
- <uh_var>_4h_member_max: rolling 4-hour max, then member max
- <temp_var>_mean: ensemble mean temperature field
- <dewpoint_var>_mean: ensemble mean dewpoint field
- <uh_var>_gt_<threshold>_nprob_4h: 4-hour neighborhood probability exceedance
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import xarray as xr

from datastore.catalog.contract import assert_report_ok, validate_zarr_contract
from datastore.catalog.manifest import ProductRecord, append_product_record, build_store_metadata
from datastore.diagnostics.ensemble import (
    infer_ensemble_dims,
    mean_field,
    neighborhood_probability_time_window,
    rolling_window_max,
    write_dataset_to_zarr,
)
from datastore.pipelines.postprocess_ensemble import load_member_ensemble


def _threshold_token(value: float) -> str:
    token = str(value)
    token = token.replace("-", "m").replace(".", "p")
    return token


def _must_have_var(ds: xr.Dataset, name: str) -> None:
    if name not in ds:
        available = ", ".join(sorted(ds.data_vars))
        raise KeyError(f"Variable '{name}' not found. Available data variables: {available}")


def build_href_products(
    *,
    member_zarrs: list[str],
    output_zarr: str,
    uh_var: str = "uh",
    temp_var: str = "tmp2m",
    dewpoint_var: str = "dpt2m",
    uh_threshold: float = 75.0,
    time_window_steps: int = 4,
    radius_x: int = 10,
    radius_y: int = 10,
    include_time_member_means: bool = False,
    system: str | None = None,
    run_id: str | None = None,
    profile: str | None = None,
    manifest_path: str | None = None,
    validate_contract: bool = False,
) -> Path:
    if time_window_steps < 1:
        raise ValueError("time_window_steps must be >= 1")

    ensemble = load_member_ensemble(member_zarrs=[Path(p) for p in member_zarrs])

    _must_have_var(ensemble, uh_var)
    _must_have_var(ensemble, temp_var)
    _must_have_var(ensemble, dewpoint_var)

    uh = ensemble[uh_var]
    uh_dims = infer_ensemble_dims(uh)

    if uh_dims.time_dim is None:
        raise ValueError("Could not infer time dimension for UH variable")

    uh_rolling = rolling_window_max(
        uh,
        window=time_window_steps,
        time_dim=uh_dims.time_dim,
        min_periods=time_window_steps,
    )
    uh_4h_member_max = uh_rolling.max(dim=uh_dims.member_dim, skipna=True)

    uh_nprob = neighborhood_probability_time_window(
        da=uh,
        threshold=uh_threshold,
        time_window_steps=time_window_steps,
        radius_x=radius_x,
        radius_y=radius_y,
        member_dim=uh_dims.member_dim,
        time_dim=uh_dims.time_dim,
        x_dim=uh_dims.x_dim,
        y_dim=uh_dims.y_dim,
        strict=True,
    )

    temp_da = ensemble[temp_var]
    dewpoint_da = ensemble[dewpoint_var]
    temp_dims = infer_ensemble_dims(temp_da)
    dewpoint_dims = infer_ensemble_dims(dewpoint_da)

    temp_mean = mean_field(temp_da, member_dim=temp_dims.member_dim)
    dewpoint_mean = mean_field(dewpoint_da, member_dim=dewpoint_dims.member_dim)

    thr_token = _threshold_token(uh_threshold)

    output_vars: dict[str, xr.DataArray] = {
        f"{uh_var}_{time_window_steps}h_member_max": uh_4h_member_max,
        f"{temp_var}_mean": temp_mean,
        f"{dewpoint_var}_mean": dewpoint_mean,
        f"{uh_var}_gt_{thr_token}_nprob_{time_window_steps}h": uh_nprob,
    }

    if include_time_member_means:
        output_vars[f"{temp_var}_time_member_mean"] = temp_da.mean(
            dim=[temp_dims.time_dim, temp_dims.member_dim],
            skipna=True,
        )
        output_vars[f"{dewpoint_var}_time_member_mean"] = dewpoint_da.mean(
            dim=[dewpoint_dims.time_dim, dewpoint_dims.member_dim],
            skipna=True,
        )

    out = xr.Dataset(output_vars)

    metadata = build_store_metadata(
        product_type="diagnostics",
        system=system or str(ensemble.attrs.get("system") or "") or None,
        run_id=run_id or str(ensemble.attrs.get("run_id") or "") or None,
        profile=profile,
        workflow="postprocess_href_products",
        source="scripts.postprocess_href_products",
        extra={
            "uh_var": uh_var,
            "temp_var": temp_var,
            "dewpoint_var": dewpoint_var,
            "uh_threshold": float(uh_threshold),
            "time_window_steps": int(time_window_steps),
            "radius_x": int(radius_x),
            "radius_y": int(radius_y),
        },
    )
    out.attrs.update(metadata)

    output_path = Path(output_zarr).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_to_zarr(out, output_path, mode="w", consolidated=True)

    if manifest_path:
        append_product_record(
            manifest_path=manifest_path,
            record=ProductRecord(
                product_type="diagnostics",
                path=str(output_path),
                system=metadata.get("system"),
                run_id=metadata.get("run_id"),
                metadata=dict(metadata),
            ),
            workflow="postprocess_href_products",
            profile=str(metadata.get("profile", "")),
            system=metadata.get("system"),
            run_id=metadata.get("run_id"),
        )

    if validate_contract:
        report = validate_zarr_contract(output_path, product_type="diagnostics")
        assert_report_ok(report)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build streamlined HREF diagnostics from member Zarr stores")
    parser.add_argument("--member-zarrs", nargs="+", required=True, help="Input member Zarr paths")
    parser.add_argument("--output-zarr", required=True, help="Output diagnostics Zarr path")
    parser.add_argument("--uh-var", default="uh", help="UH variable name")
    parser.add_argument("--temp-var", default="tmp2m", help="2m temperature variable name")
    parser.add_argument("--dewpoint-var", default="dpt2m", help="2m dewpoint variable name")
    parser.add_argument("--uh-threshold", type=float, default=75.0, help="UH exceedance threshold")
    parser.add_argument("--time-window-steps", type=int, default=4, help="Rolling time window length in steps")
    parser.add_argument("--radius-x", type=int, default=10, help="Neighborhood radius in x grid points")
    parser.add_argument("--radius-y", type=int, default=10, help="Neighborhood radius in y grid points")
    parser.add_argument(
        "--include-time-member-means",
        action="store_true",
        help="Also write time-and-member means for temperature and dewpoint",
    )
    parser.add_argument("--system", help="Optional system label for provenance metadata")
    parser.add_argument("--run-id", help="Optional run ID for provenance metadata")
    parser.add_argument("--profile", help="Optional profile label for provenance metadata")
    parser.add_argument("--manifest-path", help="Optional manifest JSON path to append output record")
    parser.add_argument("--validate-contract", action="store_true", help="Validate output contract after write")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_href_products(
        member_zarrs=args.member_zarrs,
        output_zarr=args.output_zarr,
        uh_var=args.uh_var,
        temp_var=args.temp_var,
        dewpoint_var=args.dewpoint_var,
        uh_threshold=args.uh_threshold,
        time_window_steps=args.time_window_steps,
        radius_x=args.radius_x,
        radius_y=args.radius_y,
        include_time_member_means=args.include_time_member_means,
        system=args.system,
        run_id=args.run_id,
        profile=args.profile,
        manifest_path=args.manifest_path,
        validate_contract=args.validate_contract,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

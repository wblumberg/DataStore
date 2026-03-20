"""Build a lagged ensemble Zarr store from local GRIB2 files."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import grib2io
import numpy as np
import xarray as xr

from ..catalog.manifest import build_store_metadata
from ..core.models import ForecastFile, GridInfo, VariableInfo
from ..grib.discovery import build_file_index, discover_files, match_by_valid_time
from ..grib.grid import extract_grid_info
from ..grib.inventory import inventory_variables
from .adjustments import adjust_accumulated_for_lagged

logger = logging.getLogger(__name__)


def write_zarr_store(
    matches: dict[datetime, list[ForecastFile]],
    variables: Sequence[VariableInfo],
    grid: GridInfo,
    output_path: Path,
    chunk_size: int = 8,
    store_metadata: Mapping[str, Any] | None = None,
) -> None:
    import zarr
    from numcodecs import Blosc

    logger.info("Writing Zarr store to %s", output_path)

    times = sorted(matches.keys())
    members = sorted({forecast_file.member for matched_files in matches.values() for forecast_file in matched_files})
    ny, nx = grid.nj, grid.ni

    time_arr = np.array([np.datetime64(valid_time) for valid_time in times])
    member_arr = np.array(members, dtype=f"U{max(len(m) for m in members)}")
    y_arr = np.arange(ny, dtype=np.int32)
    x_arr = np.arange(nx, dtype=np.int32)

    variable_arrays: dict[str, np.ndarray] = {}
    for variable in variables:
        variable_arrays[variable.name] = np.full((len(times), len(members), ny, nx), np.nan, dtype=np.float32)

    for start_index in range(0, len(times), chunk_size):
        end_index = min(start_index + chunk_size, len(times))
        logger.info("Processing times %s-%s of %s", start_index + 1, end_index, len(times))
        for time_index in range(start_index, end_index):
            valid_time = times[time_index]
            for forecast_file in matches[valid_time]:
                member_index = members.index(forecast_file.member)
                try:
                    with grib2io.open(str(forecast_file.path)) as grib_file:
                        for message in grib_file:
                            variable_name = message.shortName
                            if variable_name in variable_arrays:
                                variable_arrays[variable_name][time_index, member_index, :, :] = message.data
                except Exception as exc:
                    logger.error("Error reading %s: %s", forecast_file.path, exc)

    data_vars = {}
    for variable in variables:
        data_vars[variable.name] = xr.DataArray(
            variable_arrays[variable.name],
            dims=["time", "member", "y", "x"],
            attrs={
                "long_name": variable.long_name,
                "units": variable.units,
                "level1": variable.level1,
                "level2": variable.level2,
                "level_type": variable.level_type,
                "type": variable.type,
            },
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time_arr),
            "member": ("member", member_arr),
            "y": ("y", y_arr),
            "x": ("x", x_arr),
        },
        attrs={
            "grid_type": "lambert",
            "ni": grid.ni,
            "nj": grid.nj,
            "lon_0": grid.lon_0,
            "lat_0": grid.lat_0,
            "lat_std": grid.lat_std,
            "ll_lat": grid.ll_lat,
            "ll_lon": grid.ll_lon,
            "ur_lat": grid.ur_lat,
            "ur_lon": grid.ur_lon,
        },
    )

    metadata = dict(store_metadata or {})
    metadata.setdefault("product_type", "lagged_ensemble")
    metadata.setdefault("created_at", datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    dataset.attrs.update(metadata)

    encoding = {
        variable.name: {"chunks": (1, 1, 256, 256), "compressor": Blosc(cname="zstd", clevel=5)}
        for variable in variables
    }

    dataset.to_zarr(
        str(output_path),
        mode="w",
        consolidated=True,
        encoding=encoding,
        zarr_format=2,
    )
    zarr.consolidate_metadata(zarr.DirectoryStore(str(output_path)))
    logger.info("Lagged ensemble Zarr store written with consolidated metadata")


def build_lagged_ensemble(
    input_root: Path | str,
    output_zarr: Path | str,
    max_lags: int = 2,
    exclude_dirs: Sequence[str] = (),
    patterns: Sequence[str] = ("*.grib2",),
    store_metadata: Mapping[str, Any] | None = None,
) -> Path:
    input_root_path = Path(input_root)
    output_path = Path(output_zarr)

    files = discover_files([input_root_path], patterns, exclude_dirs)
    file_index = build_file_index(files)
    matches = match_by_valid_time(file_index, max_lags)

    sample_files = [matched_files[0].path for matched_files in matches.values() if matched_files]
    if not sample_files:
        raise ValueError("No files found")

    variables = inventory_variables(sample_files)
    grid = extract_grid_info(sample_files[0])

    base_cycle = max(forecast_file.cycle_time for forecast_file in file_index)
    for matched_files in matches.values():
        for forecast_file in matched_files:
            if forecast_file.cycle_time < base_cycle:
                adjust_accumulated_for_lagged(forecast_file, base_cycle, variables)

    metadata = build_store_metadata(
        product_type="lagged_ensemble",
        system=input_root_path.name,
        run_id=base_cycle.strftime("%Y%m%d%H"),
        source="datastore.pipelines.lagged_ensemble",
    )
    if store_metadata:
        metadata.update(dict(store_metadata))

    write_zarr_store(matches, variables, grid, output_path, store_metadata=metadata)
    logger.info("Ensemble build complete")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build lagged ensemble from GRIB2 files")
    parser.add_argument("--input-root", required=True, help="Root directory of GRIB2 files")
    parser.add_argument("--output-zarr", required=True, help="Output Zarr store path")
    parser.add_argument("--max-lags", type=int, default=2, help="Maximum number of lags")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_lagged_ensemble(
        input_root=args.input_root,
        output_zarr=args.output_zarr,
        max_lags=args.max_lags,
        exclude_dirs=args.exclude_dirs,
        patterns=args.patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

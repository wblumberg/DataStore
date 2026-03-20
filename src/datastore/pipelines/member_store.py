"""Build a single-member forecast Zarr store from local GRIB2 files."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import grib2io
import numpy as np
import zarr
from numcodecs import Blosc

from ..catalog.manifest import build_store_metadata
from ..core.models import ForecastFile, GridInfo, VariableInfo
from ..grib.discovery import build_file_index, discover_files, match_by_valid_time
from ..grib.grid import extract_grid_info
from ..grib.inventory import load_variables_db
from .adjustments import adjust_accumulated_for_lagged


def adjust_accumulated_for_lagged_file(file: ForecastFile, base_cycle: datetime, variables: Sequence[VariableInfo]) -> None:
    """Backward-compatible alias for the lagged accumulation hook."""
    adjust_accumulated_for_lagged(file, base_cycle, variables)


def write_member_zarr(
    matches: dict[datetime, list[ForecastFile]],
    variables: Sequence[VariableInfo],
    grid: GridInfo,
    output_path: Path,
    member: str,
    store_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a member's data into a Zarr store with xarray-compatible metadata."""
    logger = logging.getLogger(__name__)

    logger.info("Writing Zarr store for member %s to %s", member, output_path)

    times = sorted(matches.keys())
    ny, nx = grid.nj, grid.ni

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store, overwrite=True)

    member_dtype = f"U{max(1, len(member))}"
    root.create_dataset("member", data=np.array([member], dtype=member_dtype))
    root["member"].attrs["_ARRAY_DIMENSIONS"] = ["member"]
    root.create_dataset("time", data=np.array([np.datetime64(value) for value in times]), dtype="datetime64[ns]")
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root.create_dataset("y", data=np.arange(ny), dtype=np.int32)
    root["y"].attrs["_ARRAY_DIMENSIONS"] = ["y"]
    root.create_dataset("x", data=np.arange(nx), dtype=np.int32)
    root["x"].attrs["_ARRAY_DIMENSIONS"] = ["x"]

    root.attrs.update(
        {
            "member_name": member,
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
            "description": f"Ensemble member {member} from CAM forecast data, stored in Zarr format.",
            "source": "Extracted from GRIB2 files using grib2io, processed with datastore.pipelines.member_store.",
            "history": f"Created on {datetime.utcnow().isoformat()}Z by datastore.pipelines.member_store.",
            "forecast_times": len(times),
            "time_lagged": "",
        }
    )

    metadata = dict(store_metadata or {})
    metadata.setdefault("product_type", "member")
    metadata.setdefault("created_at", datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    root.attrs.update(metadata)

    compressor = Blosc(cname="zstd", clevel=5)
    variable_names = {variable.name for variable in variables}
    for variable in variables:
        logger.info("Creating variable %s", variable.name)
        root.create_dataset(
            variable.name,
            shape=(len(times), 1, ny, nx),
            chunks=(1, 1, 256, 256),
            dtype=np.float32,
            fill_value=np.float32(np.nan),
            compressor=compressor,
        )
        root[variable.name].attrs.update(
            {
                "long_name": variable.long_name,
                "units": variable.units,
                "level1": variable.level1,
                "level2": variable.level2,
                "level_type": variable.level_type,
                "type": variable.type,
                "_ARRAY_DIMENSIONS": ["time", "member", "y", "x"],
            }
        )

    total_assignments = 0
    for time_index, valid_time in enumerate(times):
        logger.info("Processing time %s/%s: %s", time_index + 1, len(times), valid_time)
        files = matches[valid_time]
        forecast_file = files[0]
        logger.info("  Processing file %s", forecast_file.path)
        try:
            with grib2io.open(str(forecast_file.path)) as grib_file:
                for message in grib_file:
                    variable_name = message.shortName
                    if variable_name in variable_names:
                        root[variable_name][time_index, 0, :, :] = message.data
                        total_assignments += 1
        except Exception as exc:
            logger.error("Error reading %s: %s", forecast_file.path, exc)

    logger.info("Completed %s assignments", total_assignments)
    zarr.consolidate_metadata(store)
    logger.info("Member Zarr store written with consolidated metadata")


def build_member_store(
    input_roots: Sequence[Path | str],
    patterns: Sequence[str],
    exclude_dirs: Sequence[str],
    member: str,
    variables_db: Path,
    output_dir: Path,
    max_lags: int = 0,
    max_times: int | None = None,
    store_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Build one member store end-to-end and return the output path."""
    logger = logging.getLogger(__name__)

    variables = load_variables_db(variables_db)
    files = discover_files(input_roots, patterns, exclude_dirs)
    logger.info("Discovered %s files", len(files))

    file_index = build_file_index(files)
    logger.info("Indexed %s files", len(file_index))

    member_files = [forecast_file for forecast_file in file_index if forecast_file.member == member]
    logger.info("Filtered to %s files for member %s", len(member_files), member)
    if not member_files:
        raise ValueError(f"No files for member {member}")

    matches = match_by_valid_time(member_files, max_lags)
    logger.info("Matched %s valid times", len(matches))

    if max_times:
        limited_times = sorted(matches.keys())[:max_times]
        matches = {valid_time: matches[valid_time] for valid_time in limited_times}
        logger.info("Limited to %s forecast times for debugging", len(matches))

    grid = extract_grid_info(member_files[0].path)
    logger.info("Grid: %sx%s", grid.ni, grid.nj)

    base_cycle = max(forecast_file.cycle_time for forecast_file in file_index)
    for matched_files in matches.values():
        for forecast_file in matched_files:
            if forecast_file.cycle_time < base_cycle:
                adjust_accumulated_for_lagged(forecast_file, base_cycle, variables)
    logger.info("Accumulated adjustment complete")

    system_name = Path(str(input_roots[0])).name if input_roots else "unknown"
    metadata = build_store_metadata(
        product_type="member",
        system=system_name,
        run_id=base_cycle.strftime("%Y%m%d%H"),
        source="datastore.pipelines.member_store",
        extra={"member": member},
    )
    if store_metadata:
        metadata.update(dict(store_metadata))

    output_path = Path(output_dir) / f"{member}.zarr"
    write_member_zarr(matches, variables, grid, output_path, member, store_metadata=metadata)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", nargs="+", required=True, help="Input root directories")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--member", required=True, help="Ensemble member name")
    parser.add_argument("--variables-db", required=True, help="Variables database JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for Zarr files")
    parser.add_argument("--max-lags", type=int, default=0, help="Maximum lag hours")
    parser.add_argument("--max-times", type=int, help="Maximum number of forecast times to process")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s]: %(message)s")
    build_member_store(
        input_roots=args.input_root,
        patterns=args.patterns,
        exclude_dirs=args.exclude_dirs,
        member=args.member,
        variables_db=Path(args.variables_db),
        output_dir=Path(args.output_dir),
        max_lags=args.max_lags,
        max_times=args.max_times,
    )
    logging.getLogger(__name__).info("Member build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

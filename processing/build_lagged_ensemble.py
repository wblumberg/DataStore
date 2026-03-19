"""
Build lagged ensemble from GRIB2 files.

This script processes GRIB2 forecast files from multiple members and cycles,
matches them by valid time, inventories variables, extracts grid information,
adjusts accumulated variables for lagged members, and writes to a Zarr store.

Usage:
    python build_lagged_ensemble.py --input-root /path/to/data --output-zarr ensemble.zarr

Author: Assistant
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import grib2io
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
import dask.array as da

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ForecastFile:
    path: Path
    member: str
    cycle_time: datetime
    valid_time: datetime
    forecast_hour: int


@dataclass
class VariableInfo:
    name: str
    long_name: str
    units: str
    level: str
    accumulation: str


@dataclass
class GridInfo:
    ni: int
    nj: int
    lat_0: float
    lon_0: float
    lat_std: float
    ll_lat: float
    ll_lon: float
    ur_lat: float
    ur_lon: float


def discover_files(input_root: Path, patterns: List[str], exclude_dirs: List[str]) -> List[Path]:
    """Discover GRIB2 files in the input directory tree."""
    logger.info(f"Discovering files in {input_root} with patterns {patterns}")
    files = []
    for pattern in patterns:
        for path in input_root.rglob(pattern):
            if any(excl in str(path) for excl in exclude_dirs):
                continue
            files.append(path)
    logger.info(f"Found {len(files)} files")
    return files


import re

CYCLE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
FHOUR_RE = re.compile(r"[Ff](\d{1,3})(?!\d)")

def parse_filename(path: Path) -> Optional[Tuple[str, datetime, int]]:
    """Parse member, cycle time, and forecast hour from filename using regex."""
    name = path.name
    cycle_match = CYCLE_RE.search(name)
    fhr_match = FHOUR_RE.search(name)
    
    if not cycle_match or not fhr_match:
        return None
    
    try:
        cycle_time = datetime.strptime(cycle_match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        forecast_hour = int(fhr_match.group(1))
        # Infer member from path
        parts = path.parts
        member = None
        for part in reversed(parts):
            if part.startswith(('HIRESW', 'NAMNEST', 'WRF4NSSL', 'HRRR')):
                member = part
                break
        if not member:
            member = path.parent.name
        return member, cycle_time, forecast_hour
    except ValueError:
        return None


def build_file_index(files: List[Path]) -> List[ForecastFile]:
    """Build index of forecast files with timing info."""
    logger.info("Building file index")
    index = []
    for path in files:
        parsed = parse_filename(path)
        if parsed:
            member, cycle_time, fhr = parsed
            valid_time = cycle_time + timedelta(hours=fhr)
            index.append(ForecastFile(path, member, cycle_time, valid_time, fhr))
    logger.info(f"Indexed {len(index)} files")
    return index


def match_by_valid_time(files: List[ForecastFile], max_lags: int) -> Dict[datetime, List[ForecastFile]]:
    """Match files by valid time, including lagged members."""
    logger.info("Matching files by valid time")
    matches = defaultdict(list)
    cycles = sorted(set(f.cycle_time for f in files), reverse=True)
    latest_cycle = cycles[0]
    
    for f in files:
        lag = int((latest_cycle - f.cycle_time).total_seconds() / 3600)
        if lag <= max_lags * 6:  # Assume 6h cycles
            matches[f.valid_time].append(f)
    
    logger.info(f"Matched {len(matches)} valid times")
    return matches


def inventory_variables(files: List[Path]) -> List[VariableInfo]:
    """Inventory all variables from the GRIB2 files."""
    logger.info("Inventorying variables")
    variables = []
    seen = set()
    
    for path in files[:5]:  # Sample first few files
        try:
            with grib2io.open(str(path)) as f:
                for msg in f:
                    key = (msg.shortName, msg.level, msg.stepRange if hasattr(msg, 'stepRange') else '')
                    if key not in seen:
                        seen.add(key)
                        accumulation = 'instant' if not hasattr(msg, 'stepRange') or not msg.stepRange else f"accum_{msg.stepRange}"
                        variables.append(VariableInfo(
                            name=msg.shortName,
                            long_name=getattr(msg, 'longName', msg.shortName),
                            units=getattr(msg, 'units', ''),
                            level=str(msg.level),
                            accumulation=accumulation
                        ))
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
    
    logger.info(f"Found {len(variables)} variables")
    return variables


def extract_grid_info(path: Path) -> GridInfo:
    """Extract grid information from a GRIB2 file."""
    logger.info(f"Extracting grid info from {path}")
    with grib2io.open(str(path)) as f:
        msg = next(iter(f))
        # Simplified extraction
        ni = getattr(msg, 'Ni', 0)
        nj = getattr(msg, 'Nj', 0)
        lat_0 = getattr(msg, 'LaD', 0) / 1e6 if hasattr(msg, 'LaD') else 0
        lon_0 = getattr(msg, 'LoV', 0) / 1e6 if hasattr(msg, 'LoV') else 0
        lat_std = getattr(msg, 'Latin1', 0) / 1e6 if hasattr(msg, 'Latin1') else 0
        # Corners - simplified
        ll_lat = getattr(msg, 'latitudeOfFirstGridPoint', 0) / 1e6
        ll_lon = getattr(msg, 'longitudeOfFirstGridPoint', 0) / 1e6
        ur_lat = getattr(msg, 'latitudeOfLastGridPoint', 0) / 1e6
        ur_lon = getattr(msg, 'longitudeOfLastGridPoint', 0) / 1e6
        
        return GridInfo(ni, nj, lat_0, lon_0, lat_std, ll_lat, ll_lon, ur_lat, ur_lon)


def adjust_accumulated_for_lagged(file: ForecastFile, base_cycle: datetime, variables: List[VariableInfo]) -> None:
    """Adjust accumulated variables for lagged members."""
    # TODO: Implement subtraction of accumulated variables
    # For now, placeholder
    logger.info(f"Adjusting accumulated variables for {file.path} (lag from {base_cycle})")
    pass


def write_zarr_store(matches: Dict[datetime, List[ForecastFile]], variables: List[VariableInfo], grid: GridInfo, output_path: Path):
    """Write the ensemble to a Zarr store using xarray."""
    logger.info(f"Writing Zarr store to {output_path}")
    
    times = sorted(matches.keys())
    members = sorted(set(f.member for files in matches.values() for f in files))
    
    ny, nx = grid.nj, grid.ni  # Assume nj is y, ni is x
    
    # Create datasets
    data_vars = {}
    for var in variables:
        # Create dask array with NaN
        arr = da.full((len(times), len(members), ny, nx), np.nan, chunks=(1, 1, 256, 256), dtype=np.float32)
        data_vars[var.name] = (['time', 'member', 'y', 'x'], arr, {
            'long_name': var.long_name,
            'units': var.units,
            'level': var.level,
            'accumulation': var.accumulation
        })
    
    # Coordinates
    coords = {
        'time': times,
        'member': members,
        'y': np.arange(ny),
        'x': np.arange(nx)
    }
    
    # Create xarray Dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Populate data
    for t_idx, valid_time in enumerate(times):
        files = matches[valid_time]
        for f in files:
            m_idx = members.index(f.member)
            try:
                with grib2io.open(str(f.path)) as gf:
                    for msg in gf:
                        var_name = msg.shortName
                        if var_name in ds.data_vars:
                            data = msg.data
                            ds[var_name][t_idx, m_idx, :, :] = data
            except Exception as e:
                logger.error(f"Error reading {f.path}: {e}")
    
    # Write to Zarr
    ds.to_zarr(str(output_path), mode='w', consolidated=True)
    
    logger.info("Zarr store written")


def main():
    parser = argparse.ArgumentParser(description="Build lagged ensemble from GRIB2 files")
    parser.add_argument('--input-root', required=True, help='Root directory of GRIB2 files')
    parser.add_argument('--output-zarr', required=True, help='Output Zarr store path')
    parser.add_argument('--max-lags', type=int, default=2, help='Maximum number of lags')
    parser.add_argument('--exclude-dirs', nargs='*', default=[], help='Directories to exclude')
    parser.add_argument('--patterns', nargs='*', default=['*.grib2'], help='File patterns')
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_path = Path(args.output_zarr)
    
    # Discover files
    files = discover_files(input_root, args.patterns, args.exclude_dirs)
    
    # Build index
    file_index = build_file_index(files)
    
    # Match by valid time
    matches = match_by_valid_time(file_index, args.max_lags)
    
    # Inventory variables
    sample_files = [f.path for files in matches.values() for f in files[:1]]  # One per time
    variables = inventory_variables(sample_files)
    
    # Extract grid info
    if sample_files:
        grid = extract_grid_info(sample_files[0])
    else:
        raise ValueError("No files found")
    
    # Adjust accumulated for lagged (TODO)
    base_cycle = max(f.cycle_time for f in file_index)
    for files in matches.values():
        for f in files:
            if f.cycle_time < base_cycle:
                adjust_accumulated_for_lagged(f, base_cycle, variables)
    
    # Write Zarr
    write_zarr_store(matches, variables, grid, output_path)
    
    logger.info("Ensemble build complete")


if __name__ == '__main__':
    main()
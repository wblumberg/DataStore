"""
Build Zarr store for a single ensemble member.

To be run after downloading all the GRIB2 files and running inventory_variables_db.py to create the variables database JSON file.
For each member, run this script to create a Zarr store with the member's data.  This will be used later for loading the ensemble and computing statistics.

Once the member Zarrs are built, we can load them with xarray and compute ensemble statistics functions in ens_postprocessing.py.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import grib2io
import numpy as np
import zarr
from numcodecs import Blosc

from adjust_accumulated import adjust_accumulated_for_lagged
from inventory_variables import VariableInfo
from discover_grib import ForecastFile, build_file_index, discover_files, match_by_valid_time
from extract_grid import extract_grid_info

def load_variables_db(variables_file: Path) -> List[VariableInfo]:
    """Load variables from JSON database."""
    with open(variables_file) as f:
        var_dicts = json.load(f)
    return [
        VariableInfo(
            name=d['name'],
            long_name=d['long_name'],
            units=d['units'],
            level1=d['level1'],
            level2=d['level2'],
            level_type=d['level_type'],
            type=d['type']
        )
        for d in var_dicts
    ]

def adjust_accumulated_for_lagged_file(f: ForecastFile, base_cycle: datetime, variables: List[VariableInfo]):
    """Placeholder: Adjust accumulated variables for a lagged file."""
    # TODO: Implement subtraction logic
    pass

def write_member_zarr(matches: Dict[datetime, List[ForecastFile]], variables: List[VariableInfo], grid, output_path: Path, member: str):
    """Write the member's data to a Zarr store."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Writing Zarr store for member {member} to {output_path}")
    
    times = sorted(matches.keys())
    ny, nx = grid.nj, grid.ni
    
    # Create Zarr store
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store, overwrite=True)
    
    # Coordinates
    member_dtype = f'U{max(1, len(member))}'
    root.create_dataset('member', data=np.array([member], dtype=member_dtype))
    root['member'].attrs['_ARRAY_DIMENSIONS'] = ['member']
    root.create_dataset('time', data=np.array([np.datetime64(t) for t in times]), dtype='datetime64[ns]')
    root['time'].attrs['_ARRAY_DIMENSIONS'] = ['time']
    root.create_dataset('y', data=np.arange(ny), dtype=np.int32)
    root['y'].attrs['_ARRAY_DIMENSIONS'] = ['y']
    root.create_dataset('x', data=np.arange(nx), dtype=np.int32)
    root['x'].attrs['_ARRAY_DIMENSIONS'] = ['x']
    root.attrs['member_name'] = member
    root.attrs['grid_type'] = 'lambert'
    root.attrs['ni'] = grid.ni
    root.attrs['nj'] = grid.nj
    root.attrs['lon_0'] = grid.lon_0
    root.attrs['lat_0'] = grid.lat_0
    root.attrs['lat_std'] = grid.lat_std
    root.attrs['ll_lat'] = grid.ll_lat
    root.attrs['ll_lon'] = grid.ll_lon
    root.attrs['ur_lat'] = grid.ur_lat
    root.attrs['ur_lon'] = grid.ur_lon
    root.attrs['description'] = f"Ensemble member {member} from CAM forecast data, stored in Zarr format."
    root.attrs['source'] = "Extracted from GRIB2 files using grib2io, processed with custom Python script."
    root.attrs['history'] = f"Created on {datetime.utcnow().isoformat()}Z by build_member_zarr.py script."
    root.attrs['forecast_times'] = len(times)
    root.attrs['time_lagged'] = "" ## Implement this
    
    # Variables
    compressor = Blosc(cname='zstd', clevel=5)
    var_names = {v.name for v in variables}
    for var in variables:
        logger.info(f"Creating variable {var.name}")
        root.create_dataset(
            var.name,
            shape=(len(times), 1, ny, nx),
            chunks=(1, 1, 256, 256),
            dtype=np.float32,
            fill_value=np.float32(np.nan),
            compressor=compressor
        )
        root[var.name].attrs.update({
            'long_name': var.long_name,
            'units': var.units,
            'level1': var.level1,
            'level2': var.level2,
            'level_type': var.level_type,
            'type': var.type,
            '_ARRAY_DIMENSIONS': ['time', 'member', 'y', 'x']
        })
    
    # Populate data
    total_assignments = 0
    for t_idx, valid_time in enumerate(times):
        logger.info(f"Processing time {t_idx+1}/{len(times)}: {valid_time}")
        files = matches[valid_time]
        # Should be only one file per time per member, but in case multiple, take the first
        f = files[0]  # Assuming matches are filtered to this member
        logger.info(f"  Processing file {f.path}")
        try:
            with grib2io.open(str(f.path)) as gf:
                for msg in gf:
                    var_name = msg.shortName
                    if var_name in var_names:
                        data = msg.data
                        root[var_name][t_idx, 0, :, :] = data
                        total_assignments += 1
        except Exception as e:
            logger.error(f"Error reading {f.path}: {e}")
    
    logger.info(f"Completed {total_assignments} assignments")

    # Write consolidated metadata so xarray can open with consolidated=True.
    zarr.consolidate_metadata(store)
    logger.info("Member Zarr store written with consolidated metadata")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', nargs='+', required=True, help='Input root directories')
    parser.add_argument('--patterns', nargs='*', default=['*.grib2'], help='File patterns')
    parser.add_argument('--exclude-dirs', nargs='*', default=[], help='Directories to exclude')
    parser.add_argument('--member', required=True, help='Ensemble member name')
    parser.add_argument('--variables-db', required=True, help='Variables database JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory for Zarr files')
    parser.add_argument('--max-lags', type=int, default=0, help='Maximum lag hours')
    parser.add_argument('--max-times', type=int, help='Maximum number of forecast times to process (for debugging)')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(levelname)s]: %(message)s')

    # Load variables
    variables_file = Path(args.variables_db)
    variables = load_variables_db(variables_file)
    var_names = {v.name for v in variables}

    # Discover files
    files = discover_files(args.input_root, args.patterns, args.exclude_dirs)
    logger = logging.getLogger(__name__)
    logger.info(f"Discovered {len(files)} files")

    # Build index
    file_index = build_file_index(files)
    logger.info(f"Indexed {len(file_index)} files")

    # Filter to this member
    member_files = [f for f in file_index if f.member == args.member]
    logger.info(f"Filtered to {len(member_files)} files for member {args.member}")

    # Match by valid time
    matches = match_by_valid_time(member_files, args.max_lags)
    logger.info(f"Matched {len(matches)} valid times")

    # Limit to max_times for debugging
    if args.max_times:
        sorted_times = sorted(matches.keys())[:args.max_times]
        matches = {t: matches[t] for t in sorted_times}
        logger.info(f"Limited to {len(matches)} forecast times for debugging")

    # Extract grid (assume same for member)
    if member_files:
        grid = extract_grid_info(member_files[0].path)
        logger.info(f"Grid: {grid.ni}x{grid.nj}")
    else:
        raise ValueError(f"No files for member {args.member}")

    # Adjust accumulated for lagged (placeholder)
    base_cycle = max(f.cycle_time for f in file_index)
    for files in matches.values():
        for f in files:
            if f.cycle_time < base_cycle:
                adjust_accumulated_for_lagged(f, base_cycle, variables)
    logger.info("Accumulated adjustment complete - NOT IMPLEMENTED")

    # Write Zarr
    output_path = Path(args.output_dir) / f"{args.member}.zarr"
    write_member_zarr(matches, variables, grid, output_path, args.member)
    logger.info("Member build complete")

if __name__ == '__main__':
    main()
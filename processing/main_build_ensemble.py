"""
Main script for building ensemble: inventory variables, then build member Zarrs.
Run once for inventory, then run per member for building.
"""

import json
import subprocess
import sys
from pathlib import Path

from discover_grib import build_file_index, discover_files

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', nargs='+', required=True, help='Input root directories')
    parser.add_argument('--patterns', nargs='*', default=['*.grib2'], help='File patterns')
    parser.add_argument('--exclude-dirs', nargs='*', default=[], help='Directories to exclude')
    parser.add_argument('--variables-db', default='variables.json', help='Variables database file')
    parser.add_argument('--member', help='Specific member to build (if not provided, run inventory and list members)')
    parser.add_argument('--output-dir', default='.', help='Output directory for member Zarrs')
    parser.add_argument('--max-lags', type=int, default=4, help='Maximum lag hours')
    parser.add_argument('--max-times', type=int, help='Maximum number of forecast times to process (for debugging)')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    variables_db = Path(args.variables_db)

    if not variables_db.exists() or args.member is None:
        # Run inventory
        logger.info("Running variable inventory")
        cmd = [
            sys.executable, 'inventory_variables_db.py',
            '--input-root'] + args.input_root + [
            '--patterns'] + args.patterns + [
            '--exclude-dirs'] + args.exclude_dirs + [
            '--output-variables', str(variables_db)
        ]
        subprocess.run(cmd, check=True)

        # Discover members
        files = discover_files(args.input_root, args.patterns, args.exclude_dirs)
        file_index = build_file_index(files)
        members = sorted(set(f.member for f in file_index))
        logger.info(f"Found members: {members}")

        if args.member is None:
            logger.info("Run with --member <name> to build each member's Zarr")
            return

    # Build specific member
    member = args.member
    output_zarr = Path(args.output_dir) / f"{member}.zarr"
    logger.info(f"Building Zarr for member {member}")

    cmd = [
        sys.executable, 'build_member_zarr.py',
        '--input-root'] + args.input_root + [
        '--patterns'] + args.patterns + [
        '--exclude-dirs'] + args.exclude_dirs + [
        '--member', member,
        '--variables-db', str(variables_db),
        '--output-zarr', str(output_zarr),
        '--max-lags', str(args.max_lags)
    ]
    if args.max_times:
        cmd += ['--max-times', str(args.max_times)]
    subprocess.run(cmd, check=True)
    logger.info(f"Completed member {member}")

if __name__ == '__main__':
    main()
"""Inventory variables and build per-member Zarr stores."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ..grib.discovery import build_file_index, discover_files
from .inventory_db import inventory_variables_database
from .member_store import build_member_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", nargs="+", required=True, help="Input root directories")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--variables-db", default="variables.json", help="Variables database file")
    parser.add_argument("--member", help="Specific member to build")
    parser.add_argument("--output-dir", default=".", help="Output directory for member Zarrs")
    parser.add_argument("--max-lags", type=int, default=4, help="Maximum lag hours")
    parser.add_argument("--max-times", type=int, help="Maximum number of forecast times to process")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    variables_db = Path(args.variables_db)

    if not variables_db.exists() or args.member is None:
        logger.info("Running variable inventory")
        inventory_variables_database(
            input_roots=args.input_root,
            patterns=args.patterns,
            exclude_dirs=args.exclude_dirs,
            output_variables=variables_db,
        )

        files = discover_files(args.input_root, args.patterns, args.exclude_dirs)
        file_index = build_file_index(files)
        members = sorted({forecast_file.member for forecast_file in file_index})
        logger.info("Found members: %s", members)

        if args.member is None:
            logger.info("Run with --member <name> to build each member's Zarr")
            return 0

    logger.info("Building Zarr for member %s", args.member)
    build_member_store(
        input_roots=args.input_root,
        patterns=args.patterns,
        exclude_dirs=args.exclude_dirs,
        member=args.member,
        variables_db=variables_db,
        output_dir=Path(args.output_dir),
        max_lags=args.max_lags,
        max_times=args.max_times,
    )
    logger.info("Completed member %s", args.member)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

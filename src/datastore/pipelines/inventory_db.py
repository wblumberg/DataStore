"""Build a reusable JSON variable inventory from GRIB2 files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..grib.discovery import discover_files
from ..grib.inventory import inventory_variables, save_variables_db


def inventory_variables_database(
    input_roots: Sequence[Path | str],
    patterns: Sequence[str],
    exclude_dirs: Sequence[str],
    output_variables: Path,
    sample_limit: int = 30,
) -> list:
    files = discover_files(input_roots, patterns, exclude_dirs)
    sample_files = files[:sample_limit]
    variables = inventory_variables(sample_files, sample_limit=sample_limit)
    save_variables_db(variables, output_variables)
    return variables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", nargs="+", required=True, help="Input root directories")
    parser.add_argument("--patterns", nargs="*", default=["*.grib2"], help="File patterns")
    parser.add_argument("--exclude-dirs", nargs="*", default=[], help="Directories to exclude")
    parser.add_argument("--output-variables", required=True, help="Output JSON file for variables")
    parser.add_argument("--sample-limit", type=int, default=30, help="Maximum sample files to inventory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output_variables)
    variables = inventory_variables_database(
        input_roots=args.input_root,
        patterns=args.patterns,
        exclude_dirs=args.exclude_dirs,
        output_variables=output_path,
        sample_limit=args.sample_limit,
    )
    print(f"Inventory complete. {len(variables)} variables saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

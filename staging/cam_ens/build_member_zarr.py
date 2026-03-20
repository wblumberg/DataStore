"""Compatibility wrapper for datastore.pipelines.member_store."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.pipelines.member_store import adjust_accumulated_for_lagged_file, build_member_store, load_variables_db, main, write_member_zarr

__all__ = [
    "adjust_accumulated_for_lagged_file",
    "build_member_store",
    "load_variables_db",
    "main",
    "write_member_zarr",
]


if __name__ == '__main__':
    raise SystemExit(main())
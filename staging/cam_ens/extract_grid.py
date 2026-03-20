"""Compatibility wrapper for datastore.grib.grid."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.grib.grid import GridInfo, extract_grid_info

__all__ = ["GridInfo", "extract_grid_info"]
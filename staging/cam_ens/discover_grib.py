"""Compatibility wrapper for datastore.grib.discovery."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.grib.discovery import CYCLE_RE, FHOUR_RE, ForecastFile, build_file_index, discover_files, match_by_valid_time, parse_filename

__all__ = [
    "CYCLE_RE",
    "FHOUR_RE",
    "ForecastFile",
    "build_file_index",
    "discover_files",
    "match_by_valid_time",
    "parse_filename",
]
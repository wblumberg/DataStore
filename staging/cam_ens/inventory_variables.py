"""Compatibility wrapper for datastore.grib.inventory."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.grib.inventory import VariableInfo, get_variable_type, inventory_variables, load_variables_db, save_variables_db

__all__ = [
    "VariableInfo",
    "get_variable_type",
    "inventory_variables",
    "load_variables_db",
    "save_variables_db",
]
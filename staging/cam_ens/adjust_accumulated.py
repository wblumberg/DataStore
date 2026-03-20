"""Compatibility wrapper for datastore.pipelines.adjustments."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.pipelines.adjustments import adjust_accumulated_for_lagged

__all__ = ["adjust_accumulated_for_lagged"]
"""Compatibility wrapper for datastore.sources.remote.gefs_dynamical."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from datastore.sources.remote.gefs_dynamical import SOURCE, open_latest_dataset, url

__all__ = ["SOURCE", "open_latest_dataset", "url"]

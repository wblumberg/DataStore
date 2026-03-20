"""Compatibility wrapper for datastore.sources.space.aurora_ovation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from datastore.sources.space.aurora_ovation import AURORA_OVATION_URL, fetch_latest_aurora_json, main

url = AURORA_OVATION_URL


if __name__ == "__main__":
	raise SystemExit(main())


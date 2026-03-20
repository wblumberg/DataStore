"""Compatibility wrapper for datastore.sources.tropical.tracks."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.sources.tropical.tracks import *


if __name__ == "__main__":
    raise SystemExit(main())



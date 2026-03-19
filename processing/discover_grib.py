"""
Discover and match GRIB2 files by valid time for ensemble building.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CYCLE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
FHOUR_RE = re.compile(r"[Ff](\d{1,3})(?=\.)")

@dataclass
class ForecastFile:
    path: Path
    member: str
    name: str
    cycle_time: datetime
    valid_time: datetime
    forecast_hour: int

def discover_files(input_roots: List[Path], patterns: List[str], exclude_dirs: List[str]) -> List[Path]:
    """Discover GRIB2 files in multiple input directory trees."""
    files = []
    for root in input_roots:
        root_path = Path(root)
        for pattern in patterns:
            for path in root_path.rglob(pattern):
                if any(excl in str(path) for excl in exclude_dirs):
                    continue
                files.append(path)
    return files

def parse_filename(path: Path) -> Optional[Tuple[str, datetime, int]]:
    """Parse member, cycle time, and forecast hour from filename using regex."""
    name = path.name
    cycle_match = CYCLE_RE.search(name)
    fhr_match = FHOUR_RE.search(name)

    if not cycle_match or not fhr_match:
        return None
    
    try:
        cycle_time = datetime.strptime(cycle_match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        forecast_hour = int(fhr_match.group(1))
        # Infer member from path
        parts = path.parts
        member = None
        for part in reversed(parts):
            if part.startswith(('HIRESW', 'NAMNEST', 'WRF4NSSL', 'HRRR')):
                member = part
                break
        if not member:
            member = path.parent.name
        return member, cycle_time, forecast_hour
    except ValueError:
        return None

def build_file_index(files: List[Path]) -> List[ForecastFile]:
    """Build index of forecast files with timing info."""
    index = []
    for path in files:
        parsed = parse_filename(path)
        if parsed:
            member, cycle_time, fhr = parsed
            valid_time = cycle_time + timedelta(hours=fhr)
            index.append(ForecastFile(path, member, path.name, cycle_time, valid_time, fhr))
    return index

def match_by_valid_time(files: List[ForecastFile], max_lags: int) -> Dict[datetime, List[ForecastFile]]:
    """Match files by valid time, including lagged members."""
    matches = defaultdict(list)
    cycles = sorted(set(f.cycle_time for f in files), reverse=True)
    latest_cycle = cycles[0]
    
    print(matches, cycles, latest_cycle)
    
    for f in files:
        lag_hours = int((latest_cycle - f.cycle_time).total_seconds() / 3600)
        if lag_hours <= max_lags * 6:  # Assume 6h cycles
            matches[f.valid_time].append(f)
    
    return matches
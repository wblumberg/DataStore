"""
Extract grid information from GRIB2 files.
"""

from dataclasses import dataclass
from pathlib import Path

import grib2io

@dataclass
class GridInfo:
    ni: int
    nj: int
    lat_0: float
    lon_0: float
    lat_std: float
    ll_lat: float
    ll_lon: float
    ur_lat: float
    ur_lon: float

def extract_grid_info(path: Path) -> GridInfo:
    """Extract grid information from a GRIB2 file."""
    with grib2io.open(str(path)) as f:
        for msg in f:
            # Try metadata first
            ni = getattr(msg, 'Ni', None) or getattr(msg, 'Nx', None)
            nj = getattr(msg, 'Nj', None) or getattr(msg, 'Ny', None)
            
            # Fallback to data shape
            if ni is None or nj is None or ni == 0 or nj == 0:
                try:
                    data = msg.data
                    if hasattr(data, 'shape') and len(data.shape) == 2:
                        nj_data, ni_data = data.shape
                        ni = ni or ni_data
                        nj = nj or nj_data
                except Exception:
                    continue
            
            if ni and nj and ni > 0 and nj > 0:
                break
        
        # Ensure valid values
        ni = ni or 0
        nj = nj or 0
        
        lat_0 = getattr(msg, 'LaD', 0) / 1e6 if hasattr(msg, 'LaD') else 0
        lon_0 = getattr(msg, 'LoV', 0) / 1e6 if hasattr(msg, 'LoV') else 0
        lat_std = getattr(msg, 'Latin1', 0) / 1e6 if hasattr(msg, 'Latin1') else 0
        # Corners - simplified
        ll_lat = getattr(msg, 'latitudeOfFirstGridPoint', 0) / 1e6
        ll_lon = getattr(msg, 'longitudeOfFirstGridPoint', 0) / 1e6
        ur_lat = getattr(msg, 'latitudeOfLastGridPoint', 0) / 1e6
        ur_lon = getattr(msg, 'longitudeOfLastGridPoint', 0) / 1e6
        
        return GridInfo(ni, nj, lat_0, lon_0, lat_std, ll_lat, ll_lon, ur_lat, ur_lon)
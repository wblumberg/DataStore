"""
Write ensemble data to Zarr store.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import grib2io
import numpy as np
import zarr
from numcodecs import Blosc

from discover_grib import ForecastFile
from extract_grid import GridInfo
from inventory_variables import VariableInfo

def write_zarr_store(matches: Dict[datetime, List[ForecastFile]], variables: List[VariableInfo], grid: GridInfo, output_path: Path, chunk_size: int = 8):
    """Write the ensemble to a Zarr store using xarray for robust metadata and chunked writing."""
    import logging
    import xarray as xr
    logger = logging.getLogger(__name__)

    logger.info(f"Writing Zarr store to {output_path}")


    times = sorted(matches.keys())
    members = sorted(set(f.member for files in matches.values() for f in files))
    ny, nx = grid.nj, grid.ni
    
    # TODO: Write grid metadata to Zarr attributes or a separate dataset as needed.  Need this to host the grid information for later use in plotting.

    # Prepare coordinate arrays
    time_arr = np.array([np.datetime64(t) for t in times])
    member_arr = np.array(members, dtype='U20')
    y_arr = np.arange(ny, dtype=np.int32)
    x_arr = np.arange(nx, dtype=np.int32)


    # Step 1: Fill numpy arrays for each variable
    var_arrays = {}
    for var in variables:
        name = var.name.replace(' ', '')
        var_arrays[name] = np.full((len(times), len(members), ny, nx), np.nan, dtype=np.float32)

    # Step 2: Populate arrays in chunks
    for t_start in range(0, len(times), chunk_size):
        t_end = min(t_start + chunk_size, len(times))
        logger.info(f"Processing times {t_start+1}-{t_end} of {len(times)}")
        for t_idx in range(t_start, t_end):
            valid_time = times[t_idx]
            files = matches[valid_time]
            for f in files:
                m_idx = members.index(f.member)
                try:
                    with grib2io.open(str(f.path)) as gf:
                        for msg in gf:
                            var_name = msg.shortName.replace(' ', '')
                            if var_name in var_arrays:
                                data = msg.data
                                var_arrays[var_name][t_idx, m_idx, :, :] = data
                except Exception as e:
                    logger.error(f"Error reading {f.path}: {e}")

    # Step 3: Build DataArrays and Dataset in one go
    data_vars = {}
    for var in variables:
        name = var.name.replace(' ', '')
        arr = var_arrays[name]
        da = xr.DataArray(
            arr,
            dims=['time', 'member', 'y', 'x'],
            attrs={
                'long_name': var.long_name,
                'units': var.units,
                'level': getattr(var, 'level', ''),
                'type': var.type,
            }
        )
        data_vars[name] = da

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': ('time', time_arr),
            'member': ('member', member_arr),
            'y': ('y', y_arr),
            'x': ('x', x_arr),
        }
        attrs = {
            'grid_type': grid.grid_type,
            'ni': grid.ni,
            'nj': grid.nj,
            'lon_0': grid.lon0,
            'lat_0': grid.lat0,
            'lat_std': grid.lon1,
            'll_lat': grid.ll_lat,
            'll_lon': grid.ll_lon,
            'ur_lat': grid.ur_lat,
            'ur_lon': grid.ur_lon,
        }
    )

    # Write to Zarr using xarray for robust metadata
    encoding = {}
    for var in variables:
        name = var.name.replace(' ', '')
        if name in ds.data_vars:
            encoding[name] = {
                "chunks": (1, 1, 256, 256),
                "compressor": Blosc(cname='zstd', clevel=5),
            }

    ds.to_zarr(
        str(output_path),
        mode="w",
        consolidated=True,
        encoding=encoding,
        zarr_version=2,
    )

    # Defensive consolidation for filesystems where metadata visibility can lag.
    store = zarr.DirectoryStore(str(output_path))
    zarr.consolidate_metadata(store)
    logger.info("Zarr store written with xarray and consolidated metadata")
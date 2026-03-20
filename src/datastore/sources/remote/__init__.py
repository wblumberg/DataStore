"""Remote model source adapters."""

from .common import RemoteZarrSource, open_remote_zarr
from .ecens_dynamical import SOURCE as ECENS_DYNAMICAL_SOURCE
from .gefs_dynamical import SOURCE as GEFS_DYNAMICAL_SOURCE
from .gfs_dynamical import SOURCE as GFS_DYNAMICAL_SOURCE
from .gefsnssl_ensemble import load_gefs_ensemble

__all__ = [
    "ECENS_DYNAMICAL_SOURCE",
    "GEFS_DYNAMICAL_SOURCE",
    "GFS_DYNAMICAL_SOURCE",
    "RemoteZarrSource",
    "load_gefs_ensemble",
    "open_remote_zarr",
]

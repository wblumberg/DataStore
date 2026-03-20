"""Pipeline entrypoints for datastore workflows."""

__all__ = [
    "build_lagged_ensemble",
    "build_member_store",
    "load_member_ensemble",
    "inventory_variables_database",
    "postprocess_member_stores",
    "write_lagged_ensemble_zarr",
    "write_member_zarr",
]


def inventory_variables_database(*args, **kwargs):
    from .inventory_db import inventory_variables_database as implementation

    return implementation(*args, **kwargs)


def build_member_store(*args, **kwargs):
    from .member_store import build_member_store as implementation

    return implementation(*args, **kwargs)


def write_member_zarr(*args, **kwargs):
    from .member_store import write_member_zarr as implementation

    return implementation(*args, **kwargs)


def build_lagged_ensemble(*args, **kwargs):
    from .lagged_ensemble import build_lagged_ensemble as implementation

    return implementation(*args, **kwargs)


def write_lagged_ensemble_zarr(*args, **kwargs):
    from .lagged_ensemble import write_zarr_store as implementation

    return implementation(*args, **kwargs)


def load_member_ensemble(*args, **kwargs):
    from .postprocess_ensemble import load_member_ensemble as implementation

    return implementation(*args, **kwargs)


def postprocess_member_stores(*args, **kwargs):
    from .postprocess_ensemble import postprocess_member_stores as implementation

    return implementation(*args, **kwargs)

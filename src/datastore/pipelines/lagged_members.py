"""Lazy loading helpers for lagged member ensembles stored as per-member Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Mapping, Sequence

import xarray as xr


CYCLE_TOKEN_RE = re.compile(r"(?<!\d)(\d{8})[._-]?(\d{2})(?!\d)")


@dataclass(frozen=True)
class LaggedMemberStore:
    path: Path
    cycle_token: str
    cycle_time: datetime
    lag_hours: int


def discover_member_zarrs(
    inputs: Sequence[Path | str] | Path | str,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Resolve one or more store paths or directories into member-level Zarr stores."""
    raw_inputs = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    if not raw_inputs:
        return []

    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw in raw_inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.suffix == ".zarr":
            if path not in seen:
                resolved.append(path)
                seen.add(path)
            continue

        if not path.is_dir():
            raise ValueError(f"Expected a Zarr store or directory, got: {path}")

        iterator = path.rglob("*.zarr") if recursive else path.glob("*.zarr")
        for store in sorted(item for item in iterator if item.is_dir()):
            if store in seen:
                continue
            resolved.append(store)
            seen.add(store)

    return resolved


def load_lagged_member_ensemble(
    inputs: Sequence[Path | str] | Path | str,
    *,
    latest_cycle: str | datetime | None = None,
    variables: Sequence[str] | str | None = None,
    chunks: Mapping[str, int] | str | None = "auto",
    join: str = "inner",
    recursive: bool = False,
    consolidated: bool | None = None,
    member_dim: str = "member",
) -> xr.Dataset:
    """Open per-member stores lazily and concatenate them into one lagged ensemble dataset."""
    store_paths = discover_member_zarrs(inputs, recursive=recursive)
    if not store_paths:
        raise ValueError("No member Zarr stores were found")

    selected_variables = _normalize_variables(variables)

    store_cycles = [(path, *_infer_cycle_from_store_path(path)) for path in store_paths]

    if latest_cycle is None:
        latest_cycle_time = max(cycle_time for _, _, cycle_time in store_cycles)
        latest_cycle_token = max(store_cycles, key=lambda item: item[2])[1]
    else:
        latest_cycle_token, latest_cycle_time = _parse_cycle_token(latest_cycle)

    lagged_stores = [
        LaggedMemberStore(
            path=path,
            cycle_token=cycle_token,
            cycle_time=cycle_time,
            lag_hours=_lag_hours(latest_cycle_time, cycle_time),
        )
        for path, cycle_token, cycle_time in store_cycles
    ]

    if any(store.lag_hours < 0 for store in lagged_stores):
        raise ValueError("latest_cycle is older than one or more source stores")

    lagged_stores.sort(key=lambda store: (store.lag_hours, store.path.stem, str(store.path)))

    datasets = [
        _open_lagged_store(
            store,
            variables=selected_variables,
            chunks=chunks,
            consolidated=consolidated,
            member_dim=member_dim,
        )
        for store in lagged_stores
    ]

    ensemble = xr.concat(datasets, dim=member_dim, join=join, combine_attrs="drop_conflicts")
    ensemble.attrs.update(
        {
            "latest_cycle": latest_cycle_token,
            "member_dim": member_dim,
            "join_strategy": join,
            "lagged_member_store_count": len(lagged_stores),
        }
    )
    if selected_variables is not None:
        ensemble.attrs["selected_variables"] = list(selected_variables)
    return ensemble


def _open_lagged_store(
    store: LaggedMemberStore,
    *,
    variables: tuple[str, ...] | None,
    chunks: Mapping[str, int] | str | None,
    consolidated: bool | None,
    member_dim: str,
) -> xr.Dataset:
    consolidated_flag = consolidated
    if consolidated_flag is None:
        consolidated_flag = (store.path / ".zmetadata").exists()

    ds = xr.open_zarr(str(store.path), chunks=chunks, consolidated=consolidated_flag)
    if variables is not None:
        missing = [name for name in variables if name not in ds.data_vars]
        if missing:
            missing_names = ", ".join(missing)
            available_names = ", ".join(sorted(ds.data_vars))
            raise KeyError(
                f"Store {store.path} is missing requested data variable(s): {missing_names}. "
                f"Available data variables: {available_names}"
            )
        ds = ds[list(variables)]

    base_members = _member_values(ds, store.path, member_dim=member_dim)

    if member_dim not in ds.dims:
        ds = ds.expand_dims({member_dim: base_members})

    lagged_members = [f"{member}_lag{store.lag_hours:02d}" for member in base_members]
    count = len(lagged_members)

    return ds.assign_coords(
        {
            member_dim: lagged_members,
            "base_member": (member_dim, base_members),
            "source_cycle": (member_dim, [store.cycle_token] * count),
            "lag_hours": (member_dim, [store.lag_hours] * count),
            "source_store": (member_dim, [str(store.path)] * count),
        }
    )


def _member_values(ds: xr.Dataset, path: Path, *, member_dim: str) -> list[str]:
    if member_dim not in ds.dims:
        return [path.stem]

    size = ds.sizes[member_dim]
    if member_dim in ds.coords:
        values = ds.coords[member_dim].values.tolist()
        if not isinstance(values, list):
            values = [values]
        members = [str(value) for value in values]
        if len(members) == size:
            return members

    if size == 1:
        return [path.stem]

    return [f"{path.stem}_{index:03d}" for index in range(size)]


def _normalize_variables(variables: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if variables is None:
        return None

    raw_variables = [variables] if isinstance(variables, str) else list(variables)
    names = [str(name).strip() for name in raw_variables if str(name).strip()]
    if not names:
        raise ValueError("variables must contain at least one non-empty variable name")

    return tuple(dict.fromkeys(names))


def _parse_cycle_token(value: str | datetime) -> tuple[str, datetime]:
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y%m%d.%H"), dt

    text = str(value).strip()
    match = CYCLE_TOKEN_RE.search(text)
    if match:
        token = f"{match.group(1)}.{match.group(2)}"
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return token, dt

    compact = re.sub(r"[^0-9]", "", text)
    if len(compact) >= 10 and compact[:10].isdigit():
        dt = datetime.strptime(compact[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y%m%d.%H"), dt

    raise ValueError(f"Could not parse cycle token from {value!r}")


def _infer_cycle_from_store_path(path: Path) -> tuple[str, datetime]:
    for parent in path.parents:
        match = CYCLE_TOKEN_RE.search(parent.name)
        if not match:
            continue
        token = f"{match.group(1)}.{match.group(2)}"
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return token, dt

    raise ValueError(f"Could not infer cycle token from store path {path}")


def _lag_hours(latest_cycle: datetime, cycle_time: datetime) -> int:
    return int((latest_cycle - cycle_time).total_seconds() / 3600.0)


__all__ = [
    "LaggedMemberStore",
    "discover_member_zarrs",
    "load_lagged_member_ensemble",
]
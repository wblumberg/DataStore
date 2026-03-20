"""Schema and metadata contract validation for datastore products."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr


TIME_DIM_CANDIDATES: tuple[str, ...] = ("time", "valid_time")
MEMBER_DIM_CANDIDATES: tuple[str, ...] = ("member", "ensemble", "realization", "number")
Y_DIM_CANDIDATES: tuple[str, ...] = ("y", "lat", "latitude", "nj")
X_DIM_CANDIDATES: tuple[str, ...] = ("x", "lon", "longitude", "ni")


@dataclass
class ContractReport:
    """Validation report for one dataset or store."""

    product_type: str
    target: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_type": self.product_type,
            "target": self.target,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_dataset_contract(
    ds: xr.Dataset,
    *,
    product_type: str,
    required_attrs: Sequence[str] | None = None,
    required_data_vars: Sequence[str] | None = None,
) -> ContractReport:
    """Validate xarray dataset against product contract requirements."""
    report = ContractReport(product_type=product_type, target="dataset")

    dims = set(ds.dims)

    time_dim = _find_dim(dims, TIME_DIM_CANDIDATES)
    y_dim = _find_dim(dims, Y_DIM_CANDIDATES)
    x_dim = _find_dim(dims, X_DIM_CANDIDATES)
    member_dim = _find_dim(dims, MEMBER_DIM_CANDIDATES)

    if product_type in {"member", "ensemble", "lagged_ensemble"}:
        if time_dim is None:
            report.errors.append("Missing required time dimension")
        if member_dim is None:
            report.errors.append("Missing required member dimension")
        if y_dim is None or x_dim is None:
            report.errors.append("Missing required spatial dimensions (y/x)")

    elif product_type in {"diagnostic", "diagnostics"}:
        if time_dim is None:
            report.errors.append("Missing required time dimension")
        if y_dim is None or x_dim is None:
            report.errors.append("Missing required spatial dimensions (y/x)")

    elif product_type in {"observation_db", "manifest"}:
        # These product types are not xarray stores; caller should use specific validators.
        report.warnings.append(
            f"product_type '{product_type}' is not an xarray-native store; dataset checks are limited"
        )

    else:
        report.warnings.append(f"Unknown product_type '{product_type}'; applying generic checks")
        if y_dim is None or x_dim is None:
            report.warnings.append("Could not infer spatial dimensions")

    attrs_to_check = list(required_attrs) if required_attrs is not None else ["product_type", "created_at"]
    for attr in attrs_to_check:
        if attr not in ds.attrs:
            report.errors.append(f"Missing required attribute '{attr}'")

    if "product_type" in ds.attrs and str(ds.attrs["product_type"]) != product_type:
        report.warnings.append(
            f"product_type attr '{ds.attrs['product_type']}' differs from expected '{product_type}'"
        )

    if required_data_vars:
        for variable_name in required_data_vars:
            if variable_name not in ds.data_vars:
                report.errors.append(f"Missing required data variable '{variable_name}'")

    if not ds.data_vars:
        report.errors.append("Dataset has no data variables")

    return report


def validate_zarr_contract(
    zarr_path: Path | str,
    *,
    product_type: str,
    consolidated_required: bool = True,
    required_attrs: Sequence[str] | None = None,
    required_data_vars: Sequence[str] | None = None,
) -> ContractReport:
    """Validate Zarr store schema, metadata, and xarray compatibility markers."""
    path = Path(zarr_path).expanduser().resolve()
    report = ContractReport(product_type=product_type, target=str(path))

    if not path.exists():
        report.errors.append("Store path does not exist")
        return report

    zmetadata_path = path / ".zmetadata"
    if consolidated_required and not zmetadata_path.exists():
        report.errors.append("Missing consolidated metadata file .zmetadata")

    try:
        ds = xr.open_zarr(str(path), consolidated=zmetadata_path.exists())
    except Exception as exc:
        report.errors.append(f"Failed to open store with xarray: {exc}")
        return report

    dataset_report = validate_dataset_contract(
        ds,
        product_type=product_type,
        required_attrs=required_attrs,
        required_data_vars=required_data_vars,
    )
    report.errors.extend(dataset_report.errors)
    report.warnings.extend(dataset_report.warnings)

    try:
        import zarr

        root = zarr.open_group(str(path), mode="r")
        for variable_name in ds.variables:
            if variable_name not in root:
                report.errors.append(f"Variable '{variable_name}' missing from root group")
                continue

            attrs = dict(root[variable_name].attrs)
            if "_ARRAY_DIMENSIONS" not in attrs:
                report.errors.append(f"Variable '{variable_name}' missing _ARRAY_DIMENSIONS attribute")
    except Exception as exc:
        report.errors.append(f"Failed to inspect Zarr array attrs: {exc}")

    return report


def validate_observation_db_contract(db_path: Path | str) -> ContractReport:
    """Validate SQLite observations store schema."""
    path = Path(db_path).expanduser().resolve()
    report = ContractReport(product_type="observation_db", target=str(path))

    if not path.exists():
        report.errors.append("Observation DB path does not exist")
        return report

    try:
        import sqlite3

        conn = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "observations" not in tables:
                report.errors.append("Missing required table 'observations'")
                return report

            cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(observations)").fetchall()
            ]

            required_cols = {
                "obs_type",
                "obs_time",
                "platform_id",
                "lat",
                "lon",
                "elevation_m",
                "source_name",
                "payload_json",
            }

            missing = sorted(required_cols - set(cols))
            if missing:
                report.errors.append(f"Missing required columns: {missing}")
        finally:
            conn.close()
    except Exception as exc:
        report.errors.append(f"Failed to inspect observation DB: {exc}")

    return report


def assert_report_ok(report: ContractReport) -> None:
    """Raise ValueError when a contract report has errors."""
    if report.ok:
        return
    raise ValueError(
        f"Contract validation failed for {report.target}: " + "; ".join(report.errors)
    )


def _find_dim(dims: set[str], candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in dims:
            return name
    return None


__all__ = [
    "ContractReport",
    "assert_report_ok",
    "validate_dataset_contract",
    "validate_observation_db_contract",
    "validate_zarr_contract",
]

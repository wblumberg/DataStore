"""Canonical data product path builders for datastore outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def normalize_run_id(value: str | datetime) -> str:
    """Normalize run identifiers to YYYYMMDDHH format when possible."""
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y%m%d%H")

    text = str(value).strip()
    if not text:
        raise ValueError("run_id cannot be empty")

    compact = text.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    if len(compact) >= 10 and compact[:10].isdigit():
        return compact[:10]

    return text


@dataclass(frozen=True)
class ProductLayout:
    """Path layout helper for canonical product storage tiers."""

    base_dir: Path

    def raw_dir(self, *, system: str, run_id: str | datetime) -> Path:
        return self.base_dir / "raw" / system / normalize_run_id(run_id)

    def member_store(self, *, system: str, run_id: str | datetime, member: str) -> Path:
        filename = f"{member}.zarr"
        return self.base_dir / "members" / system / normalize_run_id(run_id) / filename

    def ensemble_store(self, *, system: str, run_id: str | datetime, name: str = "ensemble") -> Path:
        filename = f"{name}.zarr"
        return self.base_dir / "ensembles" / system / normalize_run_id(run_id) / filename

    def diagnostic_store(self, *, system: str, run_id: str | datetime, product: str) -> Path:
        filename = f"{product}.zarr"
        return self.base_dir / "diagnostics" / system / normalize_run_id(run_id) / filename

    def observations_db(self, *, dataset: str, date_key: str | datetime | None = None, filename: str = "observations.sqlite") -> Path:
        if date_key is None:
            return self.base_dir / "observations" / dataset / filename
        return self.base_dir / "observations" / dataset / normalize_run_id(date_key) / filename

    def manifest_path(self, *, system: str, run_id: str | datetime, filename: str = "manifest.json") -> Path:
        return self.base_dir / "manifests" / system / normalize_run_id(run_id) / filename


__all__ = ["ProductLayout", "normalize_run_id"]

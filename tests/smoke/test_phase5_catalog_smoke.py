"""Phase 5 smoke tests: data product catalog, manifest, and contract validation."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.catalog.paths import ProductLayout, normalize_run_id
from datastore.catalog.manifest import (
    ProductRecord,
    RunManifest,
    append_product_record,
    build_store_metadata,
    load_manifest,
    save_manifest,
)
from datastore.catalog.contract import (
    ContractReport,
    validate_dataset_contract,
    validate_observation_db_contract,
    validate_zarr_contract,
)
from datastore.workflows.orchestrator import load_profile, run_profile


class TestProductLayout(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        self.layout = ProductLayout(base_dir=self.base)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_normalize_run_id_from_string(self) -> None:
        self.assertEqual(normalize_run_id("2026031900"), "2026031900")
        self.assertEqual(normalize_run_id("2026-03-19T00"), "2026031900")

    def test_normalize_run_id_from_datetime(self) -> None:
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 19, 6, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(normalize_run_id(dt), "2026031906")

    def test_member_store_path(self) -> None:
        path = self.layout.member_store(system="hrefv3", run_id="2026031900", member="mem001")
        self.assertEqual(path, self.base / "members" / "hrefv3" / "2026031900" / "mem001.zarr")

    def test_ensemble_store_path(self) -> None:
        path = self.layout.ensemble_store(system="hrefv3", run_id="2026031900")
        self.assertEqual(path, self.base / "ensembles" / "hrefv3" / "2026031900" / "ensemble.zarr")

    def test_diagnostic_store_path(self) -> None:
        path = self.layout.diagnostic_store(system="hrefv3", run_id="2026031900", product="prob_exceed_1in")
        self.assertEqual(
            path, self.base / "diagnostics" / "hrefv3" / "2026031900" / "prob_exceed_1in.zarr"
        )

    def test_observations_db_path_no_date(self) -> None:
        path = self.layout.observations_db(dataset="metar")
        self.assertEqual(path, self.base / "observations" / "metar" / "observations.sqlite")

    def test_manifest_path(self) -> None:
        path = self.layout.manifest_path(system="hrefv3", run_id="2026031900")
        self.assertEqual(path, self.base / "manifests" / "hrefv3" / "2026031900" / "manifest.json")


class TestRunManifest(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_manifest_roundtrip(self) -> None:
        manifest = RunManifest(
            workflow="test_workflow",
            profile="test_profile",
            system="hrefv3",
            run_id="2026031900",
        )
        record = ProductRecord(
            product_type="member",
            path="/data/members/mem001.zarr",
            system="hrefv3",
            run_id="2026031900",
        )
        manifest.products.append(record)

        saved_path = self.tmp / "manifest.json"
        save_manifest(saved_path, manifest)
        self.assertTrue(saved_path.exists())

        loaded = load_manifest(saved_path)
        self.assertEqual(loaded.workflow, "test_workflow")
        self.assertEqual(loaded.system, "hrefv3")
        self.assertEqual(loaded.run_id, "2026031900")
        self.assertEqual(len(loaded.products), 1)
        self.assertEqual(loaded.products[0].path, "/data/members/mem001.zarr")

    def test_append_product_record_creates_new(self) -> None:
        manifest_path = self.tmp / "new_manifest.json"
        self.assertFalse(manifest_path.exists())

        record = ProductRecord(product_type="diagnostics", path="/data/diag.zarr", system="hrefv3")
        out = append_product_record(
            manifest_path=manifest_path,
            record=record,
            workflow="smoke_test",
            system="hrefv3",
            run_id="2026031900",
        )
        self.assertTrue(out.exists())

        loaded = load_manifest(out)
        self.assertEqual(loaded.workflow, "smoke_test")
        self.assertEqual(len(loaded.products), 1)

    def test_append_product_record_appends_to_existing(self) -> None:
        manifest_path = self.tmp / "existing_manifest.json"

        for i in range(3):
            record = ProductRecord(product_type="member", path=f"/data/mem{i:03d}.zarr")
            append_product_record(manifest_path=manifest_path, record=record)

        loaded = load_manifest(manifest_path)
        self.assertEqual(len(loaded.products), 3)
        paths = [p.path for p in loaded.products]
        self.assertIn("/data/mem000.zarr", paths)
        self.assertIn("/data/mem002.zarr", paths)

    def test_build_store_metadata_returns_dict(self) -> None:
        meta = build_store_metadata(
            product_type="member",
            system="hrefv3",
            run_id="2026031900",
            source="test",
            extra={"member": "mem001"},
        )
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["product_type"], "member")
        self.assertEqual(meta["system"], "hrefv3")
        self.assertEqual(meta["run_id"], "2026031900")
        self.assertIn("created_at", meta)
        self.assertEqual(meta.get("member"), "mem001")


class TestContractValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _make_obs_db(self, path: Path) -> Path:
        """Create a minimal valid observations SQLite database matching ObservationSQLStore schema."""
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE observations (
                id INTEGER PRIMARY KEY,
                obs_type TEXT,
                obs_time TEXT,
                platform_id TEXT,
                lat REAL,
                lon REAL,
                elevation_m REAL,
                source_name TEXT,
                payload_json TEXT
            )"""
        )
        conn.commit()
        conn.close()
        return path

    def _make_zarr_store(self, path: Path, product_type: str = "diagnostics") -> Path:
        """Create a minimal valid Zarr store."""
        import pandas as pd
        import xarray as xr

        y = np.linspace(25, 50, 8)
        x = np.linspace(-110, -70, 10)
        times = pd.date_range("2026-03-19", periods=2, freq="1h")
        data = np.random.rand(2, 8, 10)
        ds = xr.Dataset(
            {"precip": (["time", "y", "x"], data)},
            coords={"time": times, "y": y, "x": x},
            attrs={
                "product_type": product_type,
                "created_at": "2026-03-19T00:00:00Z",
                "system": "hrefv3",
            },
        )
        encoding = {v: {"chunks": ds[v].shape} for v in ds.data_vars}
        encoding.update({c: {"chunks": ds[c].shape} for c in ds.coords})
        ds.to_zarr(str(path), mode="w", zarr_format=2, consolidated=True, encoding=encoding)
        return path

    def test_validate_obs_db_valid(self) -> None:
        db_path = self.tmp / "obs.sqlite"
        self._make_obs_db(db_path)

        report = validate_observation_db_contract(db_path)
        self.assertIsInstance(report, ContractReport)
        self.assertTrue(report.ok, msg=f"Contract errors: {report.errors}")

    def test_validate_obs_db_missing(self) -> None:
        report = validate_observation_db_contract(self.tmp / "missing.sqlite")
        self.assertFalse(report.ok)
        self.assertTrue(any("does not exist" in e for e in report.errors))

    def test_validate_zarr_contract_valid(self) -> None:
        zarr_path = self.tmp / "diag.zarr"
        self._make_zarr_store(zarr_path, product_type="diagnostics")

        report = validate_zarr_contract(zarr_path, product_type="diagnostics")
        self.assertIsInstance(report, ContractReport)
        self.assertTrue(report.ok, msg=f"Contract errors: {report.errors}")

    def test_validate_zarr_contract_missing_store(self) -> None:
        report = validate_zarr_contract(self.tmp / "no_such.zarr", product_type="diagnostics")
        self.assertFalse(report.ok)
        self.assertTrue(any("does not exist" in e for e in report.errors))

    def test_validate_dataset_contract_no_data_vars(self) -> None:
        import xarray as xr

        ds = xr.Dataset()
        report = validate_dataset_contract(ds, product_type="diagnostics")
        self.assertFalse(report.ok)
        self.assertTrue(any("no data variables" in e for e in report.errors))


class TestOrchestratorCatalogActions(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _run_profile_json(self, steps: list[dict]) -> list:
        profile_path = self.tmp / "profile.json"
        profile_path.write_text(
            json.dumps({"name": "catalog_smoke", "steps": steps}), encoding="utf-8"
        )
        profile = load_profile(profile_path)
        return run_profile(profile, dry_run=False)

    def test_catalog_paths_action(self) -> None:
        results = self._run_profile_json(
            [
                {
                    "name": "resolve_paths",
                    "action": "catalog_paths",
                    "params": {
                        "base_dir": str(self.tmp),
                        "system": "hrefv3",
                        "run_id": "2026031900",
                        "members": ["mem001", "mem002"],
                        "manifest": True,
                    },
                }
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok", msg=str(results[0].details))
        paths = results[0].details["paths"]
        self.assertIn("member_mem001", paths)
        self.assertIn("member_mem002", paths)
        self.assertIn("manifest", paths)
        self.assertIn("mem001.zarr", paths["member_mem001"])

    def test_write_manifest_action(self) -> None:
        manifest_path = self.tmp / "manifests" / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        results = self._run_profile_json(
            [
                {
                    "name": "write_manifest",
                    "action": "write_manifest",
                    "params": {
                        "manifest_path": str(manifest_path),
                        "product_type": "member",
                        "path": str(self.tmp / "mem001.zarr"),
                        "system": "hrefv3",
                        "run_id": "2026031900",
                        "workflow": "smoke_test",
                    },
                }
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok", msg=str(results[0].details))
        self.assertTrue(manifest_path.exists())
        loaded = load_manifest(manifest_path)
        self.assertEqual(len(loaded.products), 1)
        self.assertEqual(loaded.products[0].product_type, "member")

    def test_validate_contract_action_sqlite(self) -> None:
        db_path = self.tmp / "obs.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE observations (
                id INTEGER PRIMARY KEY,
                obs_type TEXT,
                obs_time TEXT,
                platform_id TEXT,
                lat REAL,
                lon REAL,
                elevation_m REAL,
                source_name TEXT,
                payload_json TEXT
            )"""
        )
        conn.commit()
        conn.close()

        results = self._run_profile_json(
            [
                {
                    "name": "validate_obs",
                    "action": "validate_contract",
                    "params": {
                        "target": str(db_path),
                        "store_type": "sqlite",
                    },
                }
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok", msg=str(results[0].details))
        self.assertTrue(results[0].details["ok"])

    def test_validate_contract_action_missing_raises(self) -> None:
        results = self._run_profile_json(
            [
                {
                    "name": "validate_missing",
                    "action": "validate_contract",
                    "params": {
                        "target": str(self.tmp / "does_not_exist.zarr"),
                        "store_type": "zarr",
                        "product_type": "diagnostics",
                        "raise_on_error": True,
                    },
                }
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "error")


if __name__ == "__main__":
    unittest.main()

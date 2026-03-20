from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.pipelines import load_lagged_member_ensemble


class TestLaggedMemberLoaderSmoke(unittest.TestCase):
    def test_load_lagged_member_ensemble_is_lazy_and_tags_lags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latest_dir = tmp_path / "20260320.00.href_members"
            lagged_dir = tmp_path / "20260319.12.href_members"
            latest_dir.mkdir(parents=True, exist_ok=True)
            lagged_dir.mkdir(parents=True, exist_ok=True)

            self._write_member_store(
                latest_dir / "wrf4nssl.zarr",
                member="wrf4nssl",
                times=pd.date_range("2026-03-20T01:00:00", periods=3, freq="1h"),
                offset=0.0,
            )
            self._write_member_store(
                lagged_dir / "wrf4nssl.zarr",
                member="wrf4nssl",
                times=pd.date_range("2026-03-20T02:00:00", periods=3, freq="1h"),
                offset=10.0,
            )

            ensemble = load_lagged_member_ensemble(
                [latest_dir, lagged_dir],
                chunks="auto",
                join="inner",
            )

            self.assertEqual(list(ensemble.member.values), ["wrf4nssl_lag00", "wrf4nssl_lag12"])
            self.assertEqual(list(ensemble.base_member.values), ["wrf4nssl", "wrf4nssl"])
            self.assertEqual(list(ensemble.source_cycle.values), ["20260320.00", "20260319.12"])
            self.assertEqual(list(ensemble.lag_hours.values), [0, 12])
            self.assertEqual(ensemble.attrs["latest_cycle"], "20260320.00")
            self.assertEqual(ensemble.sizes["time"], 2)
            self.assertTrue(hasattr(ensemble["uh"].data, "chunks"))
            self.assertIsNotNone(ensemble["uh"].data.chunks)

    def test_load_lagged_member_ensemble_filters_requested_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latest_dir = tmp_path / "20260320.00.href_members"
            lagged_dir = tmp_path / "20260319.12.href_members"
            latest_dir.mkdir(parents=True, exist_ok=True)
            lagged_dir.mkdir(parents=True, exist_ok=True)

            self._write_member_store(
                latest_dir / "wrf4nssl.zarr",
                member="wrf4nssl",
                times=pd.date_range("2026-03-20T01:00:00", periods=3, freq="1h"),
                offset=0.0,
            )
            self._write_member_store(
                lagged_dir / "wrf4nssl.zarr",
                member="wrf4nssl",
                times=pd.date_range("2026-03-20T02:00:00", periods=3, freq="1h"),
                offset=10.0,
            )

            ensemble = load_lagged_member_ensemble(
                [latest_dir, lagged_dir],
                variables=["t2m"],
                chunks="auto",
                join="inner",
            )

            self.assertEqual(list(ensemble.data_vars), ["t2m"])
            self.assertEqual(ensemble.attrs["selected_variables"], ["t2m"])
            self.assertNotIn("uh", ensemble.data_vars)
            self.assertTrue(hasattr(ensemble["t2m"].data, "chunks"))
            self.assertIsNotNone(ensemble["t2m"].data.chunks)

    @staticmethod
    def _write_member_store(path: Path, *, member: str, times: pd.DatetimeIndex, offset: float) -> None:
        uh_data = np.ones((len(times), 1, 2, 2), dtype=np.float32) * offset
        t2m_data = np.ones((len(times), 1, 2, 2), dtype=np.float32) * (offset + 273.15)
        ds = xr.Dataset(
            {
                "uh": (("time", "member", "x", "y"), uh_data),
                "t2m": (("time", "member", "x", "y"), t2m_data),
            },
            coords={
                "time": times,
                "member": [member],
                "x": [0, 1],
                "y": [0, 1],
            },
        )

        encoding = {
            name: {"chunks": tuple(max(1, min(size, 2)) for size in variable.shape)}
            for name, variable in ds.variables.items()
            if variable.ndim > 0
        }
        ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2, encoding=encoding)


if __name__ == "__main__":
    unittest.main()
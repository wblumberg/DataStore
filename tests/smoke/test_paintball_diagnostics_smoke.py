from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.diagnostics.ensemble import apply_ensemble_diagnostics, paintball_bitmask


class TestPaintballDiagnosticsSmoke(unittest.TestCase):
    def test_paintball_bitmask_packs_member_bits(self) -> None:
        values = np.array(
            [
                [[10.0, 5.0], [0.0, 9.0]],
                [[8.0, 7.0], [3.0, 2.0]],
                [[1.0, 6.0], [5.0, 4.0]],
                [[9.0, 1.0], [8.0, 0.0]],
            ],
            dtype=np.float32,
        )
        da = xr.DataArray(
            values,
            dims=("member", "y", "x"),
            coords={"member": ["m1", "m2", "m3", "m4"], "y": [0, 1], "x": [0, 1]},
            name="uh",
        )

        paintball = paintball_bitmask(da, threshold=5.0, strict=False)

        expected = np.array([[11, 7], [12, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(paintball.values, expected)
        self.assertEqual(str(paintball.dtype), "uint8")
        self.assertEqual(paintball.attrs["paintball_member_count"], 4)
        self.assertEqual(paintball.attrs["paintball_threshold"], 5.0)
        self.assertFalse(paintball.attrs["paintball_strict"])

    def test_apply_ensemble_diagnostics_supports_paintball_requests(self) -> None:
        values = np.array(
            [
                [
                    [[10.0, 5.0], [0.0, 9.0]],
                    [[8.0, 7.0], [3.0, 2.0]],
                    [[1.0, 6.0], [5.0, 4.0]],
                    [[9.0, 1.0], [8.0, 0.0]],
                ]
            ],
            dtype=np.float32,
        )
        ds = xr.Dataset(
            {
                "uh": (("time", "member", "y", "x"), values),
            },
            coords={
                "time": [np.datetime64("2026-03-19T00:00:00")],
                "member": ["m1", "m2", "m3", "m4"],
                "y": [0, 1],
                "x": [0, 1],
            },
        )

        diag = apply_ensemble_diagnostics(
            ds=ds,
            variable="uh",
            paintball_requests=[
                {"name": "uh_paintball_ge5", "threshold": 5.0},
                {"name": "uh_paintball_gt5", "threshold": 5.0, "strict": True, "output_dtype": "uint16"},
            ],
            include_pmm=False,
            include_lpmm=False,
        )

        self.assertIn("uh_paintball_ge5", diag.data_vars)
        self.assertIn("uh_paintball_gt5", diag.data_vars)
        self.assertNotIn("member", diag["uh_paintball_ge5"].dims)

        ge5_expected = np.array([[11, 7], [12, 1]], dtype=np.uint8)
        gt5_expected = np.array([[11, 6], [8, 1]], dtype=np.uint16)

        np.testing.assert_array_equal(diag["uh_paintball_ge5"].isel(time=0).values, ge5_expected)
        np.testing.assert_array_equal(diag["uh_paintball_gt5"].isel(time=0).values, gt5_expected)

        self.assertEqual(str(diag["uh_paintball_ge5"].dtype), "uint8")
        self.assertEqual(str(diag["uh_paintball_gt5"].dtype), "uint16")


if __name__ == "__main__":
    unittest.main()
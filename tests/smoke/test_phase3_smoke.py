from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.grib.discovery import build_file_index, discover_files, match_by_valid_time
from datastore.observations.obs2sql import parse_json_like_file, parse_lightning_txt_file
from datastore.observations.sql_store import ObservationSQLStore, parse_time_like
from datastore.pipelines.postprocess_ensemble import postprocess_member_stores


class TestPhase3Smoke(unittest.TestCase):
    def test_grib_discovery_and_lag_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            member_dir = root / "WRF4NSSL"
            member_dir.mkdir(parents=True, exist_ok=True)

            # 2026031900 + f003 and 2026031818 + f009 share valid time 2026031903.
            filenames = [
                "wrf4nssl_2026031900f003.grib2",
                "wrf4nssl_2026031900f006.grib2",
                "wrf4nssl_2026031818f009.grib2",
            ]
            for name in filenames:
                (member_dir / name).touch()

            discovered = discover_files([root], patterns=["*.grib2"], exclude_dirs=[])
            self.assertEqual(len(discovered), 3)

            index = build_file_index(discovered)
            self.assertEqual(len(index), 3)

            matches = match_by_valid_time(index, max_lags=1, cycle_spacing_hours=6)
            self.assertEqual(len(matches), 2)
            self.assertIn(2, [len(group) for group in matches.values()])

    def test_postprocess_pipeline_writes_diagnostics_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            member_paths = []

            time_coord = np.array([np.datetime64("2026-03-19T00:00:00"), np.datetime64("2026-03-19T01:00:00")])
            y_coord = np.array([0, 1], dtype=np.int32)
            x_coord = np.array([0, 1], dtype=np.int32)
            lat = np.array([[35.0, 35.1], [35.2, 35.3]], dtype=np.float32)
            lon = np.array([[-97.6, -97.5], [-97.4, -97.3]], dtype=np.float32)

            for idx, member in enumerate(("m01", "m02")):
                data = np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[2.0, 3.0], [4.0, 5.0]],
                    ],
                    dtype=np.float32,
                ) + float(idx)
                ds = xr.Dataset(
                    data_vars={"uh": (("time", "y", "x"), data)},
                    coords={
                        "time": ("time", time_coord),
                        "y": ("y", y_coord),
                        "x": ("x", x_coord),
                        "lat": (("y", "x"), lat),
                        "lon": (("y", "x"), lon),
                    },
                )
                member_path = tmp_path / f"{member}.zarr"
                encoding = {}
                for name, variable in ds.variables.items():
                    if variable.ndim == 0:
                        continue
                    chunks = []
                    for size in variable.shape:
                        if size <= 1:
                            chunks.append(1)
                        else:
                            chunks.append(min(size, 2))
                    encoding[name] = {"chunks": tuple(chunks)}

                ds.to_zarr(
                    member_path,
                    mode="w",
                    consolidated=True,
                    zarr_format=2,
                    encoding=encoding,
                )
                member_paths.append(member_path)

            output = tmp_path / "diagnostics.zarr"
            postprocess_member_stores(
                member_zarrs=member_paths,
                variable="uh",
                output_zarr=output,
                thresholds={"uh_gt_1": 1.0},
                percentile_probs=[0.5],
                include_pmm=False,
                include_lpmm=False,
            )

            diag = xr.open_zarr(output, consolidated=True)
            self.assertIn("uh_mean", diag.data_vars)
            self.assertIn("uh_spread", diag.data_vars)
            self.assertIn("prob_uh_gt_1", diag.data_vars)
            self.assertIn("uh_p50", diag.data_vars)

    def test_observation_ingest_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            json_file = tmp_path / "metar_20260319_0000.json"
            json_file.write_text(
                json.dumps(
                    [
                        {
                            "station": "KOUN",
                            "time": "2026-03-19T00:00:00Z",
                            "lat": 35.22,
                            "lon": -97.47,
                            "tmpf": 65.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = parse_json_like_file(
                path=json_file,
                obs_type="METAR",
                source_name="metar_test",
                fallback_time=None,
            )
            self.assertEqual(len(rows), 1)

            lightning_file = tmp_path / "lightning.txt"
            lightning_columns = [
                "2026",
                "03",
                "19",
                "00",
                "00",
                "30",
                "0",
                "35.10",
                "-97.40",
                "20.0",
                "1",
                "7",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
            lightning_file.write_text("\t".join(lightning_columns) + "\n", encoding="utf-8")

            lightning_rows = parse_lightning_txt_file(
                path=lightning_file,
                obs_type="LIGHTNING",
                source_name="lightning_test",
                fallback_time=None,
            )
            self.assertEqual(len(lightning_rows), 1)

            db_path = tmp_path / "observations.sqlite"
            store = ObservationSQLStore(db_path)
            store.initialize()

            insert_result = store.insert_many(rows + lightning_rows)
            self.assertEqual(insert_result["inserted"], 2)

            times = store.list_unique_times(obs_types=["METAR"], limit=10)
            self.assertEqual(len(times), 1)

            query = store.query_observations(
                obs_types=["METAR"],
                center_time=parse_time_like("202603190000"),
                minutes_before=5,
                minutes_after=10,
                max_rows=100,
            )
            self.assertEqual(query["count"], 1)

    def test_wrapper_help_commands(self) -> None:
        scripts = [
            PROJECT_ROOT / "ingest_model" / "ingest_atcf.py",
            PROJECT_ROOT / "ingest_model" / "ingest_tracks.py",
            PROJECT_ROOT / "utils" / "postprocess_ensemble.py",
            PROJECT_ROOT / "utils" / "extract_contours.py",
            PROJECT_ROOT / "utils" / "obs2sql.py",
            PROJECT_ROOT / "staging" / "obs" / "obs2sql.py",
        ]

        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{script} help failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()

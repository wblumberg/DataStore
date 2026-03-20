from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datastore.workflows.orchestrator import load_profile, run_profile, summarize_results


class TestPhase4OrchestratorSmoke(unittest.TestCase):
    def test_load_and_dry_run_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile_path = tmp_path / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "dry_run_profile",
                        "variables": {"msg": "hello"},
                        "steps": [
                            {
                                "name": "echo",
                                "action": "shell",
                                "params": {"command": "echo {msg}"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            profile = load_profile(profile_path)
            results = run_profile(profile, dry_run=True)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "dry-run")
            self.assertIn("echo hello", results[0].details["preview"])

    def test_obs_profile_executes_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "obs.sqlite"
            obs_path = tmp_path / "obs.json"

            obs_path.write_text(
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

            profile_path = tmp_path / "obs_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "obs_profile",
                        "variables": {
                            "db": str(db_path),
                            "obs_file": str(obs_path),
                        },
                        "steps": [
                            {
                                "name": "init",
                                "action": "obs_init",
                                "params": {"db": "{db}"},
                            },
                            {
                                "name": "ingest",
                                "action": "obs_ingest",
                                "params": {
                                    "db": "{db}",
                                    "obs_type": "METAR",
                                    "input": ["{obs_file}"],
                                    "format": "json",
                                },
                            },
                            {
                                "name": "query",
                                "action": "obs_query",
                                "params": {
                                    "db": "{db}",
                                    "obs_types": ["METAR"],
                                    "latest_only": True,
                                    "max_rows": 100,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            profile = load_profile(profile_path)
            results = run_profile(profile, dry_run=False)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.status == "ok" for result in results))
            summary = summarize_results(results)
            self.assertEqual(summary["errors"], 0)

            ingest_result = next(result for result in results if result.step == "ingest")
            self.assertEqual(ingest_result.details["inserted"], 1)

            query_result = next(result for result in results if result.step == "query")
            self.assertEqual(query_result.details["count"], 1)

    def test_profile_cli_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile_path = tmp_path / "cli_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "cli_profile",
                        "steps": [
                            {
                                "name": "dry",
                                "action": "shell",
                                "params": {"command": "echo ok"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "utils" / "run_profile.py"),
                    "--profile",
                    str(profile_path),
                    "--dry-run",
                    "--json",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["summary"]["dry_run"], 1)


if __name__ == "__main__":
    unittest.main()

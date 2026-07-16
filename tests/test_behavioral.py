import importlib.util
import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "behavioral_pipeline", ROOT / "pipeline/verify-behavioral/behavioral.py")
behavioral = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(behavioral)
warnings.filterwarnings("ignore")


def write_bundle(out: Path, bsid: int, mouse_id="42") -> None:
    n = 60
    start = np.arange(n, dtype=float)
    hit = np.arange(n) % 3 == 0
    miss = ~hit
    trials = pd.DataFrame({
        "start_time": start,
        "stop_time": start + 0.8,
        "hit": hit,
        "miss": miss,
        "aborted": np.zeros(n, dtype=bool),
        "response_latency": np.where(hit, 0.45, np.inf),
        "lick_times": [[] for _ in range(n)],
    })
    trials.to_parquet(out / f"{bsid}.trials.parquet")
    pd.DataFrame({"start_time": start, "stop_time": start + 0.25}).to_parquet(
        out / f"{bsid}.stim.parquet")
    pd.DataFrame({"timestamps": np.arange(5.0, 60.0, 5.0),
                  "auto_rewarded": False}).to_parquet(out / f"{bsid}.rewards.parquet")
    pd.DataFrame({"timestamps": np.arange(2.0, 60.0, 2.0)}).to_parquet(
        out / f"{bsid}.licks.parquet")
    (out / f"{bsid}.meta.json").write_text(json.dumps({
        "mouse_id": mouse_id,
        "project_code": "VisualBehavior",
        "equipment_name": "CAM2P.3",
        "session_type": "OPHYS_1_images_A",
    }))


class FakeSession:
    def __init__(self):
        self.trials = pd.DataFrame({"lick_times": [np.array([0.1])]}, index=[0])
        self.stimulus_presentations = pd.DataFrame({
            "start_time": [0.0], "stop_time": [0.25], "active": [True]
        }, index=[0])
        self.rewards = pd.DataFrame({"timestamps": [0.5], "auto_rewarded": [False]})
        self.licks = pd.DataFrame({"timestamps": [0.1, 0.2]})
        self.metadata = {"mouse_id": 42, "project_code": "VisualBehavior"}


class BehavioralTests(unittest.TestCase):
    def test_shard_validation(self):
        self.assertEqual(behavioral.parse_shard("3/10"), (3, 10))
        for value in ("0/10", "11/10", "1/0", "abc", "1/2/3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                behavioral.parse_shard(value)

    def test_pull_serializes_numpy_id_and_resumes_complete_bundle(self):
        target = (
            "allensdk.brain_observatory.behavior.behavior_project_cache."
            "VisualBehaviorOphysProjectCache.from_s3_cache"
        )
        cache = mock.Mock()
        cache.get_behavior_session.return_value = FakeSession()
        with tempfile.TemporaryDirectory() as tmp, mock.patch(target, return_value=cache), \
                redirect_stdout(io.StringIO()):
            root = Path(tmp)
            out = root / "bundles"
            report_name = "_pull_01-of-01.json"
            self.assertEqual(behavioral.pull(
                [np.int64(123)], out, root / "cache", retries=1,
                report_name=report_name), 0)
            report = json.loads((out / report_name).read_text())
            self.assertEqual(report["ok"][0]["behavior_session_id"], 123)
            self.assertTrue(behavioral.bundle_complete(out, 123))
            self.assertEqual(behavioral.pull(
                [123], out, root / "cache", retries=1,
                report_name=report_name), 0)
            report = json.loads((out / report_name).read_text())
            self.assertEqual(report["skipped"], [123])

    def test_pull_retries_then_publishes_complete_bundle(self):
        target = (
            "allensdk.brain_observatory.behavior.behavior_project_cache."
            "VisualBehaviorOphysProjectCache.from_s3_cache"
        )
        calls = {"count": 0}

        class FlakyCache:
            def get_behavior_session(self, _):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise OSError("temporary S3 failure")
                return FakeSession()

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch(target, return_value=FlakyCache()), \
                mock.patch.object(behavioral.time, "sleep"), \
                redirect_stdout(io.StringIO()):
            root = Path(tmp)
            out = root / "bundles"
            self.assertEqual(behavioral.pull(
                [123], out, root / "cache", retries=2,
                report_name="_pull_01-of-01.json"), 0)
            report = json.loads((out / "_pull_01-of-01.json").read_text())
            self.assertEqual(report["ok"][0]["attempts"], 2)
            self.assertTrue(behavioral.bundle_complete(out, 123))
            self.assertFalse((out / ".staging").exists())

    def test_pull_does_not_retry_deterministic_decode_error(self):
        target = (
            "allensdk.brain_observatory.behavior.behavior_project_cache."
            "VisualBehaviorOphysProjectCache.from_s3_cache"
        )
        calls = {"count": 0}

        class BrokenCache:
            def get_behavior_session(self, _):
                calls["count"] += 1
                raise TypeError("incompatible NWB schema")

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch(target, return_value=BrokenCache()), \
                mock.patch.object(behavioral.time, "sleep"), \
                redirect_stdout(output):
            root = Path(tmp)
            out = root / "bundles"
            self.assertEqual(behavioral.pull(
                [123], out, root / "cache", retries=3,
                report_name="_pull_01-of-01.json"), 1)
            report = json.loads((out / "_pull_01-of-01.json").read_text())
            self.assertEqual(calls["count"], 1)
            self.assertEqual(report["failed"][0]["attempts"], 1)
            self.assertIn("TypeError: incompatible NWB schema", output.getvalue())

    def test_validate_rejects_incomplete_and_mismatched_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "123.trials.parquet").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                behavioral.validate_bundle_set(root)
            for path in root.iterdir():
                path.unlink()
            write_bundle(root, 123)
            ids = root / "ids.csv"
            pd.DataFrame({"behavior_session_id": [999]}).to_csv(ids, index=False)
            with self.assertRaisesRegex(ValueError, "missing=.*999.*extra=.*123"):
                behavioral.validate_bundle_set(root, ids)

    def test_scan_writes_baseline_and_machine_readable_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_bundle(root, 123)
            ids = root / "ids.csv"
            pd.DataFrame({"behavior_session_id": [123]}).to_csv(ids, index=False)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(behavioral.scan(root, True, ids), 0)
            self.assertTrue((root / "_scan.parquet").is_file())
            sweep = pd.read_parquet(root / "_sweep.parquet")
            self.assertEqual(len(sweep), 36)
            self.assertIn("qualifying_sessions_A", sweep.columns)
            self.assertIn("qualifying_mice_B", sweep.columns)

    def test_scan_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no behavior bundles"):
                behavioral.scan(Path(tmp), False)


if __name__ == "__main__":
    unittest.main()

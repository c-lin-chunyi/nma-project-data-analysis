import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("split_pipeline", ROOT / "pipeline/draw/split.py")
split_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split_pipeline)


def experiment_table() -> pd.DataFrame:
    rows = []
    for i, mouse_id in enumerate(range(1001, 1009)):
        rows.append({
            "ophys_experiment_id": 2000 + i,
            "behavior_session_id": 3000 + i,
            "mouse_id": mouse_id,
            "cre_line": split_pipeline.CRE,
            "equipment_name": "CAM2P.3" if i % 2 == 0 else "CAM2P.4",
            "session_type": "OPHYS_1_images_A",
            "project_code": "VisualBehavior" if i < 4 else "VisualBehaviorTask1B",
            "imaging_depth": 175 if i % 2 == 0 else 275,
            "targeted_structure": "VISp",
            "ophys_container_id": 4000 + i,
        })
    return pd.DataFrame(rows).set_index("ophys_experiment_id")


class FakeCache:
    def __init__(self, manifest="manifest-v1.json"):
        self.manifest = manifest

    def current_manifest(self):
        return self.manifest

    def get_ophys_experiment_table(self):
        return experiment_table()


class SplitTests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_deterministic(self):
        counts = {"b": 5, "a": 5, "c": 2}
        first = split_pipeline.largest_remainder(counts, 3)
        second = split_pipeline.largest_remainder(counts, 3)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 3)

    def test_split_is_deterministic_and_collapses_thin_cells(self):
        cohort = split_pipeline.build_cohort(experiment_table())
        strata, why = split_pipeline.choose_strata(cohort)
        first, info1 = split_pipeline.split(split_pipeline.mouse_table(cohort), strata, why)
        second, info2 = split_pipeline.split(split_pipeline.mouse_table(cohort), strata, why)
        self.assertEqual(split_pipeline.checksum(first), split_pipeline.checksum(second))
        self.assertEqual(info1, info2)
        self.assertTrue(info1["collapsed"])
        self.assertEqual(info1["n_dev"], 2)

    def test_verify_detects_manifest_and_full_assignment_drift(self):
        cache = FakeCache()
        cohort = split_pipeline.build_cohort(cache.get_ophys_experiment_table())
        strata, why = split_pipeline.choose_strata(cohort)
        mt, info = split_pipeline.split(split_pipeline.mouse_table(cohort), strata, why)
        manifest = dict(
            allen_manifest=cache.current_manifest(),
            dev_checksum=split_pipeline.checksum(mt),
            dev_mice=sorted(mt.loc[mt.tier.eq("dev"), "mouse_id"].astype(int).tolist()),
            confirm_mice=sorted(mt.loc[mt.tier.eq("confirm"), "mouse_id"].astype(int).tolist()),
            **info,
        )
        target = (
            "allensdk.brain_observatory.behavior.behavior_project_cache."
            "VisualBehaviorOphysProjectCache.from_s3_cache"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split_manifest.json"
            path.write_text(json.dumps(manifest))
            with mock.patch(target, return_value=cache):
                self.assertEqual(split_pipeline.verify(path), 0)
                cache.manifest = "manifest-v2.json"
                self.assertEqual(split_pipeline.verify(path), 1)
                cache.manifest = "manifest-v1.json"
                manifest["confirm_mice"] = manifest["confirm_mice"][:-1]
                path.write_text(json.dumps(manifest))
                self.assertEqual(split_pipeline.verify(path), 1)


if __name__ == "__main__":
    unittest.main()

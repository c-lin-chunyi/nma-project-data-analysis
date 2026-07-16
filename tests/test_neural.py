import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neural_pipeline", ROOT / "pipeline/verify-neural/neural.py")
neural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(neural)


class NeuralTests(unittest.TestCase):
    def test_shard_validation(self):
        self.assertEqual(neural.parse_shard("3/10"), (3, 10))
        for value in ("0/10", "11/10", "x", "1/0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                neural.parse_shard(value)

    def test_manifest_is_exact_50_active_20_passive(self):
        dev, rows = [], []
        for container in range(10):
            mouse = 1000 + container
            for session in (1, 3, 4, 6, 6):
                oeid = 100000 + len(rows)
                dev.append({"ophys_experiment_id": oeid,
                            "ophys_container_id": 2000 + container,
                            "mouse_id": mouse})
                rows.append({"ophys_experiment_id": oeid,
                             "ophys_container_id": 2000 + container,
                             "mouse_id": mouse, "behavior_session_id": oeid + 1,
                             "session_type": f"OPHYS_{session}_images_A"})
            for session in (2, 5):
                oeid = 100000 + len(rows)
                rows.append({"ophys_experiment_id": oeid,
                             "ophys_container_id": 2000 + container,
                             "mouse_id": mouse, "behavior_session_id": oeid + 1,
                             "session_type": f"OPHYS_{session}_images_A_passive"})
        result = neural.build_experiment_manifest(pd.DataFrame(dev), pd.DataFrame(rows))
        self.assertEqual(len(result), 70)
        self.assertEqual((result.role == "passive").sum(), 20)
        self.assertEqual(result.ophys_container_id.nunique(), 10)

    def test_temporal_folds_remove_raw_gap(self):
        raw = np.arange(100)
        for train, test in neural._folds(raw, n_blocks=5, gap=10):
            low, high = raw[test].min(), raw[test].max()
            self.assertFalse(np.any((raw[train] >= low-10) & (raw[train] < low)))
            self.assertFalse(np.any((raw[train] > high) & (raw[train] <= high+10)))

    def test_prechange_feature_does_not_bridge_long_gap(self):
        times = np.r_[np.arange(0, 1, .05), np.arange(2, 3, .05)]
        values = np.ones(len(times))
        feature, missing = neural._prechange_feature(times, values, np.array([2.0]))
        self.assertGreater(missing[0], .20)
        self.assertTrue(np.isnan(feature[0]))

    def test_trial_locked_pupil_normalizes_and_preserves_long_gap(self):
        table = pd.DataFrame({"timestamps": np.r_[np.arange(0, 1, .1),
                                                   np.arange(2, 3, .1)],
                              "pupil_area": 2.0})
        locked = neural._trial_locked_scalar(
            table, "pupil_area", np.array([2.0]), np.array([-.5, 0, .5]),
            normalize_median=True)
        self.assertTrue(np.isnan(locked[0, 0]))
        self.assertAlmostEqual(float(locked[0, 1]), 1.0)
        self.assertAlmostEqual(float(locked[0, 2]), 1.0)

    def test_q2_flash_history_stops_before_change(self):
        trials = pd.DataFrame({"trials_id": [1], "change_time": [1.0],
                               "initial_image_name": ["a"], "change_image_name": ["b"],
                               "hit": [True], "miss": [False], "false_alarm": [False],
                               "correct_reject": [False]})
        stim = pd.DataFrame({"trials_id": [1, 1, 1], "start_time": [.1, .9, 1.1],
                             "omitted": [False, True, False]})
        empty = pd.DataFrame({"timestamps": pd.Series(dtype=float)})
        trace = pd.DataFrame({"timestamps": np.arange(0, 2, .05), "speed": 1.0})
        eye = trace.rename(columns={"speed": "pupil_area"})
        result = neural._q2_features(trials, stim, empty, empty, eye, trace)
        self.assertEqual(result.loc[0, "flashes_before_change"], 2)
        self.assertTrue(result.loc[0, "preceding_omission"])

    def test_synthetic_active_extraction_has_appendix_a_arrays(self):
        timestamps = np.arange(0, 10, .1)
        cells = pd.DataFrame({"cell_roi_id": [11, 12], "valid_roi": [True, True]},
                             index=pd.Index([101, 102], name="cell_specimen_id"))
        traces = [np.sin(timestamps), np.cos(timestamps)]
        exp = SimpleNamespace(
            ophys_timestamps=timestamps,
            cell_specimen_table=cells,
            dff_traces=pd.DataFrame({"dff": traces}, index=cells.index),
            events=pd.DataFrame({"events": traces}, index=cells.index))
        trials = pd.DataFrame({"trials_id": [1, 2], "change_time": [2.0, 5.0]})
        trace = pd.DataFrame({"timestamps": timestamps, "speed": 1.0})
        eye = trace.rename(columns={"speed": "pupil_area"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.h5"
            qc = neural._write_h5(path, exp, "active", trials, eye, trace)
            import h5py
            with h5py.File(path, "r") as h5:
                self.assertEqual(qc["n_cells"], 2)
                self.assertIn("events_baselined", h5["trial_locked"])
                self.assertIn("pupil", h5["trial_locked"])
                self.assertIn("running", h5["trial_locked"])

    def test_one_se_rule_chooses_strongest_regularization(self):
        block = pd.DataFrame({"C": [.01, .1, 1.0],
                              "mouse_mean_auc": [.69, .70, .705],
                              "mouse_se": [.01, .01, .02]})
        self.assertEqual(neural._choose_one_se(block), .01)

    def test_frozen_secondary_reports_absent_novelty_without_crashing(self):
        rows = pd.DataFrame({
            "auc": [.55, .60, .65, .70], "novel": [False] * 4,
            "mouse_id": [1, 1, 2, 2], "miss_B": [20] * 4,
            "project_code": ["A", "A", "B", "B"],
        })
        result = neural._safe_secondary_model(rows, include_novel=True)
        self.assertEqual(result["error"], "nonestimable_no_novelty_variation")
        self.assertEqual(result["observed_novel_levels"], [False])

    def test_secondary_numerical_failure_is_diagnostic(self):
        rows = pd.DataFrame({"auc": [.5]})
        with mock.patch.object(neural, "_weighted_random_intercept",
                               side_effect=np.linalg.LinAlgError("singular")):
            result = neural._safe_secondary_model(rows, include_novel=False)
        self.assertEqual(result["error"], "secondary_model_numerical_failure")

    def test_secondary_detects_collinear_fixed_effects(self):
        rows = pd.DataFrame({
            "auc": [.55, .60, .65, .70], "novel": [False, False, True, True],
            "mouse_id": [1, 1, 2, 2], "miss_B": [20] * 4,
            "project_code": ["A", "A", "B", "B"],
        })
        result = neural._safe_secondary_model(rows, include_novel=True)
        self.assertEqual(result["error"], "nonestimable_rank_deficient_fixed_effects")
        self.assertEqual(result["rank"], 2)

    def test_recovery_workflow_skips_nwb_pull_and_reuses_artifacts(self):
        workflow = (ROOT / ".github/workflows/neural-dev.yml").read_text()
        self.assertIn("reuse_run_id:", workflow)
        self.assertIn("if: ${{ inputs.reuse_run_id == '' }}", workflow)
        self.assertIn('gh run download "$REUSE_RUN_ID" -n "neural-container-$shard"',
                      workflow)


if __name__ == "__main__":
    unittest.main()

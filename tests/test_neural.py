import importlib.util
import inspect
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

    def test_feature_cache_materializes_registered_windows_and_exact_alignment(self):
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
        manifest = pd.DataFrame([{
            "ophys_experiment_id": 123, "behavior_session_id": 456,
            "ophys_container_id": 789, "mouse_id": 10,
            "project_code": "A", "session_type": "OPHYS_1_images_A",
            "role": "active",
        }])
        labels = pd.DataFrame({"trial_id": [1, 2], "behavior_session_id": [456, 456],
                               "trial_index": [10, 20], "engaged_B": [True, False]})
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "source", Path(tmp) / "features"
            root.mkdir(); out.mkdir()
            neural._write_h5(root / "123.neural.h5", exp, "active", trials, eye, trace)
            pd.DataFrame({"trial_id": [1, 2], "change_time": [2.0, 5.0],
                          "q2_covariates_complete": [True, True]}).to_parquet(
                              root / "123.q2.parquet", index=False)
            row = next(manifest.itertuples(index=False))
            neural._write_feature_cache_experiment(
                root, out, row, labels, data_release="neural-dev-data-1",
                data_manifest_sha256="a" * 64,
                behavioral_release="behavioral-v3.1-1")
            import h5py
            with h5py.File(out / "123.features.h5", "r") as h5:
                self.assertEqual(h5.attrs["schema"], neural.FEATURE_CACHE_SCHEMA)
                self.assertEqual(h5["events_baselined_post"].shape, (2, 2))
                self.assertEqual(h5["events_unbaselined_pre"].shape, (2, 2))
                self.assertLess(float(np.max(np.abs(
                    h5["events_baselined_full_pre"][:]))), 1e-5)
            self.assertEqual(pd.read_parquet(out / "123.labels.parquet").trial_id.tolist(),
                             [1, 2])
            self.assertEqual(neural.feature_cache_failures(out, manifest), [])
            (out / "999.extra").write_text("unexpected")
            self.assertTrue(neural.feature_cache_failures(out, manifest))

    def test_v32_freezes_only_k50_and_c50(self):
        self.assertEqual(neural.PRIMARY_K, 50)
        self.assertEqual(neural.FROZEN_C50, 1e-4)
        self.assertFalse(hasattr(neural, "K_GRID"))
        self.assertFalse(hasattr(neural, "C_GRID"))

    def test_unbaselined_tonic_state_anchor_has_auc_and_logloss_gain(self):
        rng = np.random.default_rng(4)
        y = np.tile([0, 1], 60)
        X = rng.normal(scale=.5, size=(len(y), 50)) + y[:, None] * .8
        auc, gain, error = neural._state_oof_metrics(
            X, y, np.arange(len(y)), neural.FROZEN_C50, seed=0)
        self.assertIsNone(error)
        self.assertGreater(auc, .9)
        self.assertGreater(gain, 0)

    def test_both_q1_and_q2_precision_are_required(self):
        passed = neural._precision_gates(
            appendix_complete=True, q1_mice=8, anchor_mice=8, q2_mice=8,
            q1_sd=.01, q2_sd=.001, q1_margin=.02, q2_sesoi=.002)
        self.assertTrue(passed["confirm_ready"])
        q2_failed = neural._precision_gates(
            appendix_complete=True, q1_mice=8, anchor_mice=8, q2_mice=8,
            q1_sd=.01, q2_sd=.02, q1_margin=.02, q2_sesoi=.002)
        self.assertTrue(q2_failed["q1_precision"])
        self.assertFalse(q2_failed["q2_precision"])
        self.assertFalse(q2_failed["confirm_ready"])

    def test_baseline_integrity_uses_within_fold_constant_dv(self):
        rng = np.random.default_rng(8)
        y = np.tile([False, True], 50)
        labels = pd.DataFrame({
            "engaged_B": True, "keep_B": True, "late_hit": y,
            "miss": ~y, "trial_index": np.arange(len(y)),
        })
        item = {"X": rng.normal(size=(len(y), 50)),
                "baseline_pre": np.zeros((len(y), 50)),
                "labels": labels, "meta": {"ophys_experiment_id": 123}}
        result = neural._baseline_integrity(item, neural.FROZEN_C50)
        self.assertTrue(result["passed"])
        self.assertEqual(result["constant_score_auc"], .5)
        self.assertEqual(result["max_prechange_mean_dv_range"], 0)

    def test_baseline_integrity_uses_full_extraction_window_not_anchor_window(self):
        source = inspect.getsource(neural.scan)
        self.assertIn("baselined=True, start=None, end=0.0", source)
        self.assertIn("baselined=False,\n            start=PUPIL_START, end=PUPIL_END", source)
        rel = np.array([-1.25, -1.0, -.75, -.5, -.25, 0.0])
        raw = np.array([4.0, -1.0, -1.0, -1.0, -1.0, 99.0])
        baselined = raw - raw[rel < 0].mean()
        self.assertAlmostEqual(baselined[rel < 0].mean(), 0.0)
        self.assertNotAlmostEqual(baselined[(rel >= -1.0) & (rel < 0)].mean(), 0.0)

    def test_q2_nuisance_models_use_natural_prevalence(self):
        source = inspect.getsource(neural._q2_session)
        self.assertIn('class_weight=None', source)
        self.assertIn('"m0_log_loss"', source)
        self.assertIn('"m1_log_loss"', source)
        self.assertIn('"m0_brier"', source)
        self.assertIn('"q1_auc_same_trials"', source)

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

    def test_v32_workflow_reads_immutable_data_release_and_cannot_pull_nwb(self):
        workflow = (ROOT / ".github/workflows/neural-dev-v3.2.yml").read_text()
        self.assertIn("neural_data_release:", workflow)
        self.assertIn("analysis_only=true", workflow)
        self.assertIn("allen_nwb_download=false", workflow)
        self.assertIn('gh release download "$DATA_TAG"', workflow)
        self.assertIn("bundle-files.sha256", workflow)
        self.assertNotIn("gh run download", workflow)
        self.assertNotIn("neural.py pull", workflow)
        self.assertNotIn("neural.py manifest", workflow)
        self.assertNotIn("allen-neural", workflow)

    def test_feature_cache_workflow_streams_once_and_publishes_resumable_cache(self):
        workflow = (ROOT / ".github/workflows/neural-feature-cache.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("neural_data_release:", workflow)
        self.assertIn("behavioral_release:", workflow)
        self.assertIn('gh release download "$DATA_TAG"', workflow)
        self.assertIn("neural-dev-features-v1-", workflow)
        self.assertIn("reuse verified draft assets", workflow)
        self.assertIn("neural.py features", workflow)
        self.assertIn("neural.py feature-verify", workflow)
        self.assertIn("n_active_experiments", workflow)
        self.assertIn("contains_oof_predictions", workflow)
        self.assertIn("allen_nwb_download=false", workflow)
        self.assertIn("--prerelease --draft", workflow)
        self.assertIn("--draft=false", workflow)
        self.assertNotIn("neural.py pull", workflow)
        self.assertNotIn("neural.py manifest", workflow)
        self.assertNotIn("allen-neural", workflow)

    def test_neural_data_backfill_streams_existing_artifacts_to_draft_release(self):
        workflow = (ROOT / ".github/workflows/neural-data-backfill.yml").read_text()
        self.assertIn("source_run_id:", workflow)
        self.assertIn('gh run download "$SOURCE_RUN_ID"', workflow)
        self.assertIn("analysis_only_packaging=true", workflow)
        self.assertIn("allen_nwb_download=false", workflow)
        self.assertIn("split -b 1800M", workflow)
        self.assertIn("--prerelease --draft", workflow)
        self.assertIn('--target "$GITHUB_SHA"', workflow)
        self.assertNotIn('--target "$source_sha"', workflow)
        self.assertLess(workflow.index("smoke-test Release permission"),
                        workflow.index("actions/setup-python@v6"))
        self.assertIn("--draft=false", workflow)
        self.assertNotIn("neural.py pull", workflow)
        self.assertNotIn("neural.py manifest", workflow)
        self.assertNotIn("must contain 7 experiments", workflow)
        self.assertIn("manifest must contain 70 experiments", workflow)
        self.assertIn("manifest experiment IDs must be unique", workflow)
        self.assertIn("manifest must contain exactly 10 containers", workflow)
        self.assertIn("has no manifest experiments", workflow)

    def test_v32_prereg_freezes_unbaselined_prechange_anchor(self):
        text = (ROOT / "prereg_v3.2.md").read_text()
        self.assertIn("unbaselined events mean in `[-1,0)`", text)
        self.assertIn("K=50 remains the only authoritative", text)
        self.assertIn("Q2 precision requires", text)

    def test_v33_prereg_freezes_conditional_anchor_cache_and_nested_q2(self):
        text = (ROOT / "prereg_v3.3.md").read_text()
        self.assertIn("neural-dev-features-v1-*", text)
        self.assertIn("fold-independent", text)
        self.assertIn("AUC_cond", text)
        self.assertIn("at least three of five", text)
        self.assertIn("10**{-4,-3,-2,-1,0,1,2}", text)
        self.assertIn("sigmoid", text)
        self.assertIn("calibrator", text)
        self.assertIn("usable_positive", text)
        self.assertIn("no CONFIRM workflow", text)


if __name__ == "__main__":
    unittest.main()

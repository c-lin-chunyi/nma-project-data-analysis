import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


neural = load("neural", ROOT / "pipeline/verify-neural/neural.py")
sys.modules["neural"] = neural
v33 = load("v33", ROOT / "pipeline/verify-neural/v33.py")
sys.modules["v33"] = v33
v34 = load("v34", ROOT / "pipeline/verify-neural/v34.py")


def confirm_table():
    rows = []
    oeid = 100000
    for index in range(29):
        count = 5 if index < 14 else 4
        for session in range(count):
            rows.append({
                "ophys_experiment_id": oeid,
                "behavior_session_id": oeid + 100000,
                "ophys_container_id": 3000 + index,
                "mouse_id": 4000 + index,
                "project_code": "VisualBehavior",
                "session_type": f"OPHYS_{1 + session % 6}_images_A",
            })
            oeid += 1
    assert len(rows) == 130
    return pd.DataFrame(rows)


class ConfirmV34Tests(unittest.TestCase):
    def test_exact_manifest_accepts_130_29_29_active(self):
        confirm = confirm_table()
        experiments = confirm.sample(frac=1, random_state=4).copy()
        experiments["equipment_name"] = "MESO.1"
        result = v34.build_confirm_manifest(
            confirm, experiments, pd.DataFrame({"mouse_id": [999]}))
        self.assertEqual(len(result), 130)
        self.assertEqual(result.mouse_id.nunique(), 29)
        self.assertEqual(result.ophys_container_id.nunique(), 29)
        self.assertTrue(result.role.eq("active").all())

    def test_manifest_rejects_missing_duplicate_passive_and_dev_contamination(self):
        base = confirm_table()
        cases = {}
        cases["missing"] = base.iloc[:-1].copy()
        duplicate = base.copy()
        duplicate.loc[1, "ophys_experiment_id"] = duplicate.loc[0, "ophys_experiment_id"]
        cases["duplicate"] = duplicate
        passive = base.copy()
        passive.loc[0, "session_type"] += "_passive"
        cases["passive"] = passive
        for name, table in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                v34.validate_confirm_split(table)
        with self.assertRaisesRegex(ValueError, "DEV_contamination"):
            v34.validate_confirm_split(
                base, pd.DataFrame({"mouse_id": [base.mouse_id.iloc[0]]}))

    def test_primary_typed_outcomes_and_strict_boundary(self):
        self.assertEqual(v34.primary_decision(
            {"low": .60, "high": .70}, 23)["status"],
            "nonestimable_coverage")
        self.assertEqual(v34.primary_decision(
            {"low": None, "high": None}, 24)["status"],
            "nonestimable_interval")
        self.assertEqual(v34.primary_decision(
            {"low": .55, "high": .65}, 24)["status"],
            "confirmatory_not_supported")
        self.assertEqual(v34.primary_decision(
            {"low": .549, "high": .65}, 29)["status"],
            "confirmatory_not_supported")
        self.assertEqual(v34.primary_decision(
            {"low": .5500001, "high": .65}, 24)["status"],
            "confirmatory_supported")
        self.assertEqual(v34.primary_decision(
            {"low": .70, "high": .80}, 29, integrity=False)["status"],
            "pipeline_failure")

    def test_partial_seed_and_no_session_are_typed_nonestimable(self):
        metrics, error = v34.complete_q1_result({"auc": .8, "n_seeds": 9}, None)
        self.assertEqual(metrics, {})
        self.assertEqual(error, "partial_seed_completion")
        self.assertEqual(v34.primary_decision(
            {"mean": None, "low": None, "high": None}, 0)["status"],
            "nonestimable_coverage")

    def test_degenerate_bca_is_serializable_and_nonestimable(self):
        interval = v34.bca_interval(np.full(24, .6), seed=3305)
        self.assertIsNone(interval["low"])
        self.assertIsNone(interval["high"])
        self.assertEqual(v34.primary_decision(interval, 24)["status"],
                         "nonestimable_interval")
        json.dumps(interval, allow_nan=False)

    def test_access_receipt_allows_first_and_same_run_only(self):
        args = dict(run_id=123, commit="abc", freeze_release="confirm-v3.4-freeze-x",
                    freeze_sha256="f" * 64, split_manifest_sha256="s" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "access-receipt.json"
            self.assertEqual(v34.access_receipt(path, **args), "create")
            first = path.read_bytes()
            self.assertEqual(v34.access_receipt(path, **args), "resume")
            self.assertEqual(path.read_bytes(), first)
            for changed in ({"run_id": 124}, {"commit": "def"},
                            {"freeze_sha256": "x" * 64}):
                values = {**args, **changed}
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    v34.access_receipt(path, **values)

    def test_freeze_manifest_binds_all_pre_access_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prereg, requirements = root / "prereg.md", root / "requirements.txt"
            split, prior = root / "split.json", root / "v33.json"
            order = root / "order.json"
            prereg.write_text("frozen\n")
            requirements.write_text("package==1\n")
            split.write_text('{"schema":"split"}\n')
            prior.write_text('{"schema":"neural-dev-v3.3"}\n')
            order.write_text(json.dumps({
                "schema": "confirm-v3.4-gh-order-v1",
                "immutable_upstream_releases": {
                    "split": "split-lock",
                    "dev_v33_analysis": "neural-dev-v3.3-1",
                },
            }) + "\n")
            out = root / "freeze-manifest.json"
            self.assertEqual(v34.write_freeze_manifest(
                out, code_commit="abc", prereg=prereg,
                requirements=requirements, split_release="split-lock",
                split_manifest=split, v33_release="neural-dev-v3.3-1",
                v33_manifest=prior, workflow_order=order), 0)
            value = json.loads(out.read_text())
            self.assertEqual(value["schema"], "confirm-v3.4-freeze-v1")
            self.assertEqual(value["code_commit"], "abc")
            self.assertEqual(len(value["prereg_sha256"]), 64)
            self.assertEqual(len(value["v33_manifest_sha256"]), 64)
            self.assertEqual(len(value["workflow_order_sha256"]), 64)

    def test_confirm_cache_schema_rejects_dev_shard(self):
        manifest = pd.DataFrame([{
            "ophys_experiment_id": 1, "behavior_session_id": 2,
            "ophys_container_id": 3, "mouse_id": 4,
            "project_code": "VisualBehavior", "session_type": "OPHYS_1_images_A",
            "role": "active",
        }])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_ids = np.array([10, 11])
            cells = np.array([20, 21])
            with h5py.File(root / "1.features.h5", "w") as h5:
                h5.attrs["schema"] = neural.FEATURE_CACHE_SCHEMA
                h5.attrs["behavior_session_id"] = 2
                h5.create_dataset("trial_id", data=trial_ids)
                h5.create_dataset("cell_specimen_id", data=cells)
                for name in neural.FEATURE_DATASETS:
                    h5.create_dataset(name, data=np.zeros((2, 2), np.float32))
            pd.DataFrame({"trial_id": trial_ids}).to_parquet(
                root / "1.labels.parquet", index=False)
            pd.DataFrame({"trial_id": trial_ids}).to_parquet(
                root / "1.q2.parquet", index=False)
            (root / "1.feature-meta.json").write_text(json.dumps({
                "schema": neural.FEATURE_CACHE_SCHEMA,
                "identity": {"ophys_experiment_id": 1},
            }))
            failures = neural.feature_cache_failures(
                root, manifest, feature_schema=v34.FEATURE_SCHEMA)
            problems = failures[0]["problems"]
            self.assertIn("metadata_schema", problems)
            self.assertIn("h5_schema", problems)

    def test_v34_invokes_the_v33_q1_and_q2_kernels(self):
        source = (ROOT / "pipeline/verify-neural/v34.py").read_text()
        self.assertIs(v34.q1_kernel, v33._q1_session)
        self.assertIn("q1_kernel(item)", source)
        self.assertIn("v33._q2_session(item)", source)
        self.assertNotIn("def _blocked_auc", source)
        self.assertNotIn("def _q1_session", source)
        self.assertNotIn("def _q2_session", source)

    def test_exact_table_comparator_detects_any_q1_drift(self):
        expected = pd.DataFrame({"mouse_id": [1, 2], "auc": [.6, .7]})
        v34._assert_exact_table(expected.copy(), expected, ["mouse_id"], "mouse")
        changed = expected.copy()
        changed.loc[1, "auc"] = np.nextafter(.7, 1.0)
        with self.assertRaisesRegex(ValueError, "Q1 drift"):
            v34._assert_exact_table(changed, expected, ["mouse_id"], "mouse")

    def test_q2_output_cannot_change_q1_decision(self):
        sessions = []
        for mouse in range(24):
            sessions.append({"meta": {
                "ophys_experiment_id": 1000 + mouse,
                "behavior_session_id": 2000 + mouse,
                "mouse_id": 3000 + mouse,
                "project_code": "VisualBehavior", "novel": False,
                "miss_B": 20, "late_hit_B": 20,
                "behavioral_eligible": True,
            }})

        def intervals(values, *, seed):
            if seed == 3305:
                return {"mean": .70, "low": .56, "high": .80, "n_mice": 24}
            return {"mean": -99.0, "low": -100.0, "high": -98.0,
                    "n_mice": 24}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(v34, "_feature_sessions",
                                  return_value=(sessions, confirm_table())), \
                mock.patch.object(v34, "q1_kernel",
                                  return_value=({"auc": .70, "n_seeds": 10}, [], None)), \
                mock.patch.object(v33, "_q2_session",
                                  return_value=({"delta_log_loss": -99.0}, [], [], None)), \
                mock.patch.object(v34, "bca_interval", side_effect=intervals):
            out = Path(tmp) / "out"
            self.assertEqual(v34.scan(
                Path(tmp), Path(tmp) / "manifest.csv", out,
                feature_release="test", feature_manifest_sha256="f" * 64), 0)
            result = json.loads((out / "analysis-manifest.json").read_text())
            self.assertEqual(result["outcome"], "confirmatory_supported")
            self.assertEqual(result["secondary"]["delta_log_loss_mouse_interval"]["low"],
                             -100.0)
            self.assertIn("delta_brier", result["secondary"]["mouse_intervals"])
            self.assertIn("delta_auc", result["secondary"]["mouse_intervals"])
            self.assertFalse(result["secondary"]["changes_primary_decision"])

    def test_prereg_and_workflow_contract(self):
        prereg = (ROOT / "docs/prereg_v3.4.md").read_text()
        workflow = (ROOT / ".github/workflows/neural-confirm-v3.4.yml").read_text()
        freeze_workflow = (ROOT / ".github/workflows/confirm-v3.4-freeze.yml").read_text()
        order = json.loads((ROOT / "docs/confirm_v3.4_gh_order.json").read_text())
        for text in ("0.55", "24", "2,000", "3305", "3304",
                     "confirmatory_supported", "nonestimable_coverage"):
            self.assertIn(text, prereg)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("workflow_dispatch:", freeze_workflow)
        self.assertEqual(order["schema"], "confirm-v3.4-gh-order-v1")
        self.assertEqual(order["one_shot_job_order"], [
            "open", "behavior-pull", "labels", "features", "analyze", "publish"])
        self.assertIn("neural-dev-v3.3-29551296569", freeze_workflow)
        self.assertIn("neural-dev-features-v1-29482249873", freeze_workflow)
        self.assertIn("confirm-v3.4-freeze-${GITHUB_SHA}", freeze_workflow)
        self.assertIn("confirm_v3.4_gh_order.json", workflow)
        for dependency in (
                "needs: open", "needs: [open, behavior-pull]",
                "needs: [open, labels]", "needs: [open, features]",
                "needs: [open, analyze]"):
            self.assertIn(dependency, workflow)
        self.assertIn("environment: confirm-v3.4", workflow)
        self.assertIn("confirm-v3.4-access", workflow)
        self.assertIn("dev-q1-equivalence", workflow)
        self.assertLess(workflow.index("dev-q1-equivalence"),
                        workflow.index("Create or validate immutable pre-access receipt"))
        self.assertIn("neural-confirm-feature-cache-v1", prereg)
        self.assertIn("neural-confirm-v3.4-${FREEZE_SHA}", workflow)
        self.assertIn("Q1 primary and Q2 descriptive secondary unconditionally", workflow)
        self.assertNotIn("state_anchor", workflow)
        self.assertNotIn("pipeline/verify-neural/v4", workflow)
        self.assertNotIn("miss_threshold", workflow)
        inputs = workflow.split("permissions:", 1)[0]
        self.assertNotIn("K:", inputs)
        self.assertNotIn("C:", inputs)
        self.assertNotIn("SESOI:", inputs)


if __name__ == "__main__":
    unittest.main()

"""Unit, integration, and workflow-contract tests for the isolated v4 pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import yaml

from pipeline.v4.analysis import aggregate
from pipeline.v4.behavior import compile_behavior
from pipeline.v4.cache import materialize_experiment, sha256_file, validate_source_manifest
from pipeline.v4.constants import (
    CACHE_SCHEMA,
    CELL_SEEDS,
    EMISSION_ABORT,
    EMISSION_MISSING,
    EMISSION_TASK_RESPONSE,
    EMISSION_WITHHOLD,
)
from pipeline.v4.hazard import (
    NeuralTrial,
    apply_transform,
    build_risk_rows,
    causal_history,
    event_bin,
    fit_transform,
    one_se_hazard,
    raw_blocks,
    risk_bins,
)
from pipeline.v4.hmm import one_se_smallest


ROOT = Path(__file__).resolve().parents[1]


def _empty_trace(value_name: str) -> pd.DataFrame:
    return pd.DataFrame({"timestamps": pd.Series(dtype=float), value_name: pd.Series(dtype=float)})


def _behavior_tables(lick_times: list[float], *, auto_column: bool = True):
    starts = np.arange(0.0, 15.0, 3.0)
    trials = pd.DataFrame(
        {
            "trials_id": np.arange(5),
            "start_time": starts,
            "stop_time": starts + 2.0,
            "change_time": starts + 1.0,
            "hit": True,
            "miss": False,
            "false_alarm": False,
            "correct_reject": False,
            "initial_image_name": "a",
            "change_image_name": "b",
        }
    )
    if auto_column:
        trials["auto_rewarded"] = False
    return (
        trials,
        pd.DataFrame({"trials_id": np.arange(5), "start_time": starts, "omitted": False}),
        pd.DataFrame({"timestamps": lick_times}),
        pd.DataFrame({"timestamps": []}),
        _empty_trace("pupil_area"),
        _empty_trace("speed"),
    )


class BehaviorTests(unittest.TestCase):
    def test_emission_boundaries_and_exclusivity(self):
        tables = _behavior_tables(
            [
                0.9,       # before trial start: ignored
                0.0,       # trial 0 start: abort
                4.15,      # trial 1: response lower boundary
                7.75,      # trial 2: response upper boundary
                10.149,    # trial 3: pre-window competing event
                13.751,    # trial 4: outside response window
            ]
        )
        result = compile_behavior(*tables, neural_valid=np.ones(5, bool))
        np.testing.assert_array_equal(
            result.emission,
            [
                EMISSION_ABORT,
                EMISSION_TASK_RESPONSE,
                EMISSION_TASK_RESPONSE,
                EMISSION_WITHHOLD,
                EMISSION_WITHHOLD,
            ],
        )
        self.assertTrue(result.loc[0, "abort"])
        self.assertFalse(result.loc[0, "task_response"])
        self.assertTrue(result.loc[3, "prewindow_competing_event"])
        self.assertFalse(result.loc[3, "primary_risk_eligible"])

    def test_missing_auto_reward_field_retains_transition_row(self):
        result = compile_behavior(
            *_behavior_tables([1.2], auto_column=False),
            neural_valid=np.ones(5, bool),
        )
        self.assertTrue(np.all(result.emission.eq(EMISSION_MISSING)))
        self.assertEqual(len(result), 5)

    def test_conflicting_go_catch_is_typed_invalid_row(self):
        tables = list(_behavior_tables([]))
        tables[0].loc[0, "false_alarm"] = True
        result = compile_behavior(*tables, neural_valid=np.ones(5, bool))
        self.assertFalse(result.loc[0, "outcome_valid"])
        self.assertEqual(result.loc[0, "emission"], EMISSION_MISSING)


class CacheTests(unittest.TestCase):
    def test_ragged_actual_timestamp_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "cache"
            source.mkdir()
            experiment = 101
            trials, stim, licks, rewards, eye, running = _behavior_tables([])
            trials.loc[4, "change_time"] = np.nan
            for name, table in {
                "trials": trials,
                "stim": stim,
                "licks": licks,
                "rewards": rewards,
                "eye": eye,
                "running": running,
            }.items():
                table.to_parquet(source / f"{experiment}.{name}.parquet", index=False)
            timestamps = np.arange(-1.0, 15.0, 0.05)
            with h5py.File(source / f"{experiment}.neural.h5", "w") as h5:
                h5["ophys_timestamps"] = timestamps
                h5["cell_specimen_id"] = np.arange(60)
                h5["cell_roi_id"] = np.arange(100, 160)
                signal = np.arange(60 * len(timestamps), dtype=np.float32).reshape(60, -1)
                h5["events"] = signal
                h5["dff"] = signal / 10
            row = SimpleNamespace(
                ophys_experiment_id=experiment,
                behavior_session_id=201,
                ophys_container_id=301,
                mouse_id=401,
            )
            meta = materialize_experiment(
                source,
                output,
                row,
                neural_release="neural-dev-data-1",
                data_manifest_sha256="a" * 64,
            )
            self.assertEqual(meta["schema"], CACHE_SCHEMA)
            cached_trials = pd.read_parquet(output / f"{experiment}.trials.parquet")
            self.assertEqual(len(cached_trials), 5)
            self.assertFalse(cached_trials.loc[4, "neural_valid"])
            with h5py.File(output / f"{experiment}.time.h5") as h5:
                absolute = h5["frame_timestamp"][:]
                source_index = h5["source_frame_index"][:]
                np.testing.assert_array_equal(absolute, timestamps[source_index])
                self.assertEqual(h5["events"].shape[1], 60)
                self.assertTrue(np.all(np.diff(h5["frame_offsets"][:]) > 0))

    def test_manifest_active_passive_boundary_and_checksum_corruption(self):
        rows = []
        for mouse in range(10):
            for session in range(7):
                rows.append(
                    {
                        "ophys_experiment_id": mouse * 7 + session,
                        "behavior_session_id": mouse * 7 + session,
                        "ophys_container_id": mouse,
                        "mouse_id": mouse,
                        "role": "active" if session < 5 else "passive",
                    }
                )
        manifest = pd.DataFrame(rows)
        validate_source_manifest(manifest)
        with self.assertRaises(ValueError):
            validate_source_manifest(manifest[manifest.role.eq("active")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"original")
            first = sha256_file(path)
            path.write_bytes(b"corrupt")
            self.assertNotEqual(first, sha256_file(path))


def _hazard_behavior(first_lick: float) -> pd.DataFrame:
    row = {
        "trial_id": 1,
        "raw_trial_index": 5,
        "primary_risk_eligible": True,
        "first_post_change_lick": first_lick,
        "image_transition": "a->b",
        "previous_outcome": "withhold",
        "flashes_before_change": 4.0,
        "time_since_previous_change": 2.0,
        "time_since_previous_lick": 3.0,
        "time_since_previous_reward": np.nan,
        "session_position": 0.5,
        "preceding_omission": 0.0,
        "pre_change_pupil": np.nan,
        "pre_change_running": 1.0,
    }
    return pd.DataFrame([row])


class HazardTests(unittest.TestCase):
    def _neural(self):
        times = np.arange(-1.25, 0.75, 0.05)
        signal = np.arange(len(times) * 60, dtype=float).reshape(len(times), 60)
        return NeuralTrial(1, times, signal, signal / 10)

    def test_actual_bins_event_and_postlick_rows(self):
        neural = self._neural()
        left, right = risk_bins(neural.relative_time)
        self.assertTrue(np.all(np.diff(np.r_[left, right[-1]]) > 0))
        self.assertIsNotNone(event_bin(0.75, left, right))
        self.assertIsNone(event_bin(np.nan, left, right))
        rows = build_risk_rows(
            _hazard_behavior(0.31),
            np.array([[0.7, 0.3]]),
            {1: neural},
            experiment_id=10,
            seed=0,
            basis_count=2,
            signal="events",
        )
        self.assertEqual(int(rows.event.sum()), 1)
        event_row = int(np.flatnonzero(rows.event.to_numpy())[0])
        self.assertEqual(event_row, len(rows) - 1)
        self.assertTrue(np.allclose(rows.offset, np.log(rows.right - rows.left)))

    def test_future_neural_frames_do_not_change_earlier_features(self):
        trial = self._neural()
        left = np.array([0.15, 0.25, 0.35])
        before = causal_history(trial, left, np.arange(5), 3, signal="events")
        changed = trial.events.copy()
        changed[trial.relative_time >= 0.25] += 1e6
        after = causal_history(
            NeuralTrial(1, trial.relative_time, changed, changed),
            left,
            np.arange(5),
            3,
            signal="events",
        )
        np.testing.assert_array_equal(before[:2], after[:2])

    def test_training_only_transform_and_expanding_indices(self):
        neural = self._neural()
        rows = build_risk_rows(
            _hazard_behavior(np.nan),
            np.array([[0.7, 0.3]]),
            {1: neural},
            experiment_id=10,
            seed=0,
            basis_count=2,
            signal="events",
        )
        train = pd.concat([rows, rows.assign(trial_id=2, raw_trial_index=6)])
        _, transform, _ = fit_transform(train, "M1")
        medians = transform.medians.copy()
        test = rows.copy()
        test["pre_change_running"] = 1e12
        design, _ = apply_transform(test, transform)
        np.testing.assert_array_equal(medians, transform.medians)
        self.assertTrue(np.all(np.isfinite(design)))
        blocks = raw_blocks(23)
        for index in range(1, 5):
            self.assertLess(blocks[index - 1][-1], blocks[index][0])

    def test_one_se_averages_seeds_within_tuning_session(self):
        rows = []
        for tuning in (1, 2, 3):
            for basis in (1, 2):
                for penalty in (0.1, 1.0):
                    for seed in CELL_SEEDS:
                        score = 1.0 - 0.01 * basis + 0.001 * penalty + tuning * 1e-4
                        rows.append(
                            {
                                "tuning_session": tuning,
                                "basis_count": basis,
                                "penalty": penalty,
                                "cell_seed": seed,
                                "per_trial_loglik": score,
                                "status": "estimable",
                            }
                        )
        basis, penalty = one_se_hazard(pd.DataFrame(rows), model="M1")
        self.assertEqual(basis, 1)
        self.assertEqual(penalty, 1.0)
        self.assertEqual(
            one_se_smallest({1: 0.0, 2: 0.01}, {1: 0.0, 2: 0.02}), 1
        )


class AggregateAndWorkflowTests(unittest.TestCase):
    def test_aggregate_eight_mouse_coverage_and_closed_confirm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, output = root / "mice", root / "aggregate"
            rows = []
            for mouse in range(10):
                for session in range(7):
                    rows.append(
                        {
                            "ophys_experiment_id": mouse * 7 + session,
                            "behavior_session_id": mouse * 7 + session,
                            "ophys_container_id": mouse,
                            "mouse_id": mouse,
                            "role": "active" if session < 5 else "passive",
                        }
                    )
                shard = shards / str(mouse)
                shard.mkdir(parents=True)
                (shard / "mouse-manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "neural-dev-v4-mouse-v1",
                            "mouse_id": mouse,
                            "cache_release": "cache",
                            "cache_manifest_sha256": "c",
                            "prereg_sha256": "p",
                            "environment_sha256": "e",
                            "diagnostics_complete": True,
                        }
                    )
                    + "\n"
                )
                seed_rows = []
                if mouse < 8:
                    for seed in CELL_SEEDS:
                        seed_rows.append(
                            {
                                "mouse_id": mouse,
                                "behavior_session_id": mouse * 7,
                                "cell_seed": seed,
                                "status": "estimable",
                                "n_evaluated_trials": 20,
                                "delta_ll": mouse / 100 + seed / 10000,
                                "m2_minus_m1": 0.0,
                                "dff_delta_ll": 0.0,
                            }
                        )
                else:
                    seed_rows.append(
                        {
                            "mouse_id": mouse,
                            "behavior_session_id": mouse * 7,
                            "cell_seed": 0,
                            "status": "nonestimable",
                            "n_evaluated_trials": 0,
                            "delta_ll": np.nan,
                            "m2_minus_m1": np.nan,
                            "dff_delta_ll": np.nan,
                        }
                    )
                pd.DataFrame(seed_rows).to_parquet(shard / "session_seeds.parquet")
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            result = aggregate(
                shards,
                manifest,
                output,
                cache_release="cache",
                cache_manifest_sha256="c",
                prereg_sha256="p",
                environment_sha256="e",
            )
            self.assertEqual(result["coverage"]["estimable_mice"], 8)
            self.assertTrue(result["v4_1_eligible"])
            self.assertIsNone(result["numeric_sesoi"])
            self.assertFalse(result["confirm_ready"])
            self.assertFalse(result["confirm_data_accessed"])
            for mouse in range(10):
                pd.DataFrame(
                    [
                        {
                            "mouse_id": mouse,
                            "behavior_session_id": mouse * 7,
                            "cell_seed": 0,
                            "status": "nonestimable",
                            "n_evaluated_trials": 0,
                            "delta_ll": np.nan,
                            "m2_minus_m1": np.nan,
                            "dff_delta_ll": np.nan,
                        }
                    ]
                ).to_parquet(
                    shards / str(mouse) / "session_seeds.parquet", index=False
                )
            nonestimable = aggregate(
                shards,
                manifest,
                root / "aggregate-none",
                cache_release="cache",
                cache_manifest_sha256="c",
                prereg_sha256="p",
                environment_sha256="e",
            )
            self.assertEqual(nonestimable["status"], "nonestimable_mouse_coverage")
            self.assertEqual(nonestimable["coverage"]["estimable_mice"], 0)
            self.assertIsNone(nonestimable["primary"]["mean"])

    def test_action_and_prereg_static_contract(self):
        cache_action = (ROOT / ".github/workflows/neural-time-cache-v2.yml").read_text()
        v4_action = (ROOT / ".github/workflows/neural-dev-v4.yml").read_text()
        yaml.safe_load(cache_action)
        yaml.safe_load(v4_action)
        self.assertIn("neural-dev-data-* tag required", cache_action)
        self.assertIn("max-parallel: 10", cache_action)
        self.assertIn("max-parallel: 10", v4_action)
        self.assertIn("requirements-v4.txt", cache_action)
        self.assertIn("requirements-v4.txt", v4_action)
        self.assertIn("acceptance", v4_action)
        self.assertIn("dev", v4_action)
        self.assertIn("already exists publicly; r1 cannot be overwritten", v4_action)
        self.assertNotIn(".nwb", cache_action.lower() + v4_action.lower())
        self.assertNotIn("allensdk", cache_action.lower() + v4_action.lower())
        prereg = (ROOT / "docs/prereg_v4.md").read_text()
        self.assertIn("**Status:** DRAFT", prereg)
        self.assertNotIn("TODO", prereg)
        self.assertIn(r"p(z_t", prereg)
        self.assertIn(
            "The `0.30 s` fixed window, `B-engaged` label, ten-trial transition guard, 20/20",
            prereg,
        )
        self.assertIn("have no role\nin the v4 primary estimand", prereg)
        requirements = (ROOT / "requirements-v4.txt").read_text().lower()
        requirement_lines = [
            line for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(line.startswith("allensdk") for line in requirement_lines))

    def test_legacy_requirement_hashes(self):
        expected = {
            "requirements-pipeline.txt": "c6bb9cad80317490e75daa252276df7857a2d4c923a6a94804e9f3c778c7ce0c",
            "requirements-notebooks.txt": "ee28a08c8cf5a8be52f58016909de157cb2770e88af51b55d16c7a62cf93e2ab",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()

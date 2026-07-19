"""Unit, integration, and workflow-contract tests for the isolated v4 pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import yaml

from pipeline.v4.acceptance import run_acceptance
from pipeline.v4.analysis import aggregate, aggregate_targets, fit_mouse
from pipeline.v4.behavior import compile_behavior
from pipeline.v4.cache import (
    materialize_container,
    materialize_experiment,
    sha256_file,
    validate_source_manifest,
)
from pipeline.v4.constants import (
    CACHE_SCHEMA,
    CELL_SEEDS,
    EMISSION_ABORT,
    EMISSION_MISSING,
    EMISSION_TASK_RESPONSE,
    EMISSION_WITHHOLD,
    RIDGE_GRID,
)
from pipeline.v4.hazard import (
    _cloglog_value_derivative,
    _numpy_cloglog,
    NeuralTrial,
    apply_transform,
    build_risk_rows,
    causal_history,
    event_bin,
    fit_hazard,
    fit_transform,
    one_se_hazard,
    raw_blocks,
    risk_bins,
    ridge_mask,
)
from pipeline.v4.hmm import one_se_smallest
from pipeline.v4.target import (
    _checkpoint_group,
    _preflight_one,
    assemble_seed_frame,
    hazard_plan,
    require_eligible_sessions,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_COUNTS = (5, 3, 4, 4, 4, 6, 5, 9, 6, 4)


def _uneven_source_manifest() -> pd.DataFrame:
    rows = []
    experiment_id = 10_000
    for mouse, active_count in enumerate(ACTIVE_COUNTS):
        for role, count in (("active", active_count), ("passive", 2)):
            for _ in range(count):
                rows.append(
                    {
                        "ophys_experiment_id": experiment_id,
                        "behavior_session_id": experiment_id + 100_000,
                        "ophys_container_id": 2_000 + mouse,
                        "mouse_id": 3_000 + mouse,
                        "role": role,
                    }
                )
                experiment_id += 1
    return pd.DataFrame(rows)


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
        manifest = _uneven_source_manifest()
        validate_source_manifest(manifest)
        self.assertEqual(
            sorted(
                manifest[manifest.role.eq("active")]
                .groupby("mouse_id")
                .size()
                .tolist()
            ),
            sorted(ACTIVE_COUNTS),
        )
        with self.assertRaises(ValueError):
            validate_source_manifest(manifest[manifest.role.eq("active")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"original")
            first = sha256_file(path)
            path.write_bytes(b"corrupt")
            self.assertNotEqual(first, sha256_file(path))

    def test_materialize_container_uses_manifest_active_count(self):
        manifest = _uneven_source_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            def fake_materialize(_source, _out, row, **_kwargs):
                return {"ophys_experiment_id": int(row.ophys_experiment_id)}

            with patch(
                "pipeline.v4.cache.materialize_experiment",
                side_effect=fake_materialize,
            ):
                for container, expected in ((2_001, 3), (2_007, 9)):
                    report = materialize_container(
                        root / "source",
                        manifest_path,
                        root / f"cache-{container}",
                        container,
                        neural_release="neural-dev-data-1",
                        data_manifest_sha256="a" * 64,
                    )
                    self.assertEqual(report["n_expected"], expected)
                    self.assertEqual(report["n_complete"], expected)
                    self.assertTrue(report["complete"])

                with self.assertRaisesRegex(ValueError, "no active experiments"):
                    materialize_container(
                        root / "source",
                        manifest_path,
                        root / "cache-missing",
                        999_999,
                        neural_release="neural-dev-data-1",
                        data_manifest_sha256="a" * 64,
                    )


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

    def test_incomplete_candidate_does_not_invalidate_complete_candidate(self):
        rows = []
        for tuning in (1, 2):
            for penalty in (0.1, 1.0):
                for seed in CELL_SEEDS:
                    failed = penalty == 0.1 and tuning == 2 and seed == 3
                    rows.append(
                        {
                            "tuning_session": tuning,
                            "basis_count": 1,
                            "penalty": penalty,
                            "cell_seed": seed,
                            "per_trial_loglik": np.nan if failed else 1.0,
                            "status": "nonestimable" if failed else "estimable",
                        }
                    )
        basis, penalty = one_se_hazard(
            pd.DataFrame(rows), model="M1", eligible_sessions=(1, 2)
        )
        self.assertEqual((basis, penalty), (1, 1.0))

    def test_stable_cloglog_extreme_rows_have_no_numpy_warning(self):
        eta = np.array([-1e3, -36.0, 0.0, 36.0, 1e3])
        with np.errstate(all="raise"):
            event = _numpy_cloglog(eta, np.ones(len(eta), int))
            survival = _numpy_cloglog(
                np.array([-1e3, 0.0, 36.0, 650.0]),
                np.zeros(4, int),
            )
        self.assertTrue(np.all(np.isfinite(event)))
        self.assertTrue(np.all(np.isfinite(survival)))
        import jax.numpy as jnp

        value, gradient = _cloglog_value_derivative(
            jnp.asarray(eta), jnp.ones(len(eta), dtype=jnp.int32)
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(value))))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))

    def test_cloglog_analytic_gradient_matches_finite_difference(self):
        import jax.numpy as jnp

        eta = np.array([-5.0, -1.0, 0.0, 1.5])
        step = 1e-6
        for event in (0, 1):
            events = np.full(len(eta), event, int)
            _, gradient = _cloglog_value_derivative(
                jnp.asarray(eta), jnp.asarray(events)
            )
            finite = (
                _numpy_cloglog(eta + step, events)
                - _numpy_cloglog(eta - step, events)
            ) / (2 * step)
            np.testing.assert_allclose(
                np.asarray(gradient), finite, rtol=2e-7, atol=2e-9
            )

    def test_baseline_is_ridged_and_separation_fits_all_penalties(self):
        self.assertTrue(np.array_equal(ridge_mask(12), np.ones(12)))
        rows = []
        for trial in range(80):
            for bin_index in (0, 1):
                rows.append(
                    {
                        "trial_id": trial,
                        "raw_trial_index": trial,
                        "bin_index": bin_index,
                        "offset": np.log(0.05),
                        "event": int(bin_index == 1),
                        "image_transition": "a->b",
                        "previous_outcome": "withhold",
                        "state": np.array([0.5]),
                        "neural": np.empty(0),
                        "basis_energy": np.array([0.0]),
                        **{
                            name: 0.0
                            for name in (
                                "flashes_before_change",
                                "time_since_previous_change",
                                "time_since_previous_lick",
                                "time_since_previous_reward",
                                "session_position",
                                "preceding_omission",
                                "pre_change_pupil",
                                "pre_change_running",
                            )
                        },
                    }
                )
        frame = pd.DataFrame(rows)
        with np.errstate(all="raise"):
            fits = [
                fit_hazard(frame, model="M0", penalty=penalty)
                for penalty in RIDGE_GRID
            ]
        self.assertTrue(all(np.all(np.isfinite(fit.beta)) for fit in fits))
        self.assertTrue(all(fit.protection_count == 0 for fit in fits))


class AggregateAndWorkflowTests(unittest.TestCase):
    def test_target_aggregate_requires_exact_hash_matched_fifty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            for mouse in range(10):
                for session_index in range(5):
                    target = mouse * 10 + session_index
                    manifest_rows.append(
                        {
                            "ophys_experiment_id": target + 1000,
                            "behavior_session_id": target,
                            "ophys_container_id": mouse,
                            "mouse_id": mouse,
                            "role": "active",
                        }
                    )
                    shard = root / "targets" / str(target)
                    shard.mkdir(parents=True)
                    estimable = mouse < 8
                    seeds = pd.DataFrame(
                        [
                            {
                                "mouse_id": mouse,
                                "behavior_session_id": target,
                                "cell_seed": seed,
                                "status": (
                                    "estimable" if estimable else "nonestimable"
                                ),
                                "n_evaluated_trials": 20 if estimable else 0,
                                "delta_ll": 0.01 if estimable else np.nan,
                                "m2_minus_m1": (
                                    0.0 if estimable else np.nan
                                ),
                                "dff_delta_ll": (
                                    0.0 if estimable else np.nan
                                ),
                            }
                            for seed in CELL_SEEDS
                        ]
                    )
                    seeds.to_parquet(shard / "session_seeds.parquet", index=False)
                    seeds.to_parquet(
                        shard / "k1_hazard_sensitivity.parquet", index=False
                    )
                    pd.DataFrame(
                        [
                            {
                                "mouse_id": mouse,
                                "target_session": target,
                                "k2_minus_k1": 0.1,
                                "k3_minus_k2": 0.1,
                            }
                        ]
                    ).to_parquet(
                        shard / "behavior_sensitivity.parquet", index=False
                    )
                    pd.DataFrame(
                        columns=["analysis", "reason", "detail", "exception_type"]
                    ).to_parquet(shard / "typed_failures.parquet", index=False)
                    files = {
                        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in shard.glob("*.parquet")
                    }
                    (shard / "target-manifest.json").write_text(
                        json.dumps(
                            {
                                "schema": "neural-dev-v4-target-v1",
                                "method_revision": "r3",
                                "target_session": target,
                                "mouse_id": mouse,
                                "cache_release": "cache",
                                "cache_manifest_sha256": "c",
                                "hmm_release": "hmm",
                                "hmm_manifest_sha256": "h",
                                "hmm_prereg_sha256": "hp",
                                "hazard_prereg_sha256": "vp",
                                "environment_sha256": "e",
                                "code_commit": "code",
                                "status": "complete",
                                "diagnostics_complete": True,
                                "typed_reasons": [],
                                "files": files,
                                "numeric_sesoi": None,
                                "confirm_ready": False,
                                "confirm_data_accessed": False,
                            }
                        )
                        + "\n"
                    )
            source = root / "manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(source, index=False)
            result = aggregate_targets(
                root / "targets",
                source,
                root / "out",
                cache_release="cache",
                cache_manifest_sha256="c",
                hmm_prereg_sha256="hp",
                hazard_prereg_sha256="vp",
                environment_sha256="e",
                code_commit="code",
                hmm_release="hmm",
                hmm_manifest_sha256="h",
            )
            self.assertEqual(result["n_target_shards"], 50)
            self.assertEqual(result["coverage"]["estimable_mice"], 8)
            corrupt = root / "targets/0/session_seeds.parquet"
            corrupt.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                aggregate_targets(
                    root / "targets",
                    source,
                    root / "out-corrupt",
                    cache_release="cache",
                    cache_manifest_sha256="c",
                    hmm_prereg_sha256="hp",
                    hazard_prereg_sha256="vp",
                    environment_sha256="e",
                    code_commit="code",
                    hmm_release="hmm",
                    hmm_manifest_sha256="h",
                )

    def test_target_checkpoint_resumes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def compute():
                calls.append(1)
                return {"rows": pd.DataFrame({"value": [1]})}, {"kind": "test"}

            _, tables, resumed = _checkpoint_group(
                root, "candidate", {"target": 1}, compute
            )
            self.assertFalse(resumed)
            self.assertEqual(tables["rows"].value.iloc[0], 1)
            _, _, resumed = _checkpoint_group(
                root, "candidate", {"target": 1}, compute
            )
            self.assertTrue(resumed)
            self.assertEqual(len(calls), 1)
            (root / "groups/candidate/rows.parquet").write_bytes(b"corrupt")
            _, _, resumed = _checkpoint_group(
                root, "candidate", {"target": 1}, compute
            )
            self.assertFalse(resumed)
            self.assertEqual(len(calls), 2)

    def test_preflight_catches_zero_event_before_candidate_grid(self):
        session = {
            "behavior": pd.DataFrame({"raw_trial_index": np.arange(25)}),
            "cells": np.arange(50),
            "experiment_id": 1,
            "neural": {},
        }
        risk = pd.DataFrame(
            {
                "trial_id": np.arange(25),
                "raw_trial_index": np.arange(25),
                "event": np.zeros(25, int),
            }
        )
        with patch("pipeline.v4.target.build_risk_rows", return_value=risk):
            eligible, reason, detail, _, events = _preflight_one(
                session, np.ones((25, 1))
            )
        self.assertFalse(eligible)
        self.assertEqual(reason, "hazard_tuning_session_ineligible")
        self.assertEqual(detail, "hazard_no_training_event")
        self.assertEqual(events, 0)
        with self.assertRaisesRegex(
            ValueError, "hazard_tuning_insufficient_sessions"
        ):
            require_eligible_sessions([1])
        self.assertEqual(require_eligible_sessions([3, 2]), (2, 3))

    def test_secondary_failures_do_not_change_primary_seed_status(self):
        target = {
            "experiment_id": 1,
            "behavior_session_id": 2,
            "mouse_id": 3,
        }
        results = {
            ("primary_k2", "M0", "events", 0): pd.Series(
                {
                    "status": "estimable",
                    "per_trial_loglik": -1.0,
                    "n_evaluated_trials": 20,
                    "reason": None,
                }
            )
        }
        for seed in CELL_SEEDS:
            results[("primary_k2", "M1", "events", seed)] = pd.Series(
                {
                    "status": "estimable",
                    "per_trial_loglik": -0.9,
                    "n_evaluated_trials": 20,
                    "reason": None,
                }
            )
            results[("primary_k2", "M2", "events", seed)] = pd.Series(
                {
                    "status": "nonestimable",
                    "per_trial_loglik": np.nan,
                    "reason": "hazard_nonconvergence",
                }
            )
            results[("primary_k2", "M1", "dff", seed)] = pd.Series(
                {
                    "status": "nonestimable",
                    "per_trial_loglik": np.nan,
                    "reason": "hazard_nonrepresentable_prediction",
                }
            )
        frame = assemble_seed_frame(target, "primary_k2", results)
        self.assertTrue(frame.status.eq("estimable").all())
        self.assertTrue(frame.m2_status.eq("nonestimable").all())
        self.assertTrue(frame.dff_status.eq("nonestimable").all())

    def test_acceptance_has_no_recovery_or_selection_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance.json"
            result = run_acceptance(output)
            self.assertTrue(result["passed"])
            self.assertFalse(result["simulation_recovery_performed"])
            self.assertFalse(result["k_selection_simulation_performed"])
            self.assertEqual(result, json.loads(output.read_text()))

    def test_fit_mouse_uses_manifest_session_count(self):
        manifest = _uneven_source_manifest()
        mouse_id = 3_001
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            def fake_read_session(_cache, row):
                return {
                    "experiment_id": int(row.ophys_experiment_id),
                    "behavior_session_id": int(row.behavior_session_id),
                    "mouse_id": int(row.mouse_id),
                    "behavior": pd.DataFrame([{"trial_id_v4": 0}]),
                    "cells": np.array([], dtype=np.int64),
                    "neural": {},
                }

            with (
                patch(
                    "pipeline.v4.analysis._read_session",
                    side_effect=fake_read_session,
                ),
                patch(
                    "pipeline.v4.analysis.load_target_posteriors",
                    side_effect=ValueError("no_hazard_event"),
                ),
            ):
                hmm = root / "hmm"
                hmm.mkdir()
                hmm_manifest = {
                    "schema": "neural-dev-v4-hmm-release-v1",
                    "method_revision": "r2",
                    "primary_k": 2,
                    "k_selection_performed": False,
                    "cache_release": "cache",
                    "cache_manifest_sha256": "c",
                    "prereg_sha256": "p",
                    "environment_sha256": "e",
                }
                hmm_manifest_path = hmm / "hmm-release-manifest.json"
                hmm_manifest_path.write_text(json.dumps(hmm_manifest) + "\n")
                pd.DataFrame(
                    columns=[
                        "mouse_id",
                        "target_session",
                        "k1_per_trial_loglik",
                        "k2_per_trial_loglik",
                        "k3_per_trial_loglik",
                        "k2_minus_k1",
                        "k3_minus_k2",
                    ]
                ).to_parquet(hmm / "behavior_sensitivity.parquet", index=False)
                result = fit_mouse(
                    root / "cache",
                    manifest_path,
                    root / "mouse",
                    mouse_id=mouse_id,
                    cache_release="cache",
                    cache_manifest_sha256="c",
                    prereg_sha256="p",
                    environment_sha256="e",
                    hmm_checkpoints=hmm,
                    hmm_release="hmm",
                    hmm_manifest_sha256=hashlib.sha256(
                        hmm_manifest_path.read_bytes()
                    ).hexdigest(),
                )

            self.assertEqual(result["n_expected_sessions"], 3)
            self.assertEqual(len(result["failures"]), 3)
            self.assertFalse(result["confirm_ready"])

    def test_k1_sensitivity_failure_does_not_change_primary_status(self):
        manifest = _uneven_source_manifest()
        mouse_id = 3_001
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            def fake_read_session(_cache, row):
                return {
                    "experiment_id": int(row.ophys_experiment_id),
                    "behavior_session_id": int(row.behavior_session_id),
                    "mouse_id": int(row.mouse_id),
                    "behavior": pd.DataFrame([{"trial_id_v4": 0}]),
                    "cells": np.array([], dtype=np.int64),
                    "neural": {},
                }

            def fake_posteriors(
                _release, *, mouse_id, target_session, tuning_sessions
            ):
                return np.array([[0.4, 0.6]]), {
                    session_id: np.array([[0.5, 0.5]])
                    for session_id in tuning_sessions
                }

            def fake_tune(_target, target_hmm, _sessions, *, model):
                if target_hmm.selected_k == 1:
                    raise ValueError("hazard_no_training_event")
                return (None if model == "M0" else 1), 1.0, []

            def fake_evaluate(target, _hmm, *, selected, **_kwargs):
                return (
                    [
                        {
                            "mouse_id": int(target["mouse_id"]),
                            "behavior_session_id": int(
                                target["behavior_session_id"]
                            ),
                            "cell_seed": 0,
                            "n_evaluated_trials": 1,
                            "delta_ll": 0.0,
                            "m2_minus_m1": 0.0,
                            "dff_delta_ll": 0.0,
                            "status": "estimable",
                            "reason": None,
                        }
                    ],
                    [],
                    [],
                )

            hmm = root / "hmm"
            hmm.mkdir()
            hmm_manifest = {
                "schema": "neural-dev-v4-hmm-release-v1",
                "method_revision": "r2",
                "primary_k": 2,
                "k_selection_performed": False,
                "cache_release": "cache",
                "cache_manifest_sha256": "c",
                "prereg_sha256": "p",
                "environment_sha256": "e",
            }
            hmm_manifest_path = hmm / "hmm-release-manifest.json"
            hmm_manifest_path.write_text(json.dumps(hmm_manifest) + "\n")
            pd.DataFrame(
                [
                    {
                        "mouse_id": mouse_id,
                        "target_session": 1,
                        "k2_minus_k1": 0.0,
                        "k3_minus_k2": 0.0,
                    }
                ]
            ).to_parquet(hmm / "behavior_sensitivity.parquet", index=False)
            with (
                patch(
                    "pipeline.v4.analysis._read_session",
                    side_effect=fake_read_session,
                ),
                patch(
                    "pipeline.v4.analysis.load_target_posteriors",
                    side_effect=fake_posteriors,
                ),
                patch(
                    "pipeline.v4.analysis._tune_model",
                    side_effect=fake_tune,
                ),
                patch(
                    "pipeline.v4.analysis._evaluate_target",
                    side_effect=fake_evaluate,
                ),
            ):
                result = fit_mouse(
                    root / "cache",
                    manifest_path,
                    root / "mouse",
                    mouse_id=mouse_id,
                    cache_release="cache",
                    cache_manifest_sha256="c",
                    prereg_sha256="p",
                    environment_sha256="e",
                    hmm_checkpoints=hmm,
                    hmm_release="hmm",
                    hmm_manifest_sha256=hashlib.sha256(
                        hmm_manifest_path.read_bytes()
                    ).hexdigest(),
                )
            self.assertEqual(result["n_estimable_sessions"], 3)
            self.assertEqual(result["status"], "estimable")
            self.assertEqual(len(result["failures"]), 3)
            self.assertTrue(
                all(
                    row["analysis"] == "sensitivity_k1_no_state"
                    for row in result["failures"]
                )
            )

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
                            "schema": "neural-dev-v4-mouse-v3",
                            "method_revision": "r3",
                            "primary_k": 2,
                            "k_selection_performed": False,
                            "mouse_id": mouse,
                            "cache_release": "cache",
                            "cache_manifest_sha256": "c",
                            "prereg_sha256": "p",
                            "environment_sha256": "e",
                            "hmm_release": "hmm",
                            "hmm_manifest_sha256": "h",
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
                pd.DataFrame(seed_rows).to_parquet(
                    shard / "k1_hazard_sensitivity.parquet"
                )
                pd.DataFrame(
                    [
                        {
                            "mouse_id": mouse,
                            "target_session": mouse * 7,
                            "k1_per_trial_loglik": -1.2,
                            "k2_per_trial_loglik": -1.1,
                            "k3_per_trial_loglik": -1.0,
                            "k2_minus_k1": 0.1,
                            "k3_minus_k2": 0.1,
                        }
                    ]
                ).to_parquet(shard / "behavior_sensitivity.parquet", index=False)
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
                hmm_release="hmm",
                hmm_manifest_sha256="h",
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
                hmm_release="hmm",
                hmm_manifest_sha256="h",
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
        self.assertIn("Run future-information invariance acceptance", v4_action)
        self.assertNotIn("--profile registered", v4_action)
        self.assertNotIn("registered simulation", v4_action.lower())
        self.assertIn("is public and cannot be overwritten", v4_action)
        hmm_action = (ROOT / ".github/workflows/neural-dev-v4-hmm.yml").read_text()
        yaml.safe_load(hmm_action)
        self.assertIn("neural-dev-v4-hmm-", hmm_action)
        self.assertIn("--max-fit-keys 5", hmm_action)
        self.assertIn("if: always()", hmm_action)
        self.assertIn(
            "Restore atomic checkpoints from the resumable draft", hmm_action
        )
        self.assertIn(
            "Persist completed atomic fits to the resumable draft", hmm_action
        )
        self.assertIn("hmm-checkpoint-chunk-", hmm_action)
        self.assertIn("hmm_checkpoint_release", v4_action)
        self.assertIn("key=lambda part: part['name']", cache_action)
        self.assertIn("download_draft_asset", cache_action)
        self.assertIn("Accept: application/octet-stream", cache_action)
        self.assertIn("target-${TARGET}.tar.gz", v4_action)
        self.assertIn(
            "for container in json.loads(sys.argv[1]):",
            cache_action,
        )
        self.assertNotIn(r'print("\\n".join', cache_action)
        self.assertNotIn(
            'gh release download "$CACHE_TAG" -p "$archive"',
            cache_action,
        )
        self.assertIn("run/acceptance/environment-input.sha256", v4_action)
        self.assertNotIn("run/input/environment-input.sha256", v4_action)
        self.assertNotIn("len(group)==5", v4_action)
        action_text = (cache_action + v4_action + hmm_action).lower()
        self.assertNotIn(".nwb", action_text)
        self.assertNotIn("allensdk", action_text)
        prereg = (ROOT / "docs/prereg_v4.md").read_text()
        self.assertIn("**Status:** DRAFT", prereg)
        self.assertNotIn("TODO", prereg)
        self.assertIn(r"\(p(z_t", prereg)
        self.assertIn("ordinal-bin baseline", prereg)
        self.assertIn("hazard_no_complete_candidate", prereg)
        self.assertIn("Acceptance does not run recovery simulation", prereg)
        self.assertEqual(
            hashlib.sha256((ROOT / "docs/prereg_v4_r2.md").read_bytes()).hexdigest(),
            "015e0feec8ec9330ae72121a79c35578fbe82f374e161bda3b2ffceb083bf358",
        )
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

    def test_hazard_plan_is_exact_50_and_json_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            _uneven_source_manifest().to_csv(path, index=False)
            plan = hazard_plan(path)
            self.assertEqual(plan["n_targets"], 50)
            self.assertEqual(len(plan["targets"]), 50)
            json.dumps(plan)


if __name__ == "__main__":
    unittest.main()

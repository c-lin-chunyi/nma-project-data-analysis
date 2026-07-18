"""Direct GLM-HMM wrapper tests; requires the isolated v4 lock."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pipeline.v4.hmm import (
    _fit_one_start,
    _pad_sessions,
    fit_hmm,
    make_masked_model,
    predictive_state_probs,
    select_target_hmm,
    state_order,
)
from pipeline.v4.hmm_checkpoint import (
    FitSpec,
    _valid_checkpoint,
    _write_failure_checkpoint,
    _write_checkpoint,
    fit_specs,
)


def _session(seed: int, n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    emission = rng.integers(0, 3, n, dtype=np.int32)
    emission[5] = 3
    return pd.DataFrame(
        {
            "emission": emission,
            "go": rng.random(n) < 0.7,
            "outcome_valid": True,
            "novelty": rng.random(n) < 0.5,
            "previous_emission": np.r_[-1, emission[:-1]],
            "previous_reward": np.r_[np.nan, rng.random(n - 1) < 0.5],
            "session_position": np.linspace(0, 1, n),
        }
    )


class HMMWrapperTests(unittest.TestCase):
    def test_fixed_shape_dummy_and_jitted_m_step(self):
        sessions = {index: _session(index, n=24 + index) for index in range(2)}
        designs = [
            np.ones((len(frame), 11), float) for frame in sessions.values()
        ]
        emissions = [
            frame.emission.to_numpy(np.int32) for frame in sessions.values()
        ]
        y, x, structural, observed = _pad_sessions(
            emissions,
            designs,
            batch_size=3,
            time_size=30,
        )
        self.assertEqual(y.shape, (3, 30))
        self.assertFalse(structural[2].any())
        self.assertFalse(observed[2].any())
        self.assertEqual(float(x[2].sum()), 0.0)

        model = make_masked_model(1, 11)
        model.m_step = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unjitted M-step called")
        )
        fit = _fit_one_start(
            emissions,
            designs,
            k=1,
            seed=4100,
            max_iter=4,
            tolerance=1e9,
            model=model,
            fixed_shape=(3, 30),
        )
        self.assertTrue(fit.converged)

    def test_fixed_shape_padding_preserves_exact_sufficient_statistics(self):
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        frame = _session(44, n=24)
        design = np.ones((len(frame), 11), float)
        emission = frame.emission.to_numpy(np.int32)
        model = make_masked_model(2, 11)
        params, _ = model.initialize(jr.PRNGKey(9), method="prior")

        exact = _pad_sessions([emission], [design])
        padded = _pad_sessions(
            [emission], [design], batch_size=3, time_size=30
        )

        def e_step(values):
            y, x, structural, observed = values
            return model._masked_batch_e_jit(
                params,
                jnp.asarray(y),
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray(structural),
                jnp.asarray(observed),
            )

        exact_stats, exact_ll = e_step(exact)
        padded_stats, padded_ll = e_step(padded)
        np.testing.assert_allclose(
            np.asarray(padded_ll),
            np.r_[np.asarray(exact_ll), 0.0, 0.0],
            rtol=0,
            atol=1e-10,
        )
        exact_leaves = jax.tree_util.tree_leaves(exact_stats)
        padded_leaves = jax.tree_util.tree_leaves(padded_stats)
        self.assertEqual(len(exact_leaves), len(padded_leaves))
        for exact_leaf, padded_leaf in zip(exact_leaves, padded_leaves):
            exact_value = np.asarray(exact_leaf[0])
            padded_value = np.asarray(padded_leaf[0])
            if (
                padded_value.ndim
                and exact_value.ndim
                and padded_value.shape[1:] == exact_value.shape[1:]
                and padded_value.shape[0] > exact_value.shape[0]
            ):
                np.testing.assert_allclose(
                    padded_value[: exact_value.shape[0]],
                    exact_value,
                    rtol=0,
                    atol=1e-10,
                )
                np.testing.assert_array_equal(
                    padded_value[exact_value.shape[0] :],
                    np.zeros_like(padded_value[exact_value.shape[0] :]),
                )
            else:
                np.testing.assert_allclose(
                    padded_value,
                    exact_value,
                    rtol=0,
                    atol=1e-10,
                )
            np.testing.assert_array_equal(
                np.asarray(padded_leaf[1:]),
                np.zeros_like(np.asarray(padded_leaf[1:])),
            )

    def test_three_session_mouse_uses_fixed_k2(self):
        sessions = {index: _session(index + 100) for index in range(3)}
        result = select_target_hmm(
            sessions,
            0,
            seeds=(4100,),
            max_iter=4,
            tolerance=1e9,
        )
        self.assertEqual(result.selected_k, 2)
        self.assertEqual(set(result.inner_probs), {1, 2})
        self.assertEqual(result.target_probs.shape, (30, 2))
        self.assertFalse(result.selection_rows[0]["selection_performed"])
        specs = fit_specs(tuple(sessions), 99)
        self.assertEqual(len(specs), 12)
        self.assertEqual(
            sum(spec.k == 2 and len(spec.excluded_sessions) == 2 for spec in specs),
            3,
        )
        five_session_specs = fit_specs(tuple(range(5)), 100)
        self.assertEqual(len(five_session_specs), 25)
        self.assertEqual(
            sum(
                spec.k == 2 and len(spec.excluded_sessions) == 2
                for spec in five_session_specs
            ),
            10,
        )

    def test_k1_and_transition_preserving_internal_missing(self):
        sessions = {index: _session(index) for index in range(3)}
        fit = fit_hmm(
            sessions,
            k=1,
            seeds=(4100,),
            max_iter=4,
            tolerance=1e9,
        )
        probability = predictive_state_probs(fit, sessions[0])
        np.testing.assert_array_equal(probability, np.ones((30, 1)))
        self.assertTrue(fit.converged)

    def test_future_behavior_does_not_change_earlier_predictive_state(self):
        sessions = {index: _session(index + 10) for index in range(3)}
        fit = fit_hmm(
            sessions,
            k=1,
            seeds=(4100,),
            max_iter=4,
            tolerance=1e9,
        )
        before = predictive_state_probs(fit, sessions[0])
        changed = sessions[0].copy()
        values = changed.emission.to_numpy(np.int32, copy=True)
        values[20:] = (values[20:] + 1) % 3
        changed["emission"] = values
        after = predictive_state_probs(fit, changed)
        np.testing.assert_array_equal(before[:21], after[:21])

    def test_atomic_checkpoint_reorders_posterior_and_detects_corruption(self):
        sessions = {index: _session(index + 200) for index in range(3)}
        fit = fit_hmm(
            {0: sessions[0], 1: sessions[1]},
            k=2,
            seeds=(4100,),
            max_iter=4,
            tolerance=1e9,
            fixed_shape=(2, 30),
        )
        spec = FitSpec(99, 2, (2,), "primary_outer")
        provenance = {
            "cache_release": "neural-dev-time-v2-test",
            "cache_manifest_sha256": "cache",
            "prereg_sha256": "prereg",
            "environment_sha256": "environment",
        }
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.fit_id
            _write_checkpoint(
                destination,
                spec,
                fit,
                {2: sessions[2]},
                fixed_shape=(2, 30),
                provenance=provenance,
            )
            self.assertTrue(_valid_checkpoint(destination, spec, provenance))
            parameters = np.load(
                destination / "parameters.npz", allow_pickle=False
            )
            order = state_order(fit, 11)
            np.testing.assert_array_equal(parameters["state_order"], order)
            expected = predictive_state_probs(fit, sessions[2])[:, order]
            stored = pd.read_parquet(destination / "predictive.parquet")
            np.testing.assert_array_equal(
                stored[["state_0", "state_1"]].to_numpy(float), expected
            )
            predictive_path = destination / "predictive.parquet"
            predictive_path.write_bytes(predictive_path.read_bytes() + b"corrupt")
            self.assertFalse(_valid_checkpoint(destination, spec, provenance))

            failure = Path(temporary) / "failed"
            _write_failure_checkpoint(
                failure,
                spec,
                training_sessions=(0, 1),
                fixed_shape=(2, 30),
                reason="hmm_no_converged_initialization",
                detail="hmm_no_converged_initialization",
                provenance=provenance,
            )
            self.assertTrue(_valid_checkpoint(failure, spec, provenance))
            self.assertFalse((failure / "predictive.parquet").exists())


if __name__ == "__main__":
    unittest.main()

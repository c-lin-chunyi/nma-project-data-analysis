"""Direct GLM-HMM wrapper tests; requires the isolated v4 lock."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.v4.hmm import fit_hmm, predictive_state_probs


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


if __name__ == "__main__":
    unittest.main()

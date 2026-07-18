"""Implementation invariance checks for the v4 analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .behavior import hmm_design
from .hazard import NeuralTrial, causal_history
from .hmm import make_masked_model, predictive_state_probs


def _behavior_fixture(n_trials: int = 80) -> pd.DataFrame:
    """Return a deterministic fixture without a latent-state recovery target."""

    trial = np.arange(n_trials)
    emission = (trial % 3).astype(np.int32)
    previous_reward = ((trial[:-1] // 2) % 2).astype(float)
    return pd.DataFrame(
        {
            "go": trial % 2 == 0,
            "outcome_valid": np.ones(n_trials, bool),
            "novelty": (trial // 2) % 2 == 0,
            "previous_reward": np.r_[np.nan, previous_reward],
            "session_position": np.linspace(0.0, 1.0, n_trials),
            "emission": emission,
            "previous_emission": np.r_[-1, emission[:-1]],
        }
    )


def _future_information_checks() -> dict:
    behavior = _behavior_fixture()
    design, _ = hmm_design(behavior)
    model = make_masked_model(2, design.shape[1])

    import jax.numpy as jnp
    import jax.random as jr

    transition = np.array([[0.94, 0.06], [0.06, 0.94]], float)
    weights = np.zeros((2, 2, design.shape[1]), float)
    weights[0, :, 0] = np.array([1.0, -1.0])
    weights[1, :, 0] = np.array([-1.0, 1.0])
    weights[0, :, 1] = np.array([-2.0, 2.0])
    weights[1, :, 1] = np.array([2.0, -2.0])
    params, _ = model.initialize(
        jr.PRNGKey(0),
        initial_probs=jnp.array([0.5, 0.5], dtype=jnp.float64),
        transition_matrix=jnp.asarray(transition, dtype=jnp.float64),
        emission_weights=jnp.asarray(weights, dtype=jnp.float64),
    )
    fit = type(
        "KnownFit",
        (),
        {
            "model": model,
            "params": params,
            "scaler": {
                "position_mean": float(behavior.session_position.mean()),
                "position_scale": float(behavior.session_position.std(ddof=1)),
            },
            "k": 2,
        },
    )()

    original = predictive_state_probs(fit, behavior)
    changed = behavior.copy()
    changed_values = changed.emission.to_numpy(np.int32, copy=True)
    changed_values[50:] = (changed_values[50:] + 1) % 3
    changed["emission"] = changed_values
    changed_probability = predictive_state_probs(fit, changed)
    behavior_exact = np.array_equal(original[:51], changed_probability[:51])

    raw_emissions = behavior.emission.to_numpy(np.int32, copy=True)
    raw_emissions[20] = 3
    raw_probability = np.asarray(
        model.predict_state_probs(
            params,
            jnp.asarray(raw_emissions),
            jnp.asarray(design, dtype=jnp.float64),
        )
    )
    missing_transition = np.allclose(
        raw_probability[21],
        raw_probability[20] @ transition,
        rtol=0,
        atol=1e-14,
    )

    times = np.arange(-1.25, 0.75, 0.025, dtype=float)
    signal = np.sin(times[:, None] * np.arange(1, 4)[None, :])
    trial = NeuralTrial(1, times, signal.copy(), signal.copy())
    left = np.array([0.15, 0.20, 0.30, 0.40])
    before = causal_history(trial, left, np.arange(3), 3, signal="events")
    modified = signal.copy()
    modified[times >= 0.30] += 1000.0
    trial_changed = NeuralTrial(1, times, modified, modified)
    after = causal_history(trial_changed, left, np.arange(3), 3, signal="events")
    neural_exact = np.array_equal(before[:3], after[:3])
    return {
        "predict_state_probs_callable": callable(model.predict_state_probs),
        "future_behavior_atol_0_rtol_0": bool(behavior_exact),
        "future_neural_atol_0_rtol_0": bool(neural_exact),
        "internal_missing_transition_preserved": bool(missing_transition),
        "jax_x64": bool(jnp.ones(1, dtype=jnp.float64).dtype == jnp.float64),
        "passed": bool(behavior_exact and neural_exact and missing_transition),
    }


def run_acceptance(output: Path) -> dict:
    """Write the non-simulation implementation-acceptance report."""

    checks = _future_information_checks()
    report = {
        "schema": "neural-dev-v4-acceptance-v2",
        "profile": "implementation_invariance",
        "simulation_recovery_performed": False,
        "k_selection_simulation_performed": False,
        "implementation_invariance": checks,
        "passed": bool(checks["passed"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"v4 acceptance failed; see {output}")
    return report

"""Registered simulation recovery and future-information invariance checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .behavior import hmm_design
from .constants import HMM_SEEDS
from .hazard import NeuralTrial, causal_history
from .hmm import fit_hmm, make_masked_model, predictive_state_probs, select_target_hmm


@dataclass(frozen=True)
class AcceptanceProfile:
    k2_trials: int
    k1_trials: int
    starts: tuple[int, ...]
    max_iter: int
    tolerance: float


PROFILES = {
    "registered": AcceptanceProfile(1500, 800, HMM_SEEDS, 500, 1e-4),
    # The fast profile exercises identical code paths and is only for local
    # unit/integration testing; it cannot make the preregistration frozen.
    "fast": AcceptanceProfile(120, 100, HMM_SEEDS[:2], 80, 0.1),
}


def _session_covariates(n_trials: int, rng: np.random.Generator) -> dict:
    return {
        "go": rng.random(n_trials) < 0.5,
        "outcome_valid": np.ones(n_trials, bool),
        "novelty": rng.random(n_trials) < 0.5,
        "previous_reward": np.r_[np.nan, rng.random(n_trials - 1) < 0.55],
        "session_position": np.linspace(0.0, 1.0, n_trials),
    }


def _simulate_sessions(
    *,
    k: int,
    n_sessions: int,
    n_trials: int,
    seed: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    transition = (
        np.array([[0.94, 0.06], [0.06, 0.94]], float)
        if k == 2
        else np.ones((1, 1), float)
    )
    # Two strongly separated behavioral regimes. Class 0 is the implicit
    # softmax baseline; rows correspond to explicit classes 1 and 2.
    weights = np.zeros((k, 2, 11), float)
    if k == 2:
        # State-specific crossover in the scheduled-condition effect.  A
        # previous-emission main effect cannot reproduce this XOR-like mapping.
        weights[0, :, 0] = np.array([3.0, -3.0])
        weights[1, :, 0] = np.array([-3.0, 3.0])
        weights[0, :, 1] = np.array([-6.0, 6.0])
        weights[1, :, 1] = np.array([6.0, -6.0])
    else:
        weights[0, :, 0] = np.array([0.45, -0.35])
        weights[0, :, 1] = np.array([0.2, -0.15])

    shells = [_session_covariates(n_trials, rng) for _ in range(n_sessions)]
    positions = np.concatenate([shell["session_position"] for shell in shells])
    position_mean = float(positions.mean())
    position_scale = float(positions.std(ddof=1))
    sessions: dict[int, pd.DataFrame] = {}
    states: dict[int, np.ndarray] = {}
    for session_id, shell in enumerate(shells):
        z = np.zeros(n_trials, np.int32)
        y = np.zeros(n_trials, np.int32)
        previous = -1
        z[0] = int(rng.integers(k))
        for trial in range(n_trials):
            if trial:
                z[trial] = int(rng.choice(k, p=transition[z[trial - 1]]))
            previous_reward = shell["previous_reward"][trial]
            x = np.array(
                [
                    1.0,
                    float(shell["go"][trial]),
                    float(shell["novelty"][trial]),
                    0.0,
                    float(previous == 0),
                    float(previous == 1),
                    float(previous == 2),
                    float(previous < 0),
                    0.0 if np.isnan(previous_reward) else previous_reward,
                    float(np.isnan(previous_reward)),
                    (shell["session_position"][trial] - position_mean)
                    / position_scale,
                ],
                float,
            )
            logits = np.r_[0.0, weights[z[trial]] @ x]
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            y[trial] = int(rng.choice(3, p=probability))
            previous = int(y[trial])
        frame = pd.DataFrame(
            {
                **shell,
                "emission": y,
                "previous_emission": np.r_[-1, y[:-1]],
            }
        )
        sessions[session_id] = frame
        states[session_id] = z
    return sessions, states, transition, weights


def _emission_grid() -> np.ndarray:
    rows = []
    for go in (0.0, 1.0):
        for novelty in (0.0, 1.0):
            for previous in (0, 1, 2):
                for previous_reward in (0.0, 1.0):
                    row = np.zeros(11, float)
                    row[0] = 1.0
                    row[1] = go
                    row[2] = novelty
                    row[4 + previous] = 1.0
                    row[8] = previous_reward
                    rows.append(row)
    return np.asarray(rows)


def _probabilities_from_weights(weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    explicit = np.einsum("kcm,tm->ktc", weights, grid)
    logits = np.concatenate([np.zeros((*explicit.shape[:2], 1)), explicit], axis=2)
    logits -= logits.max(axis=2, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=2, keepdims=True)


def _fit_probabilities(fit, grid: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    return np.asarray(
        jax.vmap(
            lambda state: jax.vmap(
                lambda x: fit.model.emission_component.distribution(
                    fit.params.emissions, state, x
                ).probs_parameter()
            )(jnp.asarray(grid, dtype=jnp.float64))
        )(jnp.arange(fit.k)),
        float,
    )


def _balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    labels = np.unique(truth)
    return float(np.mean([np.mean(predicted[truth == label] == label) for label in labels]))


def _k2_recovery(profile: AcceptanceProfile) -> dict:
    sessions, states, transition, weights = _simulate_sessions(
        k=2, n_sessions=5, n_trials=profile.k2_trials, seed=4102
    )
    training = {key: value for key, value in sessions.items() if key != 4}
    fit = fit_hmm(
        training,
        k=2,
        seeds=profile.starts,
        max_iter=profile.max_iter,
        tolerance=profile.tolerance,
    )
    grid = _emission_grid()
    truth_prob = _probabilities_from_weights(weights, grid)
    fit_prob = _fit_probabilities(fit, grid)
    cost = np.sqrt(
        np.mean((truth_prob[:, None, :, :] - fit_prob[None, :, :, :]) ** 2, axis=(2, 3))
    )
    true_order, fit_order = linear_sum_assignment(cost)
    if not np.array_equal(true_order, np.arange(2)):
        raise AssertionError("unexpected truth assignment order")
    fitted_transition = np.asarray(fit.params.transitions.transition_matrix, float)
    fitted_transition = fitted_transition[np.ix_(fit_order, fit_order)]
    transition_error = float(np.max(np.abs(fitted_transition - transition)))
    emission_rmse = float(
        np.sqrt(np.mean((fit_prob[fit_order] - truth_prob) ** 2))
    )

    import jax.numpy as jnp

    heldout = sessions[4]
    design, _ = hmm_design(
        heldout,
        position_mean=fit.scaler["position_mean"],
        position_scale=fit.scaler["position_scale"],
    )
    posterior = np.asarray(
        fit.model.smoother(
            fit.params,
            jnp.asarray(heldout.emission.to_numpy(np.int32)),
            jnp.asarray(design, dtype=jnp.float64),
        ).smoothed_probs
    )
    inverse = np.empty(2, int)
    inverse[fit_order] = np.arange(2)
    matched_prediction = inverse[np.argmax(posterior, axis=1)]
    accuracy = _balanced_accuracy(states[4], matched_prediction)
    return {
        "transition_max_abs_error": transition_error,
        "transition_threshold": 0.08,
        "emission_probability_rmse": emission_rmse,
        "emission_threshold": 0.08,
        "heldout_smoothed_state_balanced_accuracy": accuracy,
        "accuracy_threshold": 0.80,
        "passed": bool(
            transition_error <= 0.08 and emission_rmse <= 0.08 and accuracy >= 0.80
        ),
    }


def _k1_selection(profile: AcceptanceProfile) -> dict:
    sessions, _, _, _ = _simulate_sessions(
        k=1, n_sessions=5, n_trials=profile.k1_trials, seed=4101
    )
    result = select_target_hmm(
        sessions,
        4,
        seeds=profile.starts,
        max_iter=profile.max_iter,
        tolerance=profile.tolerance,
    )
    return {
        "selected_k": int(result.selected_k),
        "expected_k": 1,
        "candidate_grid": [1, 2, 3, 4],
        "n_starts": len(profile.starts),
        "passed": result.selected_k == 1,
    }


def _future_information_checks() -> dict:
    sessions, _, transition, weights = _simulate_sessions(
        k=2, n_sessions=1, n_trials=80, seed=4199
    )
    behavior = sessions[0]
    design, _ = hmm_design(behavior)
    model = make_masked_model(2, design.shape[1])
    import jax.numpy as jnp
    import jax.random as jr

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


def run_acceptance(output: Path, *, profile: str = "registered") -> dict:
    """Run the named suite and always write a machine-readable report."""

    if profile not in PROFILES:
        raise ValueError(f"unknown acceptance profile {profile!r}")
    config = PROFILES[profile]
    report = {
        "schema": "neural-dev-v4-acceptance-v1",
        "profile": profile,
        "registered_parameters": {
            "k2_sessions": 5,
            "k2_trials_per_session": config.k2_trials,
            "k1_sessions": 5,
            "k1_trials_per_session": config.k1_trials,
            "starts": len(config.starts),
            "max_em_iterations": config.max_iter,
            "em_tolerance": config.tolerance,
        },
        "k2_recovery": None,
        "k1_negative_control": None,
        "future_information_invariance": None,
        "passed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["k2_recovery"] = _k2_recovery(config)
        if profile != "registered":
            recovery = report["k2_recovery"]
            recovery["registered_thresholds_passed"] = recovery["passed"]
            recovery["passed"] = bool(
                np.all(
                    np.isfinite(
                        [
                            recovery["transition_max_abs_error"],
                            recovery["emission_probability_rmse"],
                            recovery["heldout_smoothed_state_balanced_accuracy"],
                        ]
                    )
                )
            )
        report["k1_negative_control"] = _k1_selection(config)
        report["future_information_invariance"] = _future_information_checks()
        report["passed"] = all(
            report[key]["passed"]
            for key in (
                "k2_recovery",
                "k1_negative_control",
                "future_information_invariance",
            )
        )
    except Exception as exc:
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"v4 acceptance failed; see {output}")
    return report

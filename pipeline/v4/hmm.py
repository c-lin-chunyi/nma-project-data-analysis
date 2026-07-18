"""Behavior-only multinomial GLM-HMM with transition-preserving missing rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import numpy as np

from .behavior import hmm_design
from .constants import (
    EMISSION_MISSING,
    HMM_K_GRID,
    HMM_MAX_ITER,
    HMM_SEEDS,
    HMM_TOL,
)


def _jax_modules():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother
    from glmhmmt import SoftmaxGLMHMM

    return jax, jnp, jr, optax, hmm_two_filter_smoother, SoftmaxGLMHMM


@lru_cache(maxsize=None)
def make_masked_model(num_states: int, input_dim: int):
    """Construct a SoftmaxGLMHMM whose E-step distinguishes padding and missing."""

    jax, jnp, _, optax, smoother, base = _jax_modules()

    class TransitionPreservingSoftmaxGLMHMM(base):
        def __init__(self):
            super().__init__(
                num_states=num_states,
                num_classes=3,
                emission_input_dim=input_dim,
                transition_input_dim=0,
                initial_probs_concentration=1.1,
                transition_matrix_concentration=1.1,
                transition_matrix_stickiness=0.0,
                weight_scale=1.0,
                baseline_class_idx=0,
                m_step_optimizer=optax.adam(1e-2),
                m_step_num_iters=100,
            )
            self._masked_batch_e_jit = jax.jit(
                jax.vmap(self.masked_e_step, in_axes=(None, 0, 0, 0, 0))
            )
            self._masked_single_e_jit = jax.jit(self.masked_e_step)

        def masked_e_step(
            self, params, emissions, inputs, structural_mask, observed_mask
        ):
            pi0 = self.initial_component._compute_initial_probs(params.initial, inputs)
            transition = self.transition_component._compute_transition_matrices(
                params.transitions, inputs
            )
            if transition.ndim == 2:
                transition = jnp.broadcast_to(
                    transition[None, :, :],
                    (emissions.shape[0] - 1, self.num_states, self.num_states),
                )
            likelihoods = self.emission_component._compute_conditional_logliks(
                params.emissions, emissions, inputs
            )
            likelihoods = jnp.where(
                (structural_mask & observed_mask)[:, None], likelihoods, 0.0
            )
            posterior = smoother(pi0, transition, likelihoods)

            if posterior.trans_probs.ndim == 2:
                # Dynamax pre-sums pairwise probabilities for homogeneous
                # transitions. This branch is exact only for an unpadded
                # session; padded batched values are replaced below.
                transition_probs = posterior.trans_probs
            else:
                structural_transition = (
                    structural_mask[:-1] & structural_mask[1:]
                )[:, None, None]
                transition_probs = jnp.where(
                    structural_transition, posterior.trans_probs, 0.0
                ).sum(axis=0)
            transition_post = posterior._replace(trans_probs=transition_probs)
            emission_post = posterior._replace(
                smoothed_probs=jnp.where(
                    (structural_mask & observed_mask)[:, None],
                    posterior.smoothed_probs,
                    0.0,
                )
            )
            initial_stats = self.initial_component.collect_suff_stats(
                params.initial, posterior, inputs
            )
            transition_stats = self.transition_component.collect_suff_stats(
                params.transitions, transition_post, inputs
            )
            emission_stats = self.emission_component.collect_suff_stats(
                params.emissions, emission_post, emissions, inputs
            )
            return (
                initial_stats,
                transition_stats,
                emission_stats,
            ), posterior.marginal_loglik

    return TransitionPreservingSoftmaxGLMHMM()


@dataclass
class HMMFit:
    model: object
    params: object
    scaler: dict
    k: int
    seed: int
    marginal_loglik: float
    likelihood_trace: list[float]
    converged: bool
    training_session_ids: tuple[int, ...] = ()
    all_starts: list[dict] = field(default_factory=list)


def _pad_sessions(
    emissions: Sequence[np.ndarray], designs: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not emissions:
        raise ValueError("at least one session is required")
    maximum = max(map(len, emissions))
    dimension = designs[0].shape[1]
    y = np.full((len(emissions), maximum), EMISSION_MISSING, np.int32)
    x = np.zeros((len(emissions), maximum, dimension), np.float64)
    structural = np.zeros((len(emissions), maximum), bool)
    observed = np.zeros((len(emissions), maximum), bool)
    for index, (session_y, session_x) in enumerate(zip(emissions, designs)):
        length = len(session_y)
        y[index, :length] = session_y
        x[index, :length] = session_x
        structural[index, :length] = True
        observed[index, :length] = session_y != EMISSION_MISSING
    return y, x, structural, observed


def _fit_one_start(
    emissions: Sequence[np.ndarray],
    designs: Sequence[np.ndarray],
    *,
    k: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    model=None,
) -> HMMFit:
    jax, jnp, jr, _, _, _ = _jax_modules()
    model = model if model is not None else make_masked_model(k, designs[0].shape[1])
    # TFP's Dirichlet distribution rejects an event dimension of one.  K=1 is
    # nevertheless a registered (and scientifically important) null model, so
    # initialize its degenerate initial/transition probabilities explicitly.
    initialize_kwargs = (
        {
            "initial_probs": jnp.ones(1, dtype=jnp.float64),
            "transition_matrix": jnp.ones((1, 1), dtype=jnp.float64),
        }
        if k == 1
        else {}
    )
    params, props = model.initialize(
        jr.PRNGKey(seed), method="prior", **initialize_kwargs
    )
    y, x, structural, observed = _pad_sessions(emissions, designs)
    yj, xj = jnp.asarray(y), jnp.asarray(x, dtype=jnp.float64)
    sj, oj = jnp.asarray(structural), jnp.asarray(observed)
    m_state = model.initialize_m_step_state(params, props)
    trace: list[float] = []
    converged_steps = 0
    for _ in range(max_iter):
        stats, likelihoods = model._masked_batch_e_jit(params, yj, xj, sj, oj)
        # The homogeneous-transition smoother returns pairwise probabilities
        # already summed over time, so padding cannot be removed afterwards.
        # Recompute only that small statistic on each exact-length session.
        transition_stats = []
        for session_y, session_x in zip(emissions, designs):
            length = len(session_y)
            exact_stats, _ = model._masked_single_e_jit(
                params,
                jnp.asarray(session_y, dtype=jnp.int32),
                jnp.asarray(session_x, dtype=jnp.float64),
                jnp.ones(length, dtype=bool),
                jnp.asarray(session_y != EMISSION_MISSING),
            )
            transition_stats.append(exact_stats[1])
        stats = (stats[0], jnp.stack(transition_stats), stats[2])
        value = float(jnp.sum(likelihoods))
        if not np.isfinite(value):
            break
        trace.append(value)
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tolerance:
            converged_steps += 1
        else:
            converged_steps = 0
        if converged_steps >= 2:
            break
        params, m_state = model.m_step(params, props, stats, m_state)
    return HMMFit(
        model=model,
        params=params,
        scaler={},
        k=k,
        seed=seed,
        marginal_loglik=trace[-1] if trace else float("-inf"),
        likelihood_trace=trace,
        converged=converged_steps >= 2,
    )


def fit_hmm(
    training: Mapping[int, object],
    *,
    k: int,
    seeds: Sequence[int] = HMM_SEEDS,
    max_iter: int = HMM_MAX_ITER,
    tolerance: float = HMM_TOL,
) -> HMMFit:
    """Fit all registered starts and retain the best converged data likelihood."""

    if not training:
        raise ValueError("hmm_insufficient_training_sessions")
    positions = np.concatenate(
        [frame.session_position.to_numpy(float) for frame in training.values()]
    )
    mean = float(np.mean(positions))
    scale = float(np.std(positions, ddof=1)) if len(positions) > 1 else 1.0
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    scaler = {"position_mean": mean, "position_scale": scale}
    session_ids = sorted(training)
    emissions = [
        training[session_id].emission.to_numpy(np.int32) for session_id in session_ids
    ]
    designs = [
        hmm_design(
            training[session_id],
            position_mean=mean,
            position_scale=scale,
        )[0]
        for session_id in session_ids
    ]
    model = make_masked_model(k, designs[0].shape[1])
    fits = [
        _fit_one_start(
            emissions,
            designs,
            k=k,
            seed=int(seed),
            max_iter=max_iter,
            tolerance=tolerance,
            model=model,
        )
        for seed in seeds
    ]
    converged = [
        fit for fit in fits if fit.converged and np.isfinite(fit.marginal_loglik)
    ]
    if not converged:
        raise ValueError("hmm_no_converged_initialization")
    best = max(converged, key=lambda fit: fit.marginal_loglik)
    best.scaler = scaler
    best.training_session_ids = tuple(map(int, session_ids))
    best.all_starts = [
        {
            "k": int(candidate.k),
            "seed": int(candidate.seed),
            "converged": bool(candidate.converged),
            "marginal_loglik": float(candidate.marginal_loglik),
            "n_iterations": len(candidate.likelihood_trace),
        }
        for candidate in fits
    ]
    return best


def marginal_loglik(fit: HMMFit, behavior) -> float:
    import jax.numpy as jnp

    design, _ = hmm_design(
        behavior,
        position_mean=fit.scaler["position_mean"],
        position_scale=fit.scaler["position_scale"],
    )
    value = fit.model.marginal_log_prob(
        fit.params,
        jnp.asarray(behavior.emission.to_numpy(np.int32)),
        jnp.asarray(design, dtype=jnp.float64),
    )
    return float(value)


def state_order(fit: HMMFit, input_dim: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    reference = jnp.zeros(input_dim, dtype=jnp.float64).at[0].set(1.0)
    # Explicit reference levels in the frozen design: no previous observed
    # emission and missing previous-reward value at session start.
    reference = reference.at[7].set(1.0).at[9].set(1.0)
    probabilities = np.asarray(
        jax.vmap(
            lambda state: fit.model.emission_component.distribution(
                fit.params.emissions, state, reference
            ).probs_parameter()
        )(jnp.arange(fit.k))
    )
    return np.lexsort(tuple(probabilities[:, index] for index in reversed(range(3))))


def predictive_state_probs(fit: HMMFit, behavior) -> np.ndarray:
    import jax.numpy as jnp

    design, _ = hmm_design(
        behavior,
        position_mean=fit.scaler["position_mean"],
        position_scale=fit.scaler["position_scale"],
    )
    probabilities = np.asarray(
        fit.model.predict_state_probs(
            fit.params,
            jnp.asarray(behavior.emission.to_numpy(np.int32)),
            jnp.asarray(design, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    probabilities = probabilities[:, state_order(fit, design.shape[1])]
    if (
        probabilities.shape != (len(behavior), fit.k)
        or not np.all(np.isfinite(probabilities))
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("hmm_nonfinite_predictive_posterior")
    return probabilities


def one_se_smallest(
    means: Mapping[int, float], standard_errors: Mapping[int, float]
) -> int:
    best = max(means, key=lambda candidate: means[candidate])
    threshold = means[best] - standard_errors[best]
    eligible = [candidate for candidate in sorted(means) if means[candidate] >= threshold]
    if not eligible:
        raise ValueError("hmm_inner_candidate_failure")
    return int(eligible[0])


@dataclass
class TargetHMM:
    target_session: int
    selected_k: int
    final_fit: HMMFit
    target_probs: np.ndarray
    inner_probs: dict[int, np.ndarray]
    selection_rows: list[dict]
    start_rows: list[dict]


def select_target_hmm(
    sessions: Mapping[int, object],
    target_session: int,
    *,
    fit_cache: dict | None = None,
    seeds: Sequence[int] = HMM_SEEDS,
    max_iter: int = HMM_MAX_ITER,
    tolerance: float = HMM_TOL,
) -> TargetHMM:
    """Nested leave-one-session-out K selection for one target session."""

    fit_cache = fit_cache if fit_cache is not None else {}
    outer_ids = sorted(set(sessions) - {int(target_session)})
    if len(outer_ids) < 3:
        raise ValueError("hmm_insufficient_training_sessions")

    def cached_fit(training_ids: Sequence[int], k: int) -> HMMFit:
        key = (
            tuple(sorted(map(int, training_ids))),
            int(k),
            tuple(map(int, seeds)),
            max_iter,
            float(tolerance),
        )
        if key not in fit_cache:
            fit_cache[key] = fit_hmm(
                {session_id: sessions[session_id] for session_id in key[0]},
                k=k,
                seeds=seeds,
                max_iter=max_iter,
                tolerance=tolerance,
            )
        return fit_cache[key]

    scores: dict[int, list[float]] = {k: [] for k in HMM_K_GRID}
    inner_fits: dict[tuple[int, int], HMMFit] = {}
    selection_rows: list[dict] = []
    for k in HMM_K_GRID:
        for heldout in outer_ids:
            training_ids = [session_id for session_id in outer_ids if session_id != heldout]
            fit = cached_fit(training_ids, k)
            inner_fits[(k, heldout)] = fit
            score = marginal_loglik(fit, sessions[heldout]) / len(sessions[heldout])
            scores[k].append(score)
            selection_rows.append(
                {
                    "target_session": int(target_session),
                    "heldout_session": int(heldout),
                    "k": int(k),
                    "per_trial_loglik": float(score),
                }
            )
    means = {k: float(np.mean(values)) for k, values in scores.items()}
    errors = {
        k: float(np.std(values, ddof=1) / np.sqrt(len(values)))
        for k, values in scores.items()
    }
    selected = one_se_smallest(means, errors)
    for row in selection_rows:
        row.update(
            mean=means[row["k"]],
            standard_error=errors[row["k"]],
            selected=row["k"] == selected,
        )

    final = cached_fit(outer_ids, selected)
    target_probs = predictive_state_probs(final, sessions[target_session])
    inner_probs = {
        heldout: predictive_state_probs(
            inner_fits[(selected, heldout)], sessions[heldout]
        )
        for heldout in outer_ids
    }
    used_fits = {id(final): final}
    used_fits.update({id(value): value for value in inner_fits.values()})
    start_rows = []
    for fit in used_fits.values():
        for start in fit.all_starts:
            start_rows.append(
                {
                    "target_session": int(target_session),
                    "training_sessions": ",".join(
                        map(str, fit.training_session_ids)
                    ),
                    **start,
                    "selected_start": int(start["seed"]) == int(fit.seed),
                }
            )
    return TargetHMM(
        target_session=int(target_session),
        selected_k=selected,
        final_fit=final,
        target_probs=target_probs,
        inner_probs=inner_probs,
        selection_rows=selection_rows,
        start_rows=start_rows,
    )

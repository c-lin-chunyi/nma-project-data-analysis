"""Actual-frame discrete-time lick-hazard models with causal neural history."""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import (
    BASIS_GRID,
    CELL_COUNT,
    CELL_SEEDS,
    N_BLOCKS,
    RIDGE_GRID,
    RISK_END,
    RISK_START,
)


def deterministic_cell_indices(
    n_cells: int, seed: int, experiment_id: int, k: int = CELL_COUNT
) -> np.ndarray | None:
    if n_cells < k:
        return None
    digest = hashlib.sha256(f"{int(experiment_id)}:{int(k)}:{int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(n_cells, int(k), replace=False))


def raised_cosine(lags: np.ndarray, basis_count: int) -> np.ndarray:
    lags = np.asarray(lags, float)
    if basis_count not in BASIS_GRID:
        raise ValueError(f"basis_count must be one of {BASIS_GRID}")
    output = np.zeros((len(lags), basis_count), float)
    inside = (lags >= 0.0) & (lags <= RISK_END)
    if basis_count == 1:
        output[inside, 0] = 1.0
        return output
    width = RISK_END / (basis_count - 1)
    for index in range(basis_count):
        center = RISK_END * index / (basis_count - 1)
        distance = np.abs(lags - center)
        use = inside & (distance <= width)
        output[use, index] = 0.5 * (
            1.0 + np.cos(np.pi * (lags[use] - center) / width)
        )
    return output


@dataclass
class NeuralTrial:
    trial_id: int
    relative_time: np.ndarray
    events: np.ndarray
    dff: np.ndarray


def load_neural_trials(path) -> tuple[np.ndarray, dict[int, NeuralTrial]]:
    import h5py

    with h5py.File(path, "r") as h5:
        cells = np.asarray(h5["cell_specimen_id"][:], np.int64)
        ids = np.asarray(h5["trial_id"][:], np.int64)
        offsets = np.asarray(h5["frame_offsets"][:], np.int64)
        relative = np.asarray(h5["relative_time"][:], np.float64)
        events = np.asarray(h5["events"][:], np.float32)
        dff = np.asarray(h5["dff"][:], np.float32)
    trials = {
        int(trial_id): NeuralTrial(
            trial_id=int(trial_id),
            relative_time=relative[lo:hi],
            events=events[lo:hi],
            dff=dff[lo:hi],
        )
        for trial_id, lo, hi in zip(ids, offsets[:-1], offsets[1:])
    }
    return cells, trials


def causal_history(
    trial: NeuralTrial,
    left_edges: np.ndarray,
    selected_cells: np.ndarray,
    basis_count: int,
    *,
    signal: str,
    return_basis_energy: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return bins x (cell*basis) using frames strictly before each left edge."""

    values = np.asarray(getattr(trial, signal), np.float64)[:, selected_cells]
    times = np.asarray(trial.relative_time, np.float64)
    baseline = (times >= -1.25) & (times < 0.0)
    if not baseline.any():
        raise ValueError("hazard_incomplete_prediction")
    centered = values - np.mean(values[baseline], axis=0, keepdims=True)
    features = np.zeros((len(left_edges), len(selected_cells), basis_count), float)
    basis_energy = np.zeros((len(left_edges), basis_count), float)
    for row, left in enumerate(np.asarray(left_edges, float)):
        eligible = times < left
        lags = left - times[eligible]
        support = (lags >= 0.0) & (lags <= RISK_END)
        if not support.any():
            continue
        basis = raised_cosine(lags[support], basis_count)
        basis_energy[row] = np.sum(basis * basis, axis=0)
        features[row] = centered[eligible][support].T @ basis
    flattened = features.reshape(len(left_edges), -1)
    return (flattened, basis_energy) if return_basis_energy else flattened


def risk_bins(relative_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    internal = np.asarray(relative_frames, float)
    internal = internal[
        np.isfinite(internal) & (internal > RISK_START) & (internal < RISK_END)
    ]
    boundaries = np.r_[RISK_START, np.unique(internal), RISK_END]
    if len(boundaries) < 2 or np.any(np.diff(boundaries) <= 0):
        raise ValueError("invalid_timestamp_grid")
    return boundaries[:-1], boundaries[1:]


def event_bin(first_lick: float, left: np.ndarray, right: np.ndarray) -> int | None:
    if not np.isfinite(first_lick) or first_lick < RISK_START or first_lick > RISK_END:
        return None
    candidates = np.flatnonzero(
        (first_lick >= left)
        & ((first_lick < right) | ((right == RISK_END) & (first_lick <= right)))
    )
    if len(candidates) != 1:
        raise ValueError("hazard_incomplete_prediction")
    return int(candidates[0])


M0_NUMERIC = (
    "flashes_before_change",
    "time_since_previous_change",
    "time_since_previous_lick",
    "time_since_previous_reward",
    "session_position",
    "preceding_omission",
    "pre_change_pupil",
    "pre_change_running",
)


def build_risk_rows(
    behavior: pd.DataFrame,
    state_probs: np.ndarray,
    neural_trials: Mapping[int, NeuralTrial],
    *,
    experiment_id: int,
    seed: int,
    basis_count: int,
    signal: str,
    include_neural: bool = True,
) -> pd.DataFrame:
    if include_neural:
        selected = deterministic_cell_indices(
            next(iter(neural_trials.values())).events.shape[1]
            if neural_trials
            else 0,
            seed,
            experiment_id,
        )
        if selected is None:
            raise ValueError("neural_fewer_than_50_cells")
    else:
        selected = np.empty(0, int)
    rows: list[dict] = []
    for trial_position, trial in behavior.iterrows():
        if not bool(trial.primary_risk_eligible):
            continue
        neural = neural_trials.get(int(trial.trial_id))
        if neural is None:
            raise ValueError("hazard_incomplete_prediction")
        left, right = risk_bins(neural.relative_time)
        event = event_bin(float(trial.first_post_change_lick), left, right)
        if include_neural:
            history, basis_energy = causal_history(
                neural,
                left,
                selected,
                basis_count,
                signal=signal,
                return_basis_energy=True,
            )
        else:
            history = np.empty((len(left), 0), float)
            basis_energy = np.zeros((len(left), basis_count), float)
        for bin_index, (lo, hi) in enumerate(zip(left, right)):
            if event is not None and bin_index > event:
                break
            row = {
                "trial_id": int(trial.trial_id),
                "raw_trial_index": int(trial.raw_trial_index),
                "bin_index": int(bin_index),
                "left": float(lo),
                "right": float(hi),
                "offset": float(np.log(hi - lo)),
                "event": int(event == bin_index),
                "image_transition": str(trial.image_transition),
                "previous_outcome": str(trial.previous_outcome),
                "neural": history[bin_index],
                "basis_energy": basis_energy[bin_index],
                "state": np.asarray(state_probs[trial_position, :-1], float),
            }
            for name in M0_NUMERIC:
                value = trial[name]
                if name.startswith("time_since_"):
                    value = np.log1p(value) if np.isfinite(value) and value >= 0 else np.nan
                row[name] = float(value) if pd.notna(value) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class DesignTransform:
    baseline_levels: tuple[int, ...]
    transition_levels: tuple[str, ...]
    outcome_levels: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    numeric_names: tuple[str, ...]
    neural_dimension: int
    neural_normalizers: np.ndarray
    state_dimension: int
    model: str


def _raw_numeric(rows: pd.DataFrame, model: str) -> tuple[np.ndarray, tuple[str, ...]]:
    state_dim = len(rows.iloc[0].state) if len(rows) else 0
    neural_dim = len(rows.iloc[0].neural) if len(rows) else 0
    names = list(M0_NUMERIC) + [f"state_{index}" for index in range(state_dim)]
    arrays = [rows[name].to_numpy(float) for name in M0_NUMERIC]
    if state_dim:
        arrays.extend(np.stack(rows.state)[:, index] for index in range(state_dim))
    if model in {"M1", "M2"}:
        arrays.extend(np.stack(rows.neural)[:, index] for index in range(neural_dim))
        names.extend(f"neural_{index}" for index in range(neural_dim))
    if model == "M2":
        state = np.stack(rows.state) if state_dim else np.zeros((len(rows), 0))
        neural = np.stack(rows.neural)
        for s in range(state_dim):
            arrays.extend(state[:, s] * neural[:, index] for index in range(neural_dim))
            names.extend(f"state_{s}:neural_{index}" for index in range(neural_dim))
    matrix = np.column_stack(arrays) if arrays else np.empty((len(rows), 0))
    return matrix.astype(float), tuple(names)


def _normalize_neural_columns(
    matrix: np.ndarray, names: Sequence[str], normalizers: np.ndarray
) -> np.ndarray:
    if len(normalizers) == 0:
        return matrix
    output = matrix.copy()
    for column, name in enumerate(names):
        neural_name = name.split(":")[-1]
        if neural_name.startswith("neural_"):
            index = int(neural_name.removeprefix("neural_"))
            output[:, column] /= normalizers[index]
    return output


def fit_transform(rows: pd.DataFrame, model: str) -> tuple[np.ndarray, DesignTransform, int]:
    if rows.empty:
        raise ValueError("hazard_no_training_event")
    baseline = tuple(sorted(rows.bin_index.astype(int).unique()))
    transitions = tuple(sorted(set(rows.image_transition.astype(str))) + ["__unseen__"])
    outcomes = tuple(sorted(set(rows.previous_outcome.astype(str))) + ["__unseen__"])
    numeric, names = _raw_numeric(rows, model)
    neural_dimension = len(rows.iloc[0].neural)
    if model in {"M1", "M2"}:
        energies = np.stack(rows.basis_energy)
        basis_normalizers = np.sqrt(np.sum(energies, axis=0))
        basis_normalizers[
            ~np.isfinite(basis_normalizers) | (basis_normalizers == 0)
        ] = 1.0
        if neural_dimension % len(basis_normalizers):
            raise ValueError("hazard_incomplete_prediction")
        neural_normalizers = np.tile(
            basis_normalizers, neural_dimension // len(basis_normalizers)
        )
        numeric = _normalize_neural_columns(numeric, names, neural_normalizers)
    else:
        neural_normalizers = np.empty(0, float)
    finite = np.isfinite(numeric)
    medians = np.array(
        [
            float(np.median(numeric[finite[:, index], index]))
            if finite[:, index].any()
            else 0.0
            for index in range(numeric.shape[1])
        ]
    )
    imputed = np.where(finite, numeric, medians)
    means = np.mean(imputed, axis=0)
    scales = np.std(imputed, axis=0, ddof=1) if len(rows) > 1 else np.ones(imputed.shape[1])
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    transform = DesignTransform(
        baseline_levels=baseline,
        transition_levels=transitions,
        outcome_levels=outcomes,
        medians=medians,
        means=means,
        scales=scales,
        numeric_names=names,
        neural_dimension=neural_dimension,
        neural_normalizers=neural_normalizers,
        state_dimension=len(rows.iloc[0].state),
        model=model,
    )
    matrix, baseline_count = apply_transform(rows, transform)
    return matrix, transform, baseline_count


def _one_hot(values: Iterable, levels: Sequence) -> np.ndarray:
    values = list(values)
    return np.column_stack(
        [np.asarray([value == level for value in values], float) for level in levels]
    )


def apply_transform(
    rows: pd.DataFrame, transform: DesignTransform
) -> tuple[np.ndarray, int]:
    if rows.empty:
        return np.empty((0, 0)), len(transform.baseline_levels)
    observed_bins = set(rows.bin_index.astype(int))
    if not observed_bins.issubset(set(transform.baseline_levels)):
        raise ValueError("hazard_incomplete_prediction")
    baseline = _one_hot(rows.bin_index.astype(int), transform.baseline_levels)
    transitions = [
        value if value in transform.transition_levels[:-1] else "__unseen__"
        for value in rows.image_transition.astype(str)
    ]
    outcomes = [
        value if value in transform.outcome_levels[:-1] else "__unseen__"
        for value in rows.previous_outcome.astype(str)
    ]
    categories = np.column_stack(
        (
            _one_hot(transitions, transform.transition_levels),
            _one_hot(outcomes, transform.outcome_levels),
        )
    )
    numeric, names = _raw_numeric(rows, transform.model)
    if names != transform.numeric_names:
        raise ValueError("hazard_incomplete_prediction")
    numeric = _normalize_neural_columns(
        numeric, names, transform.neural_normalizers
    )
    finite = np.isfinite(numeric)
    standardized = (
        np.where(finite, numeric, transform.medians) - transform.means
    ) / transform.scales
    missing = (~finite).astype(float)
    return np.column_stack((baseline, categories, standardized, missing)), len(
        transform.baseline_levels
    )


@dataclass
class HazardFit:
    beta: np.ndarray
    transform: DesignTransform
    baseline_count: int
    penalty: float
    model: str
    success: bool
    objective: float
    gradient_norm: float
    optimizer_status: int
    optimizer_message: str
    min_eta: float
    max_eta: float
    tail_low_count: int
    tail_high_count: int
    protection_count: int
    runtime_seconds: float


class HazardFitFailure(ValueError):
    def __init__(self, reason: str, diagnostics: Mapping[str, object]):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = dict(diagnostics)


class HazardEvaluationFailure(ValueError):
    def __init__(
        self,
        reason: str,
        block_rows: Sequence[Mapping],
        coefficient_rows: Sequence[Mapping],
    ):
        super().__init__(reason)
        self.reason = reason
        self.block_rows = list(block_rows)
        self.coefficient_rows = list(coefficient_rows)


_EVENT_LOW = -36.0
_EVENT_HIGH = 36.0
# exp(650) is finite with ample room for summing realistic risk sets.  Above
# this point the survival term uses a finite C1 saturating continuation with
# the exact derivative at the join.  A final fit/prediction that still needs
# this branch for a non-event row is rejected rather than reported as an
# approximation.
_SURVIVAL_GUARD = 650.0
_SURVIVAL_EXTENSION_SCALE = 1e20
_RIDGE_GUARD = 1e140


def _cloglog_value_derivative(eta, event):
    """Stable row log likelihood and analytic derivative with respect to eta."""

    import jax.numpy as jnp

    clipped = jnp.clip(eta, _EVENT_LOW, _EVENT_HIGH)
    cumulative = jnp.exp(clipped)
    event_value_middle = jnp.log(-jnp.expm1(-cumulative))
    gradient_argument = jnp.minimum(cumulative, 50.0)
    event_gradient_middle = jnp.where(
        cumulative > 50.0,
        0.0,
        cumulative / jnp.expm1(gradient_argument),
    )
    event_value = jnp.where(
        eta < _EVENT_LOW,
        eta,
        jnp.where(eta > _EVENT_HIGH, 0.0, event_value_middle),
    )
    event_gradient = jnp.where(
        eta < _EVENT_LOW,
        1.0,
        jnp.where(eta > _EVENT_HIGH, 0.0, event_gradient_middle),
    )

    guarded = jnp.minimum(eta, _SURVIVAL_GUARD)
    cumulative_guarded = jnp.exp(guarded)
    excess = jnp.maximum(eta - _SURVIVAL_GUARD, 0.0)
    extension = jnp.tanh(excess / _SURVIVAL_EXTENSION_SCALE)
    survival_value = -cumulative_guarded * (
        1.0 + _SURVIVAL_EXTENSION_SCALE * extension
    )
    survival_gradient = -cumulative_guarded * (1.0 - extension**2)
    is_event = event == 1
    return (
        jnp.where(is_event, event_value, survival_value),
        jnp.where(is_event, event_gradient, survival_gradient),
    )


@lru_cache(maxsize=1)
def _hazard_value_and_grad():
    """One reusable JIT kernel; array shapes, not candidates, drive compilation."""

    import jax
    import jax.numpy as jnp

    def objective_and_gradient(beta, design, event, offset, penalty, ridge_mask):
        eta = offset + design @ beta
        loglik_rows, eta_gradient = _cloglog_value_derivative(eta, event)
        loglik = jnp.sum(loglik_rows)
        absolute = jnp.abs(beta)
        excess = jnp.maximum(absolute - _RIDGE_GUARD, 0.0)
        extension = jnp.tanh(excess / _RIDGE_GUARD)
        guarded_square = jnp.minimum(absolute, _RIDGE_GUARD) ** 2
        square = jnp.where(
            absolute <= _RIDGE_GUARD,
            guarded_square,
            _RIDGE_GUARD**2
            + 2.0 * _RIDGE_GUARD**2 * extension,
        )
        square_gradient = jnp.where(
            absolute <= _RIDGE_GUARD,
            2.0 * beta,
            2.0
            * _RIDGE_GUARD
            * (1.0 - extension**2)
            * jnp.sign(beta),
        )
        ridge = penalty * jnp.sum(square * ridge_mask)
        value = -loglik + ridge
        gradient = -(design.T @ eta_gradient) + (
            penalty * square_gradient * ridge_mask
        )
        return value, gradient

    return jax.jit(objective_and_gradient)


@dataclass
class PreparedHazard:
    rows: pd.DataFrame
    design: np.ndarray
    event: np.ndarray
    offset: np.ndarray
    transform: DesignTransform
    baseline_count: int
    model: str


@dataclass
class PreparedFold:
    test_block: int
    train_max_raw_index: int
    test_min_raw_index: int
    train: PreparedHazard
    test_rows: pd.DataFrame
    test_design: np.ndarray


def prepare_hazard(rows: pd.DataFrame, *, model: str) -> PreparedHazard:
    if rows.empty or int(rows.event.sum()) == 0:
        raise ValueError("hazard_no_training_event")
    design, transform, baseline_count = fit_transform(rows, model)
    return PreparedHazard(
        rows=rows,
        design=np.asarray(design, np.float64),
        event=rows.event.to_numpy(np.float64),
        offset=rows.offset.to_numpy(np.float64),
        transform=transform,
        baseline_count=baseline_count,
        model=model,
    )


def ridge_mask(n_coefficients: int) -> np.ndarray:
    """r3 penalty mask: every fitted coefficient, including baseline, is one."""

    return np.ones(int(n_coefficients), np.float64)


def fit_prepared(prepared: PreparedHazard, *, penalty: float) -> HazardFit:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import minimize

    started = time.monotonic()
    design = prepared.design
    xj, yj, oj = map(
        jnp.asarray, (design, prepared.event, prepared.offset)
    )

    # r3: the ordinal-bin baseline receives the same candidate ridge penalty
    # as every other fitted coefficient.  The offset is not a coefficient.
    penalty_mask = ridge_mask(design.shape[1])
    rj = jnp.asarray(penalty_mask)
    value_grad = _hazard_value_and_grad()

    def wrapped(beta):
        value, gradient = value_grad(
            jnp.asarray(beta), xj, yj, oj, float(penalty), rj
        )
        return float(value), np.asarray(gradient, float)

    result = minimize(
        wrapped,
        np.zeros(design.shape[1], float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-7},
    )
    success = bool(
        result.success
        and np.isfinite(result.fun)
        and np.all(np.isfinite(result.x))
        and np.all(np.isfinite(result.jac))
    )
    beta = np.asarray(result.x, float)
    eta = prepared.offset + design @ beta
    protection_count = int(
        np.count_nonzero((prepared.event == 0) & (eta > _SURVIVAL_GUARD))
    )
    finite_eta = eta[np.isfinite(eta)]
    diagnostics = {
        "objective": float(result.fun) if np.isfinite(result.fun) else np.nan,
        "gradient_norm": (
            float(np.linalg.norm(np.asarray(result.jac, float)))
            if np.all(np.isfinite(result.jac))
            else np.nan
        ),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "min_eta": float(np.min(finite_eta)) if len(finite_eta) else np.nan,
        "max_eta": float(np.max(finite_eta)) if len(finite_eta) else np.nan,
        "tail_low_count": int(np.count_nonzero(eta < _EVENT_LOW)),
        "tail_high_count": int(np.count_nonzero(eta > _EVENT_HIGH)),
        "protection_count": protection_count,
        "runtime_seconds": float(time.monotonic() - started),
    }
    if not success:
        raise HazardFitFailure("hazard_nonconvergence", diagnostics)
    if protection_count:
        raise HazardFitFailure(
            "hazard_nonrepresentable_prediction", diagnostics
        )
    return HazardFit(
        beta=beta,
        transform=prepared.transform,
        baseline_count=prepared.baseline_count,
        penalty=float(penalty),
        model=prepared.model,
        success=True,
        **diagnostics,
    )


def fit_hazard(rows: pd.DataFrame, *, model: str, penalty: float) -> HazardFit:
    return fit_prepared(prepare_hazard(rows, model=model), penalty=penalty)


def _numpy_cloglog(eta: np.ndarray, event: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, np.float64)
    event = np.asarray(event, np.int8)
    result = np.empty_like(eta)
    event_rows = event == 1
    low = event_rows & (eta < _EVENT_LOW)
    high = event_rows & (eta > _EVENT_HIGH)
    middle = event_rows & ~low & ~high
    result[low] = eta[low]
    result[high] = 0.0
    z = np.exp(eta[middle])
    result[middle] = np.log(-np.expm1(-z))
    survival = ~event_rows
    survival_eta = eta[survival]
    survival_value = np.zeros_like(survival_eta)
    representable = survival_eta >= np.log(np.finfo(np.float64).tiny)
    guarded = np.minimum(
        survival_eta[representable], _SURVIVAL_GUARD
    )
    excess = np.maximum(
        survival_eta[representable] - _SURVIVAL_GUARD, 0.0
    )
    survival_value[representable] = -np.exp(guarded) * (
        1.0
        + _SURVIVAL_EXTENSION_SCALE
        * np.tanh(excess / _SURVIVAL_EXTENSION_SCALE)
    )
    result[survival] = survival_value
    return result


def score_design(
    fit: HazardFit, rows: pd.DataFrame, design: np.ndarray
) -> tuple[float, int, pd.DataFrame]:
    eta = rows.offset.to_numpy(float) + design @ fit.beta
    event = rows.event.to_numpy(int)
    if np.any((event == 0) & (eta > _SURVIVAL_GUARD)):
        raise ValueError("hazard_nonrepresentable_prediction")
    contributions = _numpy_cloglog(eta, event)
    if not np.all(np.isfinite(contributions)):
        raise ValueError("hazard_nonrepresentable_prediction")
    diagnostic = rows[["trial_id", "raw_trial_index", "bin_index", "event"]].copy()
    diagnostic["loglik"] = contributions
    per_trial = diagnostic.groupby("trial_id", sort=False).loglik.sum()
    return float(per_trial.sum()), int(len(per_trial)), diagnostic


def score_hazard(fit: HazardFit, rows: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    design, _ = apply_transform(rows, fit.transform)
    return score_design(fit, rows, design)


def raw_blocks(n_trials: int) -> list[np.ndarray]:
    return [np.asarray(block, int) for block in np.array_split(np.arange(n_trials), N_BLOCKS)]


def prepare_prequential(
    rows: pd.DataFrame,
    *,
    n_raw_trials: int,
    model: str,
) -> list[PreparedFold]:
    blocks = raw_blocks(n_raw_trials)
    prepared: list[PreparedFold] = []
    for block_index in range(1, N_BLOCKS):
        train_limit = int(blocks[block_index - 1][-1])
        test_indices = set(map(int, blocks[block_index]))
        train = rows[rows.raw_trial_index <= train_limit].copy()
        test = rows[rows.raw_trial_index.isin(test_indices)].copy()
        if test.empty:
            raise ValueError("hazard_empty_test_block")
        fitted = prepare_hazard(train, model=model)
        test_design, _ = apply_transform(test, fitted.transform)
        prepared.append(
            PreparedFold(
                test_block=block_index + 1,
                train_max_raw_index=train_limit,
                test_min_raw_index=min(test_indices),
                train=fitted,
                test_rows=test,
                test_design=np.asarray(test_design, np.float64),
            )
        )
    return prepared


def evaluate_prepared(
    folds: Sequence[PreparedFold], *, penalty: float
) -> tuple[float, int, list[dict], list[dict]]:
    total, trials = 0.0, 0
    block_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    failure_reasons: list[str] = []
    for fold in folds:
        base = {
            "test_block": fold.test_block,
            "train_max_raw_index": fold.train_max_raw_index,
            "test_min_raw_index": fold.test_min_raw_index,
            "n_train_rows": len(fold.train.rows),
            "n_test_rows": len(fold.test_rows),
        }
        try:
            fit = fit_prepared(fold.train, penalty=penalty)
            score, count, _ = score_design(
                fit, fold.test_rows, fold.test_design
            )
        except Exception as exc:
            if isinstance(exc, MemoryError) or type(exc).__name__ in {
                "XlaRuntimeError",
                "JaxRuntimeError",
            }:
                raise
            reason = str(exc)
            failure_reasons.append(reason)
            diagnostic = (
                exc.diagnostics if isinstance(exc, HazardFitFailure) else {}
            )
            block_rows.append(
                {
                    **base,
                    "n_test_trials": 0,
                    "loglik": np.nan,
                    "status": "nonestimable",
                    "reason": reason,
                    "detail": str(exc),
                    "exception_type": type(exc).__name__,
                    **diagnostic,
                }
            )
            continue
        total += score
        trials += count
        block_rows.append(
            {
                **base,
                "n_test_trials": count,
                "loglik": score,
                "status": "estimable",
                "reason": None,
                "detail": None,
                "exception_type": None,
                "objective": fit.objective,
                "gradient_norm": fit.gradient_norm,
                "optimizer_status": fit.optimizer_status,
                "optimizer_message": fit.optimizer_message,
                "min_eta": fit.min_eta,
                "max_eta": fit.max_eta,
                "tail_low_count": fit.tail_low_count,
                "tail_high_count": fit.tail_high_count,
                "protection_count": fit.protection_count,
                "runtime_seconds": fit.runtime_seconds,
            }
        )
        coefficient_rows.extend(
            {
                "test_block": fold.test_block,
                "coefficient_index": index,
                "coefficient": float(value),
                "model": fold.train.model,
            }
            for index, value in enumerate(fit.beta)
        )
    if failure_reasons:
        raise HazardEvaluationFailure(
            failure_reasons[0], block_rows, coefficient_rows
        )
    if trials == 0:
        raise ValueError("hazard_empty_test_block")
    return total / trials, trials, block_rows, coefficient_rows


def evaluate_prequential(
    rows: pd.DataFrame,
    *,
    n_raw_trials: int,
    model: str,
    penalty: float,
) -> tuple[float, int, list[dict], list[dict]]:
    return evaluate_prepared(
        prepare_prequential(rows, n_raw_trials=n_raw_trials, model=model),
        penalty=penalty,
    )


def one_se_hazard(
    candidate_rows: pd.DataFrame,
    *,
    model: str,
    eligible_sessions: Sequence[int] | None = None,
) -> tuple[int | None, float]:
    if candidate_rows.empty:
        raise ValueError("hazard_no_complete_candidate")
    expected_sessions = set(
        map(
            int,
            eligible_sessions
            if eligible_sessions is not None
            else candidate_rows.tuning_session.unique(),
        )
    )
    valid = candidate_rows[candidate_rows.status.eq("estimable")].copy()
    grouping = ["tuning_session", "basis_count", "penalty"]
    complete_keys: list[tuple[float, float]] = []
    for (basis, penalty), group in candidate_rows.groupby(
        ["basis_count", "penalty"], dropna=False
    ):
        sessions = set(map(int, group.tuning_session.unique()))
        if sessions != expected_sessions or not group.status.eq("estimable").all():
            continue
        if model != "M0":
            counts = group.groupby("tuning_session").cell_seed.nunique()
            if (
                set(map(int, counts.index)) != expected_sessions
                or not counts.eq(len(CELL_SEEDS)).all()
            ):
                continue
        complete_keys.append((float(basis), float(penalty)))
    if not complete_keys:
        raise ValueError("hazard_no_complete_candidate")
    complete = pd.DataFrame(complete_keys, columns=["basis_count", "penalty"])
    valid = valid.merge(complete, on=["basis_count", "penalty"], how="inner")
    if model != "M0":
        # The registered aggregation averages cell seeds within each tuning
        # session before using sessions as the independent units for the SE.
        session_scores = (
            valid.groupby(grouping, dropna=False, as_index=False)
            .per_trial_loglik.mean()
        )
    else:
        session_scores = valid[grouping + ["per_trial_loglik"]].copy()
    summary = (
        session_scores.groupby(["basis_count", "penalty"], dropna=False)
        .per_trial_loglik.agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    best = summary.loc[summary["mean"].idxmax()]
    eligible = summary[summary["mean"] >= best["mean"] - best["se"]].copy()
    if model == "M0":
        chosen = eligible.sort_values("penalty", ascending=False).iloc[0]
        return None, float(chosen.penalty)
    basis = int(eligible.basis_count.min())
    chosen = (
        eligible[eligible.basis_count.eq(basis)]
        .sort_values("penalty", ascending=False)
        .iloc[0]
    )
    return basis, float(chosen.penalty)

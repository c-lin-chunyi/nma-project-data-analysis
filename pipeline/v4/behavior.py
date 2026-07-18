"""Lossless behavior compilation for the predictive-state and hazard models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    EMISSION_ABORT,
    EMISSION_MISSING,
    EMISSION_TASK_RESPONSE,
    EMISSION_WITHHOLD,
    RISK_END,
    RISK_START,
    WINDOW_START,
)

HMM_FEATURE_NAMES = (
    "intercept",
    "scheduled_change",
    "novelty",
    "novelty_missing",
    "previous_abort",
    "previous_task_response",
    "previous_withhold",
    "no_previous_observed_emission",
    "previous_reward",
    "previous_reward_missing",
    "session_position_standardized",
)


def _bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame), dtype=bool)
    return frame[name].fillna(False).astype(bool).to_numpy()


def _time_column(frame: pd.DataFrame) -> str | None:
    return next((name for name in ("timestamps", "timestamp", "time") if name in frame), None)


def _value_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in frame), None)


def _finite_times(frame: pd.DataFrame) -> np.ndarray:
    column = _time_column(frame)
    if column is None:
        return np.empty(0, float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    return np.sort(values[np.isfinite(values)])


def _trial_ids(trials: pd.DataFrame) -> np.ndarray:
    name = "trial_id_v4" if "trial_id_v4" in trials else (
        "trials_id" if "trials_id" in trials else None
    )
    values = trials[name].to_numpy() if name else trials.index.to_numpy()
    return np.asarray(values, np.int64)


def _novelty(trials: pd.DataFrame, stim: pd.DataFrame) -> np.ndarray:
    if "is_image_novel" in trials:
        return trials["is_image_novel"].astype("boolean").to_numpy(dtype=float, na_value=np.nan)
    if "is_image_novel" not in stim:
        return np.full(len(trials), np.nan)
    valid = stim["is_image_novel"].notna()
    if "trials_id" in stim and valid.any():
        by_trial = (
            stim.loc[valid]
            .groupby("trials_id")["is_image_novel"]
            .agg(lambda values: bool(pd.Series(values).astype(bool).max()))
        )
        return pd.Series(_trial_ids(trials)).map(by_trial).to_numpy(float)
    levels = stim.loc[valid, "is_image_novel"].astype(bool).unique()
    return np.full(len(trials), float(levels[0]) if len(levels) == 1 else np.nan)


def _median_trace(
    table: pd.DataFrame,
    center: float,
    candidates: tuple[str, ...],
) -> float:
    time_name = _time_column(table)
    value_name = _value_column(table, candidates)
    if time_name is None or value_name is None or not np.isfinite(center):
        return np.nan
    times = pd.to_numeric(table[time_name], errors="coerce").to_numpy(float)
    values = pd.to_numeric(table[value_name], errors="coerce").to_numpy(float)
    use = (
        np.isfinite(times)
        & np.isfinite(values)
        & (times >= center + WINDOW_START)
        & (times < center)
    )
    return float(np.median(values[use])) if use.any() else np.nan


def _stim_history(
    stim: pd.DataFrame, trial_id: int, start: float, change: float
) -> tuple[float, object]:
    if "trials_id" not in stim:
        return np.nan, np.nan
    subset = stim[stim["trials_id"].eq(trial_id)].copy()
    time_name = _value_column(subset, ("start_time", "timestamps", "time"))
    if time_name is not None:
        times = pd.to_numeric(subset[time_name], errors="coerce")
        subset = subset[(times >= start) & (times < change)]
        subset = subset.assign(_time=times.loc[subset.index]).sort_values("_time")
    if subset.empty:
        return 0.0, False
    omitted = (
        subset["omitted"].fillna(False).astype(bool)
        if "omitted" in subset
        else pd.Series(False, index=subset.index)
    )
    return float(len(subset)), bool(omitted.iloc[-1])


def _prior_interval(events: np.ndarray, center: float) -> float:
    index = int(np.searchsorted(events, center, side="left") - 1)
    return float(center - events[index]) if index >= 0 else np.nan


def compile_behavior(
    trials: pd.DataFrame,
    stim: pd.DataFrame,
    licks: pd.DataFrame,
    rewards: pd.DataFrame,
    eye: pd.DataFrame,
    running: pd.DataFrame,
    *,
    neural_valid: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compile all trial-level HMM, risk-set, and M0 variables."""

    trials = trials.reset_index(drop=True).copy()
    n = len(trials)
    ids = _trial_ids(trials)
    starts = (
        pd.to_numeric(trials["start_time"], errors="coerce").to_numpy(float)
        if "start_time" in trials
        else np.full(n, np.nan)
    )
    stops = (
        pd.to_numeric(trials["stop_time"], errors="coerce").to_numpy(float)
        if "stop_time" in trials
        else np.full(n, np.nan)
    )
    changes = (
        pd.to_numeric(trials["change_time"], errors="coerce").to_numpy(float)
        if "change_time" in trials
        else np.full(n, np.nan)
    )
    lick_times = _finite_times(licks)
    reward_times = _finite_times(rewards)

    go_flags = np.column_stack((_bool_column(trials, "hit"), _bool_column(trials, "miss")))
    catch_flags = np.column_stack(
        (_bool_column(trials, "false_alarm"), _bool_column(trials, "correct_reject"))
    )
    go = go_flags.any(axis=1) & ~catch_flags.any(axis=1)
    catch = catch_flags.any(axis=1) & ~go_flags.any(axis=1)
    outcome_valid = go ^ catch

    if "auto_rewarded" in trials:
        auto_rewarded = trials["auto_rewarded"].astype("boolean")
        auto_known = auto_rewarded.notna().to_numpy()
        auto = auto_rewarded.fillna(False).astype(bool).to_numpy()
    else:
        auto_known = np.zeros(n, dtype=bool)
        auto = np.zeros(n, dtype=bool)

    novelty = _novelty(trials, stim)
    emissions = np.full(n, EMISSION_MISSING, dtype=np.int8)
    abort = np.zeros(n, dtype=bool)
    response = np.zeros(n, dtype=bool)
    withhold = np.zeros(n, dtype=bool)
    first_post = np.full(n, np.nan)
    prewindow = np.zeros(n, dtype=bool)
    flashes = np.full(n, np.nan)
    preceding_omission = np.full(n, np.nan, dtype=object)
    pupil = np.full(n, np.nan)
    speed = np.full(n, np.nan)

    for index, (trial_id, start, stop, change) in enumerate(
        zip(ids, starts, stops, changes)
    ):
        fields_ok = (
            np.isfinite(start)
            and np.isfinite(stop)
            and np.isfinite(change)
            and start <= change <= stop
            and outcome_valid[index]
            and auto_known[index]
        )
        within_trial = lick_times[
            (lick_times >= start) & (lick_times <= stop)
        ] if np.isfinite(start) and np.isfinite(stop) else np.empty(0)
        abort[index] = bool(np.any(within_trial < change)) if np.isfinite(change) else False
        post = within_trial[within_trial >= change] - change if np.isfinite(change) else np.empty(0)
        if len(post):
            first_post[index] = float(post[0])
        prewindow[index] = bool(np.any((post >= 0.0) & (post < RISK_START)))
        response[index] = (
            not abort[index]
            and bool(np.any((post >= RISK_START) & (post <= RISK_END)))
        )
        withhold[index] = not abort[index] and not response[index]
        if fields_ok and not auto[index]:
            emissions[index] = (
                EMISSION_ABORT
                if abort[index]
                else EMISSION_TASK_RESPONSE
                if response[index]
                else EMISSION_WITHHOLD
            )
        flashes[index], preceding_omission[index] = _stim_history(
            stim, int(trial_id), start, change
        )
        pupil[index] = _median_trace(
            eye, change, ("pupil_area", "pupil_area_raw", "pupil_width")
        )
        speed[index] = _median_trace(
            running, change, ("speed", "running_speed", "velocity")
        )

    rewards_by_trial = np.zeros(n, dtype=bool)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        if np.isfinite(start) and np.isfinite(stop):
            rewards_by_trial[index] = bool(
                np.any((reward_times >= start) & (reward_times <= stop))
            )

    previous_emission = np.full(n, -1, dtype=np.int8)
    last_observed = -1
    for index, emission in enumerate(emissions):
        previous_emission[index] = last_observed
        if emission != EMISSION_MISSING:
            last_observed = int(emission)
    previous_reward = np.r_[np.nan, rewards_by_trial[:-1].astype(float)]
    previous_outcome = np.r_[
        np.array(["session_start"], object),
        np.where(
            emissions[:-1] == EMISSION_ABORT,
            "abort",
            np.where(
                emissions[:-1] == EMISSION_TASK_RESPONSE,
                "task_response",
                np.where(emissions[:-1] == EMISSION_WITHHOLD, "withhold", "missing"),
            ),
        ),
    ]
    finite_changes = np.sort(changes[np.isfinite(changes)])

    image_before = (
        trials["initial_image_name"].fillna("missing").astype(str)
        if "initial_image_name" in trials
        else pd.Series("missing", index=trials.index)
    )
    image_after = (
        trials["change_image_name"].fillna("missing").astype(str)
        if "change_image_name" in trials
        else pd.Series("missing", index=trials.index)
    )
    valid_neural = (
        np.asarray(neural_valid, dtype=bool)
        if neural_valid is not None
        else np.zeros(n, dtype=bool)
    )
    output = pd.DataFrame(
        {
            "trial_id": ids,
            "raw_trial_index": np.arange(n, dtype=np.int64),
            "start_time": starts,
            "stop_time": stops,
            "change_time": changes,
            "go": go,
            "catch": catch,
            "outcome_valid": outcome_valid,
            "auto_reward_known": auto_known,
            "auto_rewarded": auto,
            "abort": abort,
            "task_response": response,
            "withhold": withhold,
            "emission": emissions,
            "first_post_change_lick": first_post,
            "prewindow_competing_event": prewindow,
            "neural_valid": valid_neural,
            "novelty": novelty,
            "previous_emission": previous_emission,
            "previous_reward": previous_reward,
            "rewarded": rewards_by_trial,
            "session_position": np.arange(n, dtype=float) / max(n - 1, 1),
            "flashes_before_change": flashes,
            "time_since_previous_change": [
                _prior_interval(finite_changes, change)
                if np.isfinite(change)
                else np.nan
                for change in changes
            ],
            "time_since_previous_lick": [
                _prior_interval(lick_times, change) if np.isfinite(change) else np.nan
                for change in changes
            ],
            "time_since_previous_reward": [
                _prior_interval(reward_times, change) if np.isfinite(change) else np.nan
                for change in changes
            ],
            "image_transition": image_before + "->" + image_after,
            "preceding_omission": preceding_omission,
            "previous_outcome": previous_outcome,
            "pre_change_pupil": pupil,
            "pre_change_running": speed,
        }
    )
    output["primary_risk_eligible"] = (
        output.go
        & ~output.abort
        & ~output.auto_rewarded
        & output.auto_reward_known
        & output.neural_valid
        & ~output.prewindow_competing_event
    )
    return output


def hmm_design(
    behavior: pd.DataFrame,
    *,
    position_mean: float | None = None,
    position_scale: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Return the frozen numeric HMM design and training standardizer."""

    position = behavior.session_position.to_numpy(float)
    if position_mean is None:
        position_mean = float(np.mean(position))
    if position_scale is None:
        position_scale = float(np.std(position, ddof=1)) if len(position) > 1 else 1.0
    if not np.isfinite(position_scale) or position_scale == 0:
        position_scale = 1.0
    previous = behavior.previous_emission.to_numpy(int)
    novelty = behavior.novelty.to_numpy(float)
    previous_reward = behavior.previous_reward.to_numpy(float)
    columns = [
        np.ones(len(behavior)),
        behavior.go.to_numpy(float),
        np.nan_to_num(novelty, nan=0.0),
        np.isnan(novelty).astype(float),
        *(previous == value for value in (0, 1, 2)),
        (previous < 0).astype(float),
        np.nan_to_num(previous_reward, nan=0.0),
        np.isnan(previous_reward).astype(float),
        (position - position_mean) / position_scale,
    ]
    design = np.column_stack(columns).astype(np.float64)
    if design.shape[1] != len(HMM_FEATURE_NAMES):
        raise AssertionError("registered HMM design dimension changed")
    return design, {
        "position_mean": position_mean,
        "position_scale": position_scale,
    }

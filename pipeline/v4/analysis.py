"""Mouse-sharded v4 model execution and registered aggregation."""

from __future__ import annotations

import hashlib
import json
import platform
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .behavior import compile_behavior
from .constants import (
    BASIS_GRID,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CACHE_SCHEMA,
    CELL_SEEDS,
    MOUSE_SCHEMA,
    METHOD_REVISION,
    PRIMARY_HMM_K,
    REQUIRED_MICE,
    RESULT_SCHEMA,
    RIDGE_GRID,
    TYPED_REASONS,
)
from .hazard import (
    build_risk_rows,
    evaluate_prequential,
    load_neural_trials,
    one_se_hazard,
)
from .hmm_checkpoint import load_target_posteriors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _typed_reason(exc: Exception) -> str:
    if isinstance(exc, MemoryError):
        return "runtime_resource_exhaustion"
    message = str(exc)
    if "Cannot allocate memory" in message or "std::bad_alloc" in message:
        return "runtime_resource_exhaustion"
    if type(exc).__name__ in {"XlaRuntimeError", "JaxRuntimeError"}:
        return "hmm_backend_failure"
    for reason in TYPED_REASONS:
        if message == reason or message.startswith(reason + ":"):
            return reason
    return "source_integrity_failure"


def _read_session(cache: Path, row) -> dict:
    experiment_id = int(row.ophys_experiment_id)
    trials = pd.read_parquet(cache / f"{experiment_id}.trials.parquet")
    stim = pd.read_parquet(cache / f"{experiment_id}.stim.parquet")
    licks = pd.read_parquet(cache / f"{experiment_id}.licks.parquet")
    rewards = pd.read_parquet(cache / f"{experiment_id}.rewards.parquet")
    eye = pd.read_parquet(cache / f"{experiment_id}.eye.parquet")
    running = pd.read_parquet(cache / f"{experiment_id}.running.parquet")
    behavior = compile_behavior(
        trials,
        stim,
        licks,
        rewards,
        eye,
        running,
        neural_valid=trials.neural_valid,
    )
    cells, neural = load_neural_trials(cache / f"{experiment_id}.time.h5")
    return {
        "experiment_id": experiment_id,
        "behavior_session_id": int(row.behavior_session_id),
        "mouse_id": int(row.mouse_id),
        "behavior": behavior,
        "cells": cells,
        "neural": neural,
    }


def _candidate_grid(model: str):
    if model == "M0":
        for penalty in RIDGE_GRID:
            yield 1, 0, penalty
    else:
        for basis in BASIS_GRID:
            for seed in CELL_SEEDS:
                for penalty in RIDGE_GRID:
                    yield basis, seed, penalty


def _tune_model(
    target: dict,
    target_hmm,
    sessions_by_id: Mapping[int, dict],
    *,
    model: str,
) -> tuple[int | None, float, list[dict]]:
    rows: list[dict] = []
    for basis, seed, penalty in _candidate_grid(model):
        for tuning_id, probabilities in target_hmm.inner_probs.items():
            tuning = sessions_by_id[tuning_id]
            try:
                risk = build_risk_rows(
                    tuning["behavior"],
                    probabilities,
                    tuning["neural"],
                    experiment_id=tuning["experiment_id"],
                    seed=seed,
                    basis_count=basis,
                    signal="events",
                    include_neural=model != "M0",
                )
                score, n_trials, _, _ = evaluate_prequential(
                    risk,
                    n_raw_trials=len(tuning["behavior"]),
                    model=model,
                    penalty=penalty,
                )
                status, reason = "estimable", None
            except Exception as exc:
                score, n_trials = np.nan, 0
                status, reason = "nonestimable", _typed_reason(exc)
            rows.append(
                {
                    "target_session": target["behavior_session_id"],
                    "tuning_session": tuning_id,
                    "model": model,
                    "basis_count": basis if model != "M0" else np.nan,
                    "cell_seed": seed if model != "M0" else np.nan,
                    "penalty": penalty,
                    "per_trial_loglik": score,
                    "n_trials": n_trials,
                    "status": status,
                    "reason": reason,
                }
            )
    frame = pd.DataFrame(rows)
    basis, penalty = one_se_hazard(frame, model=model)
    frame["selected"] = (
        np.isclose(frame.penalty, penalty)
        & (
            True
            if model == "M0"
            else frame.basis_count.astype(float).eq(float(basis))
        )
    )
    return basis, penalty, frame.to_dict("records")


def _evaluate_target(
    target: dict,
    target_hmm,
    *,
    selected: Mapping[str, tuple[int | None, float]],
    include_dff: bool = True,
    analysis_label: str = "primary_k2",
) -> tuple[list[dict], list[dict], list[dict]]:
    seed_rows: list[dict] = []
    block_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    m0_basis, m0_penalty = selected["M0"]
    m0_risk = build_risk_rows(
        target["behavior"],
        target_hmm.target_probs,
        target["neural"],
        experiment_id=target["experiment_id"],
        seed=0,
        basis_count=1,
        signal="events",
        include_neural=False,
    )
    m0, m0_trials, blocks, coefficients = evaluate_prequential(
        m0_risk,
        n_raw_trials=len(target["behavior"]),
        model="M0",
        penalty=m0_penalty,
    )
    for row in blocks:
        block_rows.append(
            {
                **row,
                "behavior_session_id": target["behavior_session_id"],
                "model": "M0",
                "signal": "events",
                "cell_seed": np.nan,
                "analysis": analysis_label,
            }
        )
    for row in coefficients:
        coefficient_rows.append(
            {
                **row,
                "behavior_session_id": target["behavior_session_id"],
                "signal": "events",
                "cell_seed": np.nan,
                "analysis": analysis_label,
            }
        )

    for seed in CELL_SEEDS:
        m1_basis, m1_penalty = selected["M1"]
        event_risk = build_risk_rows(
            target["behavior"],
            target_hmm.target_probs,
            target["neural"],
            experiment_id=target["experiment_id"],
            seed=seed,
            basis_count=int(m1_basis),
            signal="events",
        )
        m1, n_trials, blocks, coefficients = evaluate_prequential(
            event_risk,
            n_raw_trials=len(target["behavior"]),
            model="M1",
            penalty=m1_penalty,
        )
        if n_trials != m0_trials:
            raise ValueError("hazard_incomplete_prediction")
        dff_m1, dff_blocks, dff_coefficients = np.nan, [], []
        if include_dff:
            dff_risk = build_risk_rows(
                target["behavior"],
                target_hmm.target_probs,
                target["neural"],
                experiment_id=target["experiment_id"],
                seed=seed,
                basis_count=int(m1_basis),
                signal="dff",
            )
            dff_m1, dff_trials, dff_blocks, dff_coefficients = evaluate_prequential(
                dff_risk,
                n_raw_trials=len(target["behavior"]),
                model="M1",
                penalty=m1_penalty,
            )
            if dff_trials != m0_trials:
                raise ValueError("hazard_incomplete_prediction")
        m2 = np.nan
        m2_status = "not_applicable_k1"
        if target_hmm.selected_k > 1:
            m2_basis, m2_penalty = selected["M2"]
            m2_risk = build_risk_rows(
                target["behavior"],
                target_hmm.target_probs,
                target["neural"],
                experiment_id=target["experiment_id"],
                seed=seed,
                basis_count=int(m2_basis),
                signal="events",
            )
            m2, m2_trials, m2_blocks, m2_coefficients = evaluate_prequential(
                m2_risk,
                n_raw_trials=len(target["behavior"]),
                model="M2",
                penalty=m2_penalty,
            )
            if m2_trials != m0_trials:
                raise ValueError("hazard_incomplete_prediction")
            m2_status = "estimable"
            for row in m2_blocks:
                block_rows.append(
                    {
                        **row,
                        "behavior_session_id": target["behavior_session_id"],
                        "model": "M2",
                        "signal": "events",
                        "cell_seed": seed,
                        "analysis": analysis_label,
                    }
                )
            for row in m2_coefficients:
                coefficient_rows.append(
                    {
                        **row,
                        "behavior_session_id": target["behavior_session_id"],
                        "signal": "events",
                        "cell_seed": seed,
                        "analysis": analysis_label,
                    }
                )
        seed_rows.append(
            {
                "ophys_experiment_id": target["experiment_id"],
                "behavior_session_id": target["behavior_session_id"],
                "mouse_id": target["mouse_id"],
                "cell_seed": seed,
                "selected_k": target_hmm.selected_k,
                "n_evaluated_trials": n_trials,
                "m0_per_trial_loglik": m0,
                "m1_per_trial_loglik": m1,
                "delta_ll": m1 - m0,
                "m2_per_trial_loglik": m2,
                "m2_minus_m1": m2 - m1 if np.isfinite(m2) else np.nan,
                "m2_status": m2_status,
                "dff_m1_per_trial_loglik": dff_m1,
                "dff_delta_ll": dff_m1 - m0,
                "status": "estimable",
                "reason": None,
                "analysis": analysis_label,
            }
        )
        for signal, source_blocks, source_coefficients in (
            ("events", blocks, coefficients),
            ("dff", dff_blocks, dff_coefficients),
        ):
            for row in source_blocks:
                block_rows.append(
                    {
                        **row,
                        "behavior_session_id": target["behavior_session_id"],
                        "model": "M1",
                        "signal": signal,
                        "cell_seed": seed,
                        "analysis": analysis_label,
                    }
                )
            for row in source_coefficients:
                coefficient_rows.append(
                    {
                        **row,
                        "behavior_session_id": target["behavior_session_id"],
                        "signal": signal,
                        "cell_seed": seed,
                        "analysis": analysis_label,
                    }
                )
    return seed_rows, block_rows, coefficient_rows


def _write_table(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def fit_mouse(
    cache: Path,
    manifest_path: Path,
    out: Path,
    *,
    mouse_id: int,
    cache_release: str,
    cache_manifest_sha256: str,
    prereg_sha256: str,
    environment_sha256: str,
    hmm_checkpoints: Path,
    hmm_release: str,
    hmm_manifest_sha256: str,
    hmm_seeds=None,
    hmm_max_iter=None,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest.role.eq("active") & manifest.mouse_id.astype(int).eq(int(mouse_id))
    ]
    n_expected_sessions = len(selected)
    if n_expected_sessions == 0:
        raise ValueError(f"mouse {mouse_id} has no active sessions")
    sessions = [_read_session(cache, row) for row in selected.itertuples(index=False)]
    sessions_by_id = {item["behavior_session_id"]: item for item in sessions}
    behavior_by_id = {key: value["behavior"] for key, value in sessions_by_id.items()}
    out.mkdir(parents=True, exist_ok=True)
    hmm_manifest_path = hmm_checkpoints / "hmm-release-manifest.json"
    if not hmm_manifest_path.exists() or _sha256(hmm_manifest_path) != hmm_manifest_sha256:
        raise ValueError("checkpoint_integrity_failure: HMM manifest hash mismatch")
    hmm_manifest = json.loads(hmm_manifest_path.read_text())
    expected_hmm = {
        "schema": "neural-dev-v4-hmm-release-v1",
        "method_revision": METHOD_REVISION,
        "primary_k": PRIMARY_HMM_K,
        "k_selection_performed": False,
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "prereg_sha256": prereg_sha256,
        "environment_sha256": environment_sha256,
    }
    if any(hmm_manifest.get(key) != value for key, value in expected_hmm.items()):
        raise ValueError("checkpoint_integrity_failure: HMM provenance mismatch")

    trial_flow: list[dict] = []
    posterior_rows: list[dict] = []
    hmm_selection: list[dict] = []
    hmm_starts: list[dict] = []
    hmm_parameters: list[dict] = []
    tuning_rows: list[dict] = []
    seed_rows: list[dict] = []
    block_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    sensitivity_seed_rows: list[dict] = []
    failures: list[dict] = []
    for session in sessions:
        for row in session["behavior"].to_dict("records"):
            trial_flow.append(
                {
                    "ophys_experiment_id": session["experiment_id"],
                    "behavior_session_id": session["behavior_session_id"],
                    "mouse_id": int(mouse_id),
                    **row,
                }
            )
        target_id = session["behavior_session_id"]
        try:
            tuning_ids = sorted(set(sessions_by_id) - {int(target_id)})
            target_probs, inner_probs = load_target_posteriors(
                hmm_checkpoints,
                mouse_id=int(mouse_id),
                target_session=int(target_id),
                tuning_sessions=tuning_ids,
            )
            if len(target_probs) != len(session["behavior"]):
                raise ValueError("checkpoint_integrity_failure: target alignment")
            target_hmm = SimpleNamespace(
                selected_k=PRIMARY_HMM_K,
                target_probs=target_probs,
                inner_probs=inner_probs,
            )
            hmm_selection.append(
                {
                    "target_session": target_id,
                    "k": PRIMARY_HMM_K,
                    "selected": True,
                    "selection_performed": False,
                }
            )
            for trial_index, probabilities in enumerate(target_hmm.target_probs):
                posterior_rows.append(
                    {
                        "target_session": target_id,
                        "trial_index": trial_index,
                        **{
                            f"state_{state}": float(value)
                            for state, value in enumerate(probabilities)
                        },
                    }
                )
            selected_hyper: dict[str, tuple[int | None, float]] = {}
            for model in ("M0", "M1"):
                basis, penalty, candidates = _tune_model(
                    session, target_hmm, sessions_by_id, model=model
                )
                selected_hyper[model] = (basis, penalty)
                for candidate in candidates:
                    candidate["analysis"] = "primary_k2"
                tuning_rows.extend(candidates)
            if target_hmm.selected_k > 1:
                basis, penalty, candidates = _tune_model(
                    session, target_hmm, sessions_by_id, model="M2"
                )
                selected_hyper["M2"] = (basis, penalty)
                for candidate in candidates:
                    candidate["analysis"] = "primary_k2"
                tuning_rows.extend(candidates)
            target_seeds, target_blocks, target_coefficients = _evaluate_target(
                session, target_hmm, selected=selected_hyper
            )
            seed_rows.extend(target_seeds)
            block_rows.extend(target_blocks)
            coefficient_rows.extend(target_coefficients)

            # Sensitivity failures are isolated from the K=2 primary status.
            # The K=1 analysis has independent external tuning, no dF/F
            # replication, and no M2 interaction.
            try:
                k1_hmm = SimpleNamespace(
                    selected_k=1,
                    target_probs=np.ones((len(session["behavior"]), 1), float),
                    inner_probs={
                        tuning_id: np.ones(
                            (len(sessions_by_id[tuning_id]["behavior"]), 1), float
                        )
                        for tuning_id in tuning_ids
                    },
                )
                k1_selected: dict[str, tuple[int | None, float]] = {}
                for model in ("M0", "M1"):
                    basis, penalty, candidates = _tune_model(
                        session, k1_hmm, sessions_by_id, model=model
                    )
                    k1_selected[model] = (basis, penalty)
                    for candidate in candidates:
                        candidate["analysis"] = "sensitivity_k1_no_state"
                    tuning_rows.extend(candidates)
                k1_seeds, k1_blocks, k1_coefficients = _evaluate_target(
                    session,
                    k1_hmm,
                    selected=k1_selected,
                    include_dff=False,
                    analysis_label="sensitivity_k1_no_state",
                )
                sensitivity_seed_rows.extend(k1_seeds)
                block_rows.extend(k1_blocks)
                coefficient_rows.extend(k1_coefficients)
            except Exception as exc:
                typed_reason = _typed_reason(exc)
                failures.append(
                    {
                        "behavior_session_id": target_id,
                        "analysis": "sensitivity_k1_no_state",
                        "reason": typed_reason,
                        "detail": str(exc),
                        "exception_type": type(exc).__name__,
                    }
                )
                sensitivity_seed_rows.append(
                    {
                        "ophys_experiment_id": session["experiment_id"],
                        "behavior_session_id": target_id,
                        "mouse_id": int(mouse_id),
                        "analysis": "sensitivity_k1_no_state",
                        "status": "nonestimable",
                        "reason": typed_reason,
                    }
                )
        except Exception as exc:
            typed_reason = _typed_reason(exc)
            failures.append(
                {
                    "behavior_session_id": target_id,
                    "reason": typed_reason,
                    "detail": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            seed_rows.append(
                {
                    "ophys_experiment_id": session["experiment_id"],
                    "behavior_session_id": target_id,
                    "mouse_id": int(mouse_id),
                    "status": "nonestimable",
                    "reason": typed_reason,
                }
            )

    _write_table(trial_flow, out / "trial_flow.parquet")
    _write_table(posterior_rows, out / "predictive_state.parquet")
    _write_table(hmm_selection, out / "hmm_selection.parquet")
    _write_table(hmm_starts, out / "hmm_starts.parquet")
    _write_table(hmm_parameters, out / "hmm_parameters.parquet")
    _write_table(tuning_rows, out / "hazard_tuning.parquet")
    _write_table(seed_rows, out / "session_seeds.parquet")
    _write_table(block_rows, out / "hazard_blocks.parquet")
    _write_table(coefficient_rows, out / "hazard_coefficients.parquet")
    _write_table(failures, out / "typed_failures.parquet")
    pd.DataFrame(
        sensitivity_seed_rows,
        columns=sorted(
            {
                "ophys_experiment_id",
                "behavior_session_id",
                "mouse_id",
                "cell_seed",
                "selected_k",
                "n_evaluated_trials",
                "m0_per_trial_loglik",
                "m1_per_trial_loglik",
                "delta_ll",
                "m2_per_trial_loglik",
                "m2_minus_m1",
                "m2_status",
                "dff_m1_per_trial_loglik",
                "dff_delta_ll",
                "status",
                "reason",
                "analysis",
            }
            | {
                key
                for row in sensitivity_seed_rows
                for key in row
            }
        ),
    ).to_parquet(out / "k1_hazard_sensitivity.parquet", index=False)
    behavior_sensitivity = pd.read_parquet(
        hmm_checkpoints / "behavior_sensitivity.parquet"
    )
    behavior_sensitivity[
        behavior_sensitivity.mouse_id.astype(int).eq(int(mouse_id))
    ].to_parquet(out / "behavior_sensitivity.parquet", index=False)

    estimable = pd.DataFrame(seed_rows)
    n_estimable_sessions = (
        int(
            estimable[estimable.get("status", pd.Series(dtype=str)).eq("estimable")]
            .behavior_session_id.nunique()
        )
        if len(estimable) and "behavior_session_id" in estimable
        else 0
    )
    result = {
        "schema": MOUSE_SCHEMA,
        "method_revision": METHOD_REVISION,
        "primary_k": PRIMARY_HMM_K,
        "k_selection_performed": False,
        "mouse_id": int(mouse_id),
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "prereg_sha256": prereg_sha256,
        "environment_sha256": environment_sha256,
        "hmm_release": hmm_release,
        "hmm_manifest_sha256": hmm_manifest_sha256,
        "n_expected_sessions": n_expected_sessions,
        "n_estimable_sessions": n_estimable_sessions,
        "status": "estimable" if n_estimable_sessions else "nonestimable",
        "diagnostics_complete": True,
        "failures": failures,
        "typed_reasons": sorted(
            {
                str(row["reason"])
                for row in [*failures, *tuning_rows, *seed_rows]
                if row.get("reason")
            }
        ),
        "numeric_sesoi": None,
        "confirm_ready": False,
        "allen_nwb_download": False,
    }
    (out / "mouse-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _bca(values: np.ndarray) -> dict:
    from scipy.stats import norm

    values = np.asarray(values, float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return {"mean": None, "low": None, "high": None, "status": "nonestimable"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, (BOOTSTRAP_REPLICATES, len(values)), replace=True)
    bootstrap = samples.mean(axis=1)
    observed = float(values.mean())
    proportion = np.clip(np.mean(bootstrap < observed), 1e-12, 1 - 1e-12)
    z0 = norm.ppf(proportion)
    jackknife = np.array(
        [np.delete(values, index).mean() for index in range(len(values))]
    )
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * np.sum(centered**2) ** 1.5
    acceleration = np.sum(centered**3) / denominator if denominator else 0.0
    adjusted = []
    for alpha in (0.025, 0.975):
        z = norm.ppf(alpha)
        adjusted.append(
            norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        )
    low, high = np.quantile(bootstrap, adjusted)
    return {
        "mean": observed,
        "low": float(low),
        "high": float(high),
        "status": "estimable",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def aggregate(
    mouse_results: Path,
    manifest_path: Path,
    out: Path,
    *,
    cache_release: str,
    cache_manifest_sha256: str,
    prereg_sha256: str,
    environment_sha256: str,
    hmm_release: str,
    hmm_manifest_sha256: str,
) -> dict:
    from scipy.stats import t

    source_manifest = pd.read_csv(manifest_path)
    expected_mice = sorted(
        source_manifest.loc[source_manifest.role.eq("active"), "mouse_id"]
        .astype(int)
        .unique()
    )
    if len(expected_mice) != 10:
        raise ValueError("aggregate requires exactly ten immutable DEV mice")
    manifests, seed_frames, k1_frames, behavior_frames = [], [], [], []
    for path in sorted(mouse_results.rglob("mouse-manifest.json")):
        manifest = json.loads(path.read_text())
        manifests.append(manifest)
        seed_frames.append(pd.read_parquet(path.parent / "session_seeds.parquet"))
        k1_frames.append(
            pd.read_parquet(path.parent / "k1_hazard_sensitivity.parquet")
        )
        behavior_frames.append(
            pd.read_parquet(path.parent / "behavior_sensitivity.parquet")
        )
    found = sorted(int(item["mouse_id"]) for item in manifests)
    if found != expected_mice:
        raise ValueError(f"mouse shard mismatch expected={expected_mice} found={found}")
    for item in manifests:
        expected = {
            "schema": MOUSE_SCHEMA,
            "cache_release": cache_release,
            "cache_manifest_sha256": cache_manifest_sha256,
            "prereg_sha256": prereg_sha256,
            "environment_sha256": environment_sha256,
            "hmm_release": hmm_release,
            "hmm_manifest_sha256": hmm_manifest_sha256,
            "method_revision": METHOD_REVISION,
            "primary_k": PRIMARY_HMM_K,
            "k_selection_performed": False,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"mouse {item.get('mouse_id')} {key} mismatch")
    seeds = pd.concat(seed_frames, ignore_index=True)
    k1_seeds = pd.concat(k1_frames, ignore_index=True)
    behavior_sensitivity = pd.concat(behavior_frames, ignore_index=True)
    valid = seeds[
        seeds.get("status", pd.Series(index=seeds.index, dtype=str)).eq("estimable")
        & seeds.get("delta_ll", pd.Series(index=seeds.index, dtype=float)).notna()
    ].copy()
    session_rows = []
    for (mouse, session), group in valid.groupby(["mouse_id", "behavior_session_id"]):
        if set(group.cell_seed.astype(int)) != set(CELL_SEEDS):
            continue
        session_rows.append(
            {
                "mouse_id": int(mouse),
                "behavior_session_id": int(session),
                "n_evaluated_trials": int(group.n_evaluated_trials.iloc[0]),
                "delta_ll": float(group.delta_ll.mean()),
                "m2_minus_m1": float(group.m2_minus_m1.mean())
                if group.m2_minus_m1.notna().all()
                else np.nan,
                "dff_delta_ll": float(group.dff_delta_ll.mean()),
            }
        )
    sessions = pd.DataFrame(
        session_rows,
        columns=[
            "mouse_id",
            "behavior_session_id",
            "n_evaluated_trials",
            "delta_ll",
            "m2_minus_m1",
            "dff_delta_ll",
        ],
    )
    mouse_rows = []
    for mouse, group in sessions.groupby("mouse_id"):
        weights = group.n_evaluated_trials.to_numpy(float)
        mouse_rows.append(
            {
                "mouse_id": int(mouse),
                "n_sessions": len(group),
                "n_evaluated_trials": int(weights.sum()),
                "delta_ll": float(np.average(group.delta_ll, weights=weights)),
                "m2_minus_m1": float(
                    np.average(
                        group.loc[group.m2_minus_m1.notna(), "m2_minus_m1"],
                        weights=group.loc[group.m2_minus_m1.notna(), "n_evaluated_trials"],
                    )
                )
                if group.m2_minus_m1.notna().any()
                else np.nan,
                "dff_delta_ll": float(np.average(group.dff_delta_ll, weights=weights)),
            }
        )
    mice = pd.DataFrame(
        mouse_rows,
        columns=[
            "mouse_id",
            "n_sessions",
            "n_evaluated_trials",
            "delta_ll",
            "m2_minus_m1",
            "dff_delta_ll",
        ],
    )

    k1_valid = k1_seeds[
        k1_seeds.get(
            "status", pd.Series(index=k1_seeds.index, dtype=str)
        ).eq("estimable")
        & k1_seeds.get(
            "delta_ll", pd.Series(index=k1_seeds.index, dtype=float)
        ).notna()
    ].copy()
    k1_session_rows = []
    for (mouse, session), group in k1_valid.groupby(
        ["mouse_id", "behavior_session_id"]
    ):
        if set(group.cell_seed.astype(int)) != set(CELL_SEEDS):
            continue
        k1_session_rows.append(
            {
                "mouse_id": int(mouse),
                "behavior_session_id": int(session),
                "n_evaluated_trials": int(group.n_evaluated_trials.iloc[0]),
                "delta_ll": float(group.delta_ll.mean()),
            }
        )
    k1_sessions = pd.DataFrame(
        k1_session_rows,
        columns=[
            "mouse_id",
            "behavior_session_id",
            "n_evaluated_trials",
            "delta_ll",
        ],
    )
    k1_mouse_rows = []
    for mouse, group in k1_sessions.groupby("mouse_id"):
        weights = group.n_evaluated_trials.to_numpy(float)
        k1_mouse_rows.append(
            {
                "mouse_id": int(mouse),
                "n_sessions": int(len(group)),
                "n_evaluated_trials": int(weights.sum()),
                "delta_ll": float(np.average(group.delta_ll, weights=weights)),
            }
        )
    k1_mice = pd.DataFrame(
        k1_mouse_rows,
        columns=["mouse_id", "n_sessions", "n_evaluated_trials", "delta_ll"],
    )

    behavior_mouse = (
        behavior_sensitivity.groupby("mouse_id", as_index=False)[
            ["k2_minus_k1", "k3_minus_k2"]
        ]
        .mean()
        .sort_values("mouse_id")
    )

    def diagnostic_t(values) -> dict:
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return {
                "n_mice": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
                "t95": [None, None],
            }
        value_mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        critical = float(t.ppf(0.975, len(values) - 1))
        return {
            "n_mice": int(len(values)),
            "mean": value_mean,
            "standard_error": se,
            "t95": [value_mean - critical * se, value_mean + critical * se],
        }
    coverage = len(mice)
    group_status = "estimable" if coverage >= REQUIRED_MICE else "nonestimable_mouse_coverage"
    values = mice.delta_ll.to_numpy(float) if len(mice) else np.empty(0)
    if coverage >= REQUIRED_MICE:
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / np.sqrt(coverage))
        critical = float(t.ppf(0.975, coverage - 1))
        interval = [mean - critical * standard_error, mean + critical * standard_error]
    else:
        mean, standard_error, interval = None, None, [None, None]
    lomo = [
        {
            "left_out_mouse": int(mouse),
            "mean": float(mice.loc[~mice.mouse_id.eq(mouse), "delta_ll"].mean()),
        }
        for mouse in mice.mouse_id
    ]
    diagnostics_complete = all(bool(item["diagnostics_complete"]) for item in manifests)
    integrity_complete = True  # all strict schema/hash comparisons above passed
    v41 = bool(
        integrity_complete
        and coverage >= REQUIRED_MICE
        and diagnostics_complete
    )
    typed_failures = [
        {
            "mouse_id": int(item["mouse_id"]),
            **failure,
        }
        for item in manifests
        for failure in item.get("failures", [])
    ]
    typed_reason_summary = [
        {"mouse_id": int(item["mouse_id"]), "reason": reason}
        for item in manifests
        for reason in item.get("typed_reasons", [])
    ]
    mouse_coverage = [
        {
            "mouse_id": int(item["mouse_id"]),
            "status": item.get("status"),
            "n_estimable_sessions": int(item.get("n_estimable_sessions", 0)),
            "reasons": sorted(
                {
                    failure.get("reason", "source_integrity_failure")
                    for failure in item.get("failures", [])
                }
            ),
        }
        for item in manifests
    ]
    out.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(out / "sessions.parquet", index=False)
    mice.to_parquet(out / "mice.parquet", index=False)
    k1_sessions.to_parquet(
        out / "k1_sensitivity_sessions.parquet", index=False
    )
    k1_mice.to_parquet(out / "k1_sensitivity_mice.parquet", index=False)
    behavior_sensitivity.to_parquet(
        out / "behavior_sensitivity_sessions.parquet", index=False
    )
    behavior_mouse.to_parquet(
        out / "behavior_sensitivity_mice.parquet", index=False
    )
    pd.DataFrame(lomo).to_parquet(out / "leave_one_mouse_out.parquet", index=False)
    result = {
        "schema": RESULT_SCHEMA,
        "method_revision": METHOD_REVISION,
        "behavior_model": {
            "primary_k": PRIMARY_HMM_K,
            "k_selection_performed": False,
            "sensitivity_k": [1, 3],
            "checkpoint_release": hmm_release,
            "checkpoint_manifest_sha256": hmm_manifest_sha256,
        },
        "status": group_status,
        "cache_source": {
            "release": cache_release,
            "manifest_sha256": cache_manifest_sha256,
            "schema": CACHE_SCHEMA,
            "n_active_experiments": 50,
            "allen_nwb_download": False,
        },
        "prereg_sha256": prereg_sha256,
        "environment_sha256": environment_sha256,
        "primary": {
            "status": group_status,
            "estimand": "M1_minus_M0_per_trial_heldout_loglik",
            "n_estimable_mice": coverage,
            "mean": mean,
            "standard_error": standard_error,
            "t95": interval,
        },
        "secondary": {
            "m2_minus_m1": {
                "status": (
                    "estimable"
                    if len(mice) and mice.m2_minus_m1.notna().any()
                    else "not_applicable_or_nonestimable"
                ),
                "mouse_mean": (
                    float(mice.m2_minus_m1.mean())
                    if len(mice) and mice.m2_minus_m1.notna().any()
                    else None
                ),
            },
            "dff_replication": {
                "status": "estimable" if len(mice) else "nonestimable",
                "m1_minus_m0_mouse_mean": (
                    float(mice.dff_delta_ll.mean()) if len(mice) else None
                ),
            },
        },
        "sensitivity": {
            "k1_no_state_events_m1_minus_m0": {
                "status": (
                    "estimable"
                    if len(k1_mice) >= REQUIRED_MICE
                    else "nonestimable_mouse_coverage"
                ),
                "required_mice": REQUIRED_MICE,
                **diagnostic_t(k1_mice.delta_ll),
            },
            "behavior_adequacy": {
                "k2_minus_k1": diagnostic_t(behavior_mouse.k2_minus_k1),
                "k3_minus_k2": diagnostic_t(behavior_mouse.k3_minus_k2),
                "changes_primary_k": False,
            },
        },
        "diagnostics": {
            "mouse_bca": _bca(values),
            "leave_one_mouse_out_complete": len(lomo) == coverage,
            "registered_diagnostics_complete": diagnostics_complete,
        },
        "coverage": {
            "required_mice": REQUIRED_MICE,
            "estimable_mice": coverage,
            "population_interpretation": coverage >= REQUIRED_MICE,
            "mice": mouse_coverage,
        },
        "integrity": {
            "status": "passed",
            "schema_hash_provenance_match": integrity_complete,
        },
        "typed_failures": typed_failures,
        "typed_reason_summary": typed_reason_summary,
        "v4_1_eligible": v41,
        "numeric_sesoi": None,
        "confirm_ready": False,
        "confirm_data_accessed": False,
        "allen_nwb_download": False,
    }
    (out / "analysis-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

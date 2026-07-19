"""r3 target-sharded hazard execution with atomic candidate checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis import _read_session, _typed_reason
from .constants import (
    BASIS_GRID,
    CELL_COUNT,
    CELL_SEEDS,
    HAZARD_CHECKPOINT_SCHEMA,
    HAZARD_PLAN_SCHEMA,
    METHOD_REVISION,
    N_BLOCKS,
    PRIMARY_HMM_K,
    RIDGE_GRID,
    TARGET_SCHEMA,
)
from .hazard import (
    HazardEvaluationFailure,
    build_risk_rows,
    evaluate_prepared,
    one_se_hazard,
    prepare_prequential,
    raw_blocks,
)
from .hmm_checkpoint import load_target_posteriors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hazard_plan(manifest_path: Path) -> dict:
    source = pd.read_csv(manifest_path)
    active = source[source.role.eq("active")].copy()
    if (
        len(active) != 50
        or active.behavior_session_id.nunique() != 50
        or active.mouse_id.nunique() != 10
    ):
        raise ValueError("source_integrity_failure: expected exact 50 active / 10 mice")
    targets = []
    for row in active.sort_values("behavior_session_id").itertuples(index=False):
        targets.append(
            {
                "target_session": int(row.behavior_session_id),
                "ophys_experiment_id": int(row.ophys_experiment_id),
                "mouse_id": int(row.mouse_id),
                "container_id": int(row.ophys_container_id),
            }
        )
    if len({item["target_session"] for item in targets}) != 50:
        raise ValueError("source_integrity_failure: duplicate target session")
    return {
        "schema": HAZARD_PLAN_SCHEMA,
        "method_revision": METHOD_REVISION,
        "n_targets": 50,
        "targets": targets,
    }


def _provenance_digest(provenance: Mapping) -> str:
    payload = json.dumps(dict(provenance), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _valid_group(path: Path, provenance: Mapping, group_id: str) -> bool:
    manifest_path = path / "group-manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != HAZARD_CHECKPOINT_SCHEMA
            or manifest.get("group_id") != group_id
            or manifest.get("provenance_sha256") != _provenance_digest(provenance)
            or manifest.get("status") != "complete"
        ):
            return False
        return all(
            (path / name).is_file() and _sha256(path / name) == digest
            for name, digest in manifest.get("files", {}).items()
        )
    except Exception:
        return False


def _write_group(
    destination: Path,
    *,
    group_id: str,
    provenance: Mapping,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping | None = None,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        files: dict[str, str] = {}
        for name, frame in tables.items():
            filename = f"{name}.parquet"
            frame.to_parquet(temporary / filename, index=False)
            files[filename] = _sha256(temporary / filename)
        manifest = {
            "schema": HAZARD_CHECKPOINT_SCHEMA,
            "method_revision": METHOD_REVISION,
            "group_id": group_id,
            "status": "complete",
            "provenance_sha256": _provenance_digest(provenance),
            "files": files,
            **dict(metadata or {}),
        }
        (temporary / "group-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _checkpoint_group(
    root: Path,
    group_id: str,
    provenance: Mapping,
    compute: Callable[[], tuple[Mapping[str, pd.DataFrame], Mapping]],
) -> tuple[dict, dict[str, pd.DataFrame], bool]:
    path = root / "groups" / group_id
    resumed = _valid_group(path, provenance, group_id)
    if not resumed:
        tables, metadata = compute()
        _write_group(
            path,
            group_id=group_id,
            provenance=provenance,
            tables=tables,
            metadata=metadata,
        )
    manifest = json.loads((path / "group-manifest.json").read_text())
    tables = {
        name.removesuffix(".parquet"): pd.read_parquet(path / name)
        for name in manifest["files"]
        if name.endswith(".parquet")
    }
    return manifest, tables, resumed


def _preflight_one(
    session: dict, probabilities: np.ndarray
) -> tuple[bool, str | None, str | None, int, int]:
    behavior = session["behavior"]
    if (
        probabilities.shape[0] != len(behavior)
        or probabilities.ndim != 2
        or not np.all(np.isfinite(probabilities))
    ):
        return False, "hazard_tuning_session_ineligible", "invalid HMM posterior", 0, 0
    if len(session["cells"]) < CELL_COUNT:
        return (
            False,
            "hazard_tuning_session_ineligible",
            "neural_fewer_than_50_cells",
            0,
            0,
        )
    try:
        risk = build_risk_rows(
            behavior,
            probabilities,
            session["neural"],
            experiment_id=session["experiment_id"],
            seed=0,
            basis_count=1,
            signal="events",
            include_neural=True,
        )
        blocks = raw_blocks(len(behavior))
        test_counts, event_counts = [], []
        for block_index in range(1, N_BLOCKS):
            train_limit = int(blocks[block_index - 1][-1])
            test_indices = set(map(int, blocks[block_index]))
            train = risk[risk.raw_trial_index <= train_limit]
            test = risk[risk.raw_trial_index.isin(test_indices)]
            test_counts.append(int(test.trial_id.nunique()))
            event_counts.append(int(train.event.sum()))
        if any(count == 0 for count in test_counts):
            return (
                False,
                "hazard_tuning_session_ineligible",
                "hazard_empty_test_block",
                min(test_counts),
                min(event_counts),
            )
        if any(count == 0 for count in event_counts):
            return (
                False,
                "hazard_tuning_session_ineligible",
                "hazard_no_training_event",
                min(test_counts),
                min(event_counts),
            )
        return True, None, None, min(test_counts), min(event_counts)
    except Exception as exc:
        return (
            False,
            "hazard_tuning_session_ineligible",
            f"{_typed_reason(exc)}: {exc}",
            0,
            0,
        )


def require_eligible_sessions(session_ids: Sequence[int]) -> tuple[int, ...]:
    eligible = tuple(sorted(map(int, session_ids)))
    if len(eligible) < 2:
        raise ValueError("hazard_tuning_insufficient_sessions")
    return eligible


def _candidate_group(
    target: dict,
    sessions_by_id: Mapping[int, dict],
    probabilities: Mapping[int, np.ndarray],
    eligible_sessions: Sequence[int],
    *,
    analysis: str,
    model: str,
    basis: int,
    seed: int,
) -> tuple[Mapping[str, pd.DataFrame], Mapping]:
    candidate_rows: list[dict] = []
    block_rows: list[dict] = []
    for tuning_id in eligible_sessions:
        tuning = sessions_by_id[int(tuning_id)]
        try:
            risk = build_risk_rows(
                tuning["behavior"],
                probabilities[int(tuning_id)],
                tuning["neural"],
                experiment_id=tuning["experiment_id"],
                seed=seed,
                basis_count=basis,
                signal="events",
                include_neural=model != "M0",
            )
            folds = prepare_prequential(
                risk, n_raw_trials=len(tuning["behavior"]), model=model
            )
        except Exception as exc:
            reason = _typed_reason(exc)
            if reason in {"runtime_resource_exhaustion", "hmm_backend_failure"}:
                raise
            for penalty in RIDGE_GRID:
                candidate_rows.append(
                    {
                        "target_session": target["behavior_session_id"],
                        "tuning_session": int(tuning_id),
                        "analysis": analysis,
                        "model": model,
                        "basis_count": np.nan if model == "M0" else basis,
                        "cell_seed": np.nan if model == "M0" else seed,
                        "penalty": penalty,
                        "per_trial_loglik": np.nan,
                        "n_trials": 0,
                        "status": "nonestimable",
                        "reason": reason,
                        "detail": str(exc),
                        "exception_type": type(exc).__name__,
                    }
                )
            continue
        for penalty in RIDGE_GRID:
            try:
                score, n_trials, blocks, _ = evaluate_prepared(
                    folds, penalty=penalty
                )
                status, reason, detail, exception_type = (
                    "estimable",
                    None,
                    None,
                    None,
                )
                for row in blocks:
                    block_rows.append(
                        {
                            **row,
                            "target_session": target["behavior_session_id"],
                            "tuning_session": int(tuning_id),
                            "analysis": analysis,
                            "model": model,
                            "basis_count": np.nan if model == "M0" else basis,
                            "cell_seed": np.nan if model == "M0" else seed,
                            "penalty": penalty,
                            "status": "estimable",
                            "reason": None,
                        }
                    )
            except (MemoryError, KeyboardInterrupt):
                raise
            except Exception as exc:
                score, n_trials = np.nan, 0
                status, reason = "nonestimable", _typed_reason(exc)
                if reason in {"runtime_resource_exhaustion", "hmm_backend_failure"}:
                    raise
                detail, exception_type = str(exc), type(exc).__name__
                if isinstance(exc, HazardEvaluationFailure):
                    for row in exc.block_rows:
                        block_rows.append(
                            {
                                **row,
                                "target_session": target["behavior_session_id"],
                                "tuning_session": int(tuning_id),
                                "analysis": analysis,
                                "model": model,
                                "basis_count": (
                                    np.nan if model == "M0" else basis
                                ),
                                "cell_seed": np.nan if model == "M0" else seed,
                                "penalty": penalty,
                            }
                        )
            candidate_rows.append(
                {
                    "target_session": target["behavior_session_id"],
                    "tuning_session": int(tuning_id),
                    "analysis": analysis,
                    "model": model,
                    "basis_count": np.nan if model == "M0" else basis,
                    "cell_seed": np.nan if model == "M0" else seed,
                    "penalty": penalty,
                    "per_trial_loglik": score,
                    "n_trials": n_trials,
                    "status": status,
                    "reason": reason,
                    "detail": detail,
                    "exception_type": exception_type,
                }
            )
    return (
        {
            "candidates": pd.DataFrame(candidate_rows),
            "blocks": pd.DataFrame(
                block_rows,
                columns=sorted(
                    {
                        "target_session",
                        "tuning_session",
                        "analysis",
                        "model",
                        "basis_count",
                        "cell_seed",
                        "penalty",
                        "test_block",
                        "status",
                        "reason",
                    }
                    | {key for row in block_rows for key in row}
                ),
            ),
        },
        {
            "analysis": analysis,
            "model": model,
            "basis_count": int(basis),
            "cell_seed": int(seed),
            "n_eligible_sessions": len(eligible_sessions),
        },
    )


def _select(
    rows: pd.DataFrame,
    *,
    model: str,
    eligible_sessions: Sequence[int],
) -> tuple[tuple[int | None, float] | None, pd.DataFrame]:
    summaries: list[dict] = []
    for (basis, penalty), group in rows.groupby(
        ["basis_count", "penalty"], dropna=False
    ):
        session_counts = group.groupby("tuning_session").cell_seed.nunique(
            dropna=True
        )
        expected_seeds = 1 if model == "M0" else len(CELL_SEEDS)
        complete = bool(
            set(map(int, group.tuning_session.unique()))
            == set(map(int, eligible_sessions))
            and group.status.eq("estimable").all()
            and (
                model == "M0"
                or (
                    set(map(int, session_counts.index))
                    == set(map(int, eligible_sessions))
                    and session_counts.eq(expected_seeds).all()
                )
            )
        )
        summaries.append(
            {
                "basis_count": basis,
                "penalty": float(penalty),
                "complete": complete,
                "n_rows": len(group),
                "n_estimable_rows": int(group.status.eq("estimable").sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    try:
        selected = one_se_hazard(
            rows, model=model, eligible_sessions=eligible_sessions
        )
    except ValueError as exc:
        if str(exc) != "hazard_no_complete_candidate":
            raise
        summary["selected"] = False
        return None, summary
    basis, penalty = selected
    summary["selected"] = np.isclose(summary.penalty, penalty) & (
        True
        if model == "M0"
        else summary.basis_count.astype(float).eq(float(basis))
    )
    return selected, summary


def _evaluation_group(
    target: dict,
    probabilities: np.ndarray,
    *,
    analysis: str,
    model: str,
    basis: int,
    seed: int,
    penalty: float,
    signal: str,
) -> tuple[Mapping[str, pd.DataFrame], Mapping]:
    try:
        risk = build_risk_rows(
            target["behavior"],
            probabilities,
            target["neural"],
            experiment_id=target["experiment_id"],
            seed=seed,
            basis_count=basis,
            signal=signal,
            include_neural=model != "M0",
        )
        folds = prepare_prequential(
            risk, n_raw_trials=len(target["behavior"]), model=model
        )
        score, n_trials, blocks, coefficients = evaluate_prepared(
            folds, penalty=penalty
        )
        result = pd.DataFrame(
            [
                {
                    "analysis": analysis,
                    "model": model,
                    "signal": signal,
                    "cell_seed": np.nan if model == "M0" else seed,
                    "basis_count": np.nan if model == "M0" else basis,
                    "penalty": penalty,
                    "per_trial_loglik": score,
                    "n_evaluated_trials": n_trials,
                    "status": "estimable",
                    "reason": None,
                    "detail": None,
                    "exception_type": None,
                }
            ]
        )
        block_frame = pd.DataFrame(blocks)
        coefficient_frame = pd.DataFrame(coefficients)
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as exc:
        reason = _typed_reason(exc)
        if reason in {"runtime_resource_exhaustion", "hmm_backend_failure"}:
            raise
        result = pd.DataFrame(
            [
                {
                    "analysis": analysis,
                    "model": model,
                    "signal": signal,
                    "cell_seed": np.nan if model == "M0" else seed,
                    "basis_count": np.nan if model == "M0" else basis,
                    "penalty": penalty,
                    "per_trial_loglik": np.nan,
                    "n_evaluated_trials": 0,
                    "status": "nonestimable",
                    "reason": reason,
                    "detail": str(exc),
                    "exception_type": type(exc).__name__,
                }
            ]
        )
        block_frame = pd.DataFrame(
            exc.block_rows
            if isinstance(exc, HazardEvaluationFailure)
            else [],
            columns=sorted(
                {
                    "test_block",
                    "analysis",
                    "model",
                    "signal",
                    "cell_seed",
                }
                | (
                    {
                        key
                        for row in exc.block_rows
                        for key in row
                    }
                    if isinstance(exc, HazardEvaluationFailure)
                    else set()
                )
            ),
        )
        coefficient_frame = pd.DataFrame(
            exc.coefficient_rows
            if isinstance(exc, HazardEvaluationFailure)
            else [],
            columns=sorted(
                {
                    "test_block",
                    "coefficient_index",
                    "coefficient",
                    "analysis",
                    "model",
                    "signal",
                    "cell_seed",
                }
                | (
                    {
                        key
                        for row in exc.coefficient_rows
                        for key in row
                    }
                    if isinstance(exc, HazardEvaluationFailure)
                    else set()
                )
            ),
        )
    for frame in (block_frame, coefficient_frame):
        if not frame.empty:
            frame["analysis"] = analysis
            frame["model"] = model
            frame["signal"] = signal
            frame["cell_seed"] = np.nan if model == "M0" else seed
    return (
        {
            "result": result,
            "blocks": block_frame,
            "coefficients": coefficient_frame,
        },
        {
            "analysis": analysis,
            "model": model,
            "signal": signal,
            "basis_count": int(basis),
            "cell_seed": int(seed),
            "penalty": float(penalty),
        },
    )


def _empty_status(
    target: dict, analysis: str, reason: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ophys_experiment_id": target["experiment_id"],
                "behavior_session_id": target["behavior_session_id"],
                "mouse_id": target["mouse_id"],
                "analysis": analysis,
                "status": "nonestimable",
                "reason": reason,
            }
        ]
    )


def assemble_seed_frame(
    target: Mapping,
    analysis: str,
    evaluation_results: Mapping[tuple[str, str, str, int], pd.Series],
) -> pd.DataFrame:
    """Assemble one analysis without letting secondary failures alter primary."""

    m0 = evaluation_results.get((analysis, "M0", "events", 0))
    rows = []
    for seed in CELL_SEEDS:
        m1 = evaluation_results.get((analysis, "M1", "events", seed))
        primary_ok = bool(
            m0 is not None
            and m1 is not None
            and m0.status == "estimable"
            and m1.status == "estimable"
        )
        common = {
            "ophys_experiment_id": target["experiment_id"],
            "behavior_session_id": target["behavior_session_id"],
            "mouse_id": target["mouse_id"],
            "cell_seed": seed,
            "selected_k": 2 if analysis == "primary_k2" else 1,
            "analysis": analysis,
        }
        if not primary_ok:
            reasons = [
                item.get("reason")
                for item in (m0, m1)
                if item is not None and item.get("reason")
            ]
            rows.append(
                {
                    **common,
                    "status": "nonestimable",
                    "reason": (
                        reasons[0]
                        if reasons
                        else "hazard_no_complete_candidate"
                    ),
                }
            )
            continue
        m2 = evaluation_results.get((analysis, "M2", "events", seed))
        dff = evaluation_results.get((analysis, "M1", "dff", seed))
        m2_ok = m2 is not None and m2.status == "estimable"
        dff_ok = dff is not None and dff.status == "estimable"
        rows.append(
            {
                **common,
                "n_evaluated_trials": int(m1.n_evaluated_trials),
                "m0_per_trial_loglik": float(m0.per_trial_loglik),
                "m1_per_trial_loglik": float(m1.per_trial_loglik),
                "delta_ll": float(m1.per_trial_loglik - m0.per_trial_loglik),
                "m2_per_trial_loglik": (
                    float(m2.per_trial_loglik) if m2_ok else np.nan
                ),
                "m2_minus_m1": (
                    float(m2.per_trial_loglik - m1.per_trial_loglik)
                    if m2_ok
                    else np.nan
                ),
                "m2_status": (
                    "estimable"
                    if m2_ok
                    else (
                        "not_applicable_k1"
                        if analysis != "primary_k2"
                        else "nonestimable"
                    )
                ),
                "m2_reason": (
                    None if m2_ok or m2 is None else m2.get("reason")
                ),
                "dff_m1_per_trial_loglik": (
                    float(dff.per_trial_loglik) if dff_ok else np.nan
                ),
                "dff_delta_ll": (
                    float(dff.per_trial_loglik - m0.per_trial_loglik)
                    if dff_ok
                    else np.nan
                ),
                "dff_status": "estimable" if dff_ok else "nonestimable",
                "dff_reason": (
                    None if dff_ok or dff is None else dff.get("reason")
                ),
                "status": "estimable",
                "reason": None,
            }
        )
    return pd.DataFrame(rows)


def fit_target(
    cache: Path,
    manifest_path: Path,
    out: Path,
    *,
    target_session: int,
    cache_release: str,
    cache_manifest_sha256: str,
    hmm_prereg_sha256: str,
    hazard_prereg_sha256: str,
    environment_sha256: str,
    code_commit: str,
    hmm_checkpoints: Path,
    hmm_release: str,
    hmm_manifest_sha256: str,
) -> dict:
    source = pd.read_csv(manifest_path)
    active = source[source.role.eq("active")].copy()
    target_rows = active[
        active.behavior_session_id.astype(int).eq(int(target_session))
    ]
    if len(active) != 50 or len(target_rows) != 1:
        raise ValueError("source_integrity_failure: target is not in exact-50 manifest")
    target_row = target_rows.iloc[0]
    mouse_id = int(target_row.mouse_id)
    mouse_rows = active[active.mouse_id.astype(int).eq(mouse_id)]
    sessions = [_read_session(cache, row) for row in mouse_rows.itertuples(index=False)]
    sessions_by_id = {item["behavior_session_id"]: item for item in sessions}
    target = sessions_by_id[int(target_session)]
    tuning_ids = sorted(set(sessions_by_id) - {int(target_session)})
    if len(tuning_ids) < 2:
        raise ValueError("hazard_tuning_insufficient_sessions")

    hmm_manifest_path = hmm_checkpoints / "hmm-release-manifest.json"
    if (
        not hmm_manifest_path.exists()
        or _sha256(hmm_manifest_path) != hmm_manifest_sha256
    ):
        raise ValueError("checkpoint_integrity_failure: HMM manifest hash mismatch")
    hmm_manifest = json.loads(hmm_manifest_path.read_text())
    expected_hmm = {
        "schema": "neural-dev-v4-hmm-release-v1",
        "method_revision": "r2",
        "primary_k": PRIMARY_HMM_K,
        "k_selection_performed": False,
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "prereg_sha256": hmm_prereg_sha256,
        "environment_sha256": environment_sha256,
        "n_fits": 263,
        "n_nonestimable_fits": 0,
    }
    for key, value in expected_hmm.items():
        if hmm_manifest.get(key) != value:
            raise ValueError(f"checkpoint_integrity_failure: HMM {key} mismatch")

    target_probs, inner_probs = load_target_posteriors(
        hmm_checkpoints,
        mouse_id=mouse_id,
        target_session=int(target_session),
        tuning_sessions=tuning_ids,
    )
    k1_target = np.ones((len(target["behavior"]), 1), np.float64)
    k1_inner = {
        session_id: np.ones((len(sessions_by_id[session_id]["behavior"]), 1))
        for session_id in tuning_ids
    }
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "target_session": int(target_session),
        "mouse_id": mouse_id,
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "hmm_release": hmm_release,
        "hmm_manifest_sha256": hmm_manifest_sha256,
        "hmm_prereg_sha256": hmm_prereg_sha256,
        "hazard_prereg_sha256": hazard_prereg_sha256,
        "environment_sha256": environment_sha256,
        "code_commit": code_commit,
        "basis_grid": list(BASIS_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "cell_seeds": list(CELL_SEEDS),
    }
    stage_rows: list[dict] = []

    def compute_preflight():
        rows = []
        for session_id in tuning_ids:
            eligible, reason, detail, min_test, min_events = _preflight_one(
                sessions_by_id[session_id], inner_probs[session_id]
            )
            rows.append(
                {
                    "target_session": int(target_session),
                    "tuning_session": int(session_id),
                    "eligible": eligible,
                    "status": "eligible" if eligible else "excluded",
                    "reason": reason,
                    "detail": detail,
                    "min_test_trials": min_test,
                    "min_training_prefix_events": min_events,
                }
            )
        return {"preflight": pd.DataFrame(rows)}, {"stage": "preflight"}

    _, preflight_tables, resumed = _checkpoint_group(
        out, "preflight", provenance, compute_preflight
    )
    preflight = preflight_tables["preflight"]
    eligible_sessions = sorted(
        preflight.loc[preflight.eligible, "tuning_session"].astype(int)
    )
    if len(eligible_sessions) >= 2:
        eligible_sessions = list(require_eligible_sessions(eligible_sessions))
    stage_rows.append(
        {
            "stage": "preflight",
            "status": "completed",
            "resumed": resumed,
            "reason": (
                None
                if len(eligible_sessions) >= 2
                else "hazard_tuning_insufficient_sessions"
            ),
        }
    )

    all_candidates: list[pd.DataFrame] = []
    all_tuning_blocks: list[pd.DataFrame] = []
    selections: dict[tuple[str, str], tuple[int | None, float] | None] = {}
    selection_summaries: list[pd.DataFrame] = []
    analyses = (
        ("primary_k2", inner_probs, ("M0", "M1", "M2")),
        ("sensitivity_k1_no_state", k1_inner, ("M0", "M1")),
    )
    if len(eligible_sessions) >= 2:
        for analysis, probabilities, models in analyses:
            for model in models:
                bases = (1,) if model == "M0" else BASIS_GRID
                seeds = (0,) if model == "M0" else CELL_SEEDS
                model_frames: list[pd.DataFrame] = []
                for basis in bases:
                    for seed in seeds:
                        group_id = (
                            f"tune-{analysis}-{model}-b{basis}-seed{seed}"
                        )

                        def compute(
                            analysis=analysis,
                            probabilities=probabilities,
                            model=model,
                            basis=basis,
                            seed=seed,
                        ):
                            return _candidate_group(
                                target,
                                sessions_by_id,
                                probabilities,
                                eligible_sessions,
                                analysis=analysis,
                                model=model,
                                basis=basis,
                                seed=seed,
                            )

                        _, tables, was_resumed = _checkpoint_group(
                            out, group_id, provenance, compute
                        )
                        model_frames.append(tables["candidates"])
                        all_candidates.append(tables["candidates"])
                        if "blocks" in tables and not tables["blocks"].empty:
                            all_tuning_blocks.append(tables["blocks"])
                        stage_rows.append(
                            {
                                "stage": group_id,
                                "status": "completed",
                                "resumed": was_resumed,
                                "reason": None,
                            }
                        )
                model_rows = pd.concat(model_frames, ignore_index=True)
                selection, summary = _select(
                    model_rows,
                    model=model,
                    eligible_sessions=eligible_sessions,
                )
                selections[(analysis, model)] = selection
                summary["analysis"] = analysis
                summary["model"] = model
                selection_summaries.append(summary)
    else:
        for analysis, _, models in analyses:
            for model in models:
                selections[(analysis, model)] = None
                bases = (1,) if model == "M0" else BASIS_GRID
                seeds = (0,) if model == "M0" else CELL_SEEDS
                skipped = []
                for basis in bases:
                    for seed in seeds:
                        group_id = (
                            f"tune-{analysis}-{model}-b{basis}-seed{seed}"
                        )
                        stage_rows.append(
                            {
                                "stage": group_id,
                                "status": "skipped",
                                "resumed": False,
                                "reason": "hazard_tuning_insufficient_sessions",
                            }
                        )
                        for penalty in RIDGE_GRID:
                            skipped.append(
                                {
                                    "target_session": int(target_session),
                                    "tuning_session": np.nan,
                                    "analysis": analysis,
                                    "model": model,
                                    "basis_count": (
                                        np.nan if model == "M0" else basis
                                    ),
                                    "cell_seed": (
                                        np.nan if model == "M0" else seed
                                    ),
                                    "penalty": penalty,
                                    "per_trial_loglik": np.nan,
                                    "n_trials": 0,
                                    "status": "skipped",
                                    "reason": (
                                        "hazard_tuning_insufficient_sessions"
                                    ),
                                    "detail": None,
                                    "exception_type": None,
                                }
                            )
                all_candidates.append(pd.DataFrame(skipped))

    evaluation_results: dict[tuple[str, str, str, int], pd.Series] = {}
    outer_blocks: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []

    def run_evaluation(
        analysis: str,
        probabilities: np.ndarray,
        model: str,
        signal: str,
        seed: int,
    ) -> None:
        selected = selections.get((analysis, model))
        group_id = f"eval-{analysis}-{model}-{signal}-seed{seed}"
        if selected is None:
            stage_rows.append(
                {
                    "stage": group_id,
                    "status": "skipped",
                    "resumed": False,
                    "reason": "hazard_no_complete_candidate",
                }
            )
            return
        basis, penalty = selected

        def compute():
            return _evaluation_group(
                target,
                probabilities,
                analysis=analysis,
                model=model,
                basis=int(basis or 1),
                seed=seed,
                penalty=penalty,
                signal=signal,
            )

        _, tables, was_resumed = _checkpoint_group(
            out, group_id, provenance, compute
        )
        row = tables["result"].iloc[0]
        evaluation_results[(analysis, model, signal, seed)] = row
        if "blocks" in tables and not tables["blocks"].empty:
            frame = tables["blocks"].copy()
            frame["behavior_session_id"] = int(target_session)
            outer_blocks.append(frame)
        if "coefficients" in tables and not tables["coefficients"].empty:
            frame = tables["coefficients"].copy()
            frame["behavior_session_id"] = int(target_session)
            coefficients.append(frame)
        stage_rows.append(
            {
                "stage": group_id,
                "status": "completed",
                "resumed": was_resumed,
                "reason": row.get("reason"),
            }
        )

    for analysis, probabilities in (
        ("primary_k2", target_probs),
        ("sensitivity_k1_no_state", k1_target),
    ):
        run_evaluation(analysis, probabilities, "M0", "events", 0)
        for seed in CELL_SEEDS:
            run_evaluation(analysis, probabilities, "M1", "events", seed)
    for seed in CELL_SEEDS:
        run_evaluation("primary_k2", target_probs, "M2", "events", seed)
        # dF/F reuses the event-selected M1 basis and penalty.
        run_evaluation("primary_k2", target_probs, "M1", "dff", seed)

    primary_seeds = assemble_seed_frame(
        target, "primary_k2", evaluation_results
    )
    k1_seeds = assemble_seed_frame(
        target, "sensitivity_k1_no_state", evaluation_results
    )
    primary_complete = bool(
        len(primary_seeds) == len(CELL_SEEDS)
        and primary_seeds.status.eq("estimable").all()
    )
    failures = []
    for frame in all_candidates:
        if not frame.empty:
            failures.extend(
                frame.loc[frame.status.ne("estimable")]
                .to_dict("records")
            )
    for row in evaluation_results.values():
        if row.status != "estimable":
            failures.append(row.to_dict())
    if not primary_complete and not failures:
        failures.append(
            {
                "analysis": "primary_k2",
                "reason": (
                    "hazard_tuning_insufficient_sessions"
                    if len(eligible_sessions) < 2
                    else "hazard_no_complete_candidate"
                ),
            }
        )

    (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame(
            columns=[
                "target_session",
                "tuning_session",
                "analysis",
                "model",
                "basis_count",
                "cell_seed",
                "penalty",
                "per_trial_loglik",
                "n_trials",
                "status",
                "reason",
                "detail",
                "exception_type",
            ]
        )
    ).to_parquet(out / "hazard_tuning.parquet", index=False)
    (
        pd.concat(all_tuning_blocks, ignore_index=True)
        if all_tuning_blocks
        else pd.DataFrame(columns=["target_session", "tuning_session", "status"])
    ).to_parquet(out / "hazard_tuning_blocks.parquet", index=False)
    (
        pd.concat(selection_summaries, ignore_index=True)
        if selection_summaries
        else pd.DataFrame(
            columns=[
                "analysis",
                "model",
                "basis_count",
                "penalty",
                "complete",
                "selected",
            ]
        )
    ).to_parquet(out / "hazard_candidate_summary.parquet", index=False)
    preflight.to_parquet(out / "hazard_preflight.parquet", index=False)
    primary_seeds.to_parquet(out / "session_seeds.parquet", index=False)
    k1_seeds.to_parquet(out / "k1_hazard_sensitivity.parquet", index=False)
    (
        pd.concat(outer_blocks, ignore_index=True)
        if outer_blocks
        else pd.DataFrame(columns=["behavior_session_id", "test_block", "status"])
    ).to_parquet(out / "hazard_blocks.parquet", index=False)
    (
        pd.concat(coefficients, ignore_index=True)
        if coefficients
        else pd.DataFrame(
            columns=[
                "behavior_session_id",
                "test_block",
                "coefficient_index",
                "coefficient",
            ]
        )
    ).to_parquet(out / "hazard_coefficients.parquet", index=False)
    pd.DataFrame(stage_rows).to_parquet(out / "stage_status.parquet", index=False)
    pd.DataFrame(
        failures,
        columns=sorted(
            {"analysis", "reason", "detail", "exception_type"}
            | {key for row in failures for key in row}
        ),
    ).to_parquet(out / "typed_failures.parquet", index=False)
    pd.DataFrame(
        [
            {
                "target_session": int(target_session),
                "trial_index": index,
                **{
                    f"state_{state}": float(value)
                    for state, value in enumerate(probability)
                },
            }
            for index, probability in enumerate(target_probs)
        ]
    ).to_parquet(out / "predictive_state.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ophys_experiment_id": target["experiment_id"],
                "behavior_session_id": int(target_session),
                "mouse_id": mouse_id,
                **row,
            }
            for row in target["behavior"].to_dict("records")
        ]
    ).to_parquet(out / "trial_flow.parquet", index=False)
    behavior_sensitivity = pd.read_parquet(
        hmm_checkpoints / "behavior_sensitivity.parquet"
    )
    behavior_sensitivity[
        behavior_sensitivity.target_session.astype(int).eq(
            int(target_session)
        )
    ].to_parquet(out / "behavior_sensitivity.parquet", index=False)

    typed_reasons = sorted(
        {
            str(row.get("reason"))
            for row in failures
            if row.get("reason") is not None
        }
        | set(
            preflight.loc[preflight.reason.notna(), "reason"].astype(str)
        )
    )
    result_files = {
        path.name: _sha256(path)
        for path in sorted(out.glob("*.parquet"))
    }
    result = {
        "schema": TARGET_SCHEMA,
        "method_revision": METHOD_REVISION,
        "target_session": int(target_session),
        "ophys_experiment_id": target["experiment_id"],
        "mouse_id": mouse_id,
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "hmm_release": hmm_release,
        "hmm_manifest_sha256": hmm_manifest_sha256,
        "hmm_prereg_sha256": hmm_prereg_sha256,
        "hazard_prereg_sha256": hazard_prereg_sha256,
        "environment_sha256": environment_sha256,
        "code_commit": code_commit,
        "n_eligible_tuning_sessions": len(eligible_sessions),
        "n_expected_seeds": len(CELL_SEEDS),
        "primary_status": "estimable" if primary_complete else "nonestimable",
        "status": "complete",
        "diagnostics_complete": True,
        "typed_reasons": typed_reasons,
        "files": result_files,
        "numeric_sesoi": None,
        "confirm_ready": False,
        "confirm_data_accessed": False,
        "allen_nwb_download": False,
    }
    _, _, final_resumed = _checkpoint_group(
        out,
        "final-target",
        provenance,
        lambda: (
            {
                "result_files": pd.DataFrame(
                    [
                        {"filename": name, "sha256": digest}
                        for name, digest in sorted(result_files.items())
                    ]
                )
            },
            {
                "stage": "final-target",
                "primary_status": result["primary_status"],
            },
        ),
    )
    result["final_checkpoint_resumed"] = final_resumed
    result["final_checkpoint_sha256"] = _sha256(
        out / "groups" / "final-target" / "group-manifest.json"
    )
    (out / "target-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def fit_mouse_targets(
    cache: Path,
    manifest_path: Path,
    out: Path,
    *,
    mouse_id: int,
    **kwargs,
) -> dict:
    """Local compatibility wrapper; the formal workflow uses ``fit-target``."""

    source = pd.read_csv(manifest_path)
    targets = sorted(
        source.loc[
            source.role.eq("active")
            & source.mouse_id.astype(int).eq(int(mouse_id)),
            "behavior_session_id",
        ].astype(int)
    )
    if not targets:
        raise ValueError(f"mouse {mouse_id} has no active sessions")
    results = [
        fit_target(
            cache,
            manifest_path,
            out / f"target-{target}",
            target_session=target,
            **kwargs,
        )
        for target in targets
    ]
    result = {
        "schema": "neural-dev-v4-local-mouse-wrapper-v1",
        "method_revision": METHOD_REVISION,
        "mouse_id": int(mouse_id),
        "n_targets": len(results),
        "n_primary_estimable": sum(
            item["primary_status"] == "estimable" for item in results
        ),
        "status": "complete",
        "numeric_sesoi": None,
        "confirm_ready": False,
        "confirm_data_accessed": False,
    }
    (out / "mouse-wrapper-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result

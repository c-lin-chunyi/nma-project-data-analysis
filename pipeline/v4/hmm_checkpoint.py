"""Immutable, fixed-shape GLM-HMM checkpoints for the v4 r2 analysis."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .behavior import compile_behavior
from .constants import (
    HMM_CHECKPOINT_SCHEMA,
    HMM_K_GRID,
    HMM_MAX_ITER,
    HMM_RELEASE_SCHEMA,
    HMM_SEEDS,
    HMM_TOL,
    HMM_METHOD_REVISION,
    PRIMARY_HMM_K,
)
from .hmm import (
    HMMNoConvergence,
    fit_hmm,
    marginal_loglik,
    predictive_state_probs,
    state_order,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FitSpec:
    mouse_id: int
    k: int
    excluded_sessions: tuple[int, ...]
    role: str

    @property
    def fit_id(self) -> str:
        excluded = "-".join(map(str, self.excluded_sessions))
        return f"mouse-{self.mouse_id}-k{self.k}-exclude-{excluded}"


def fit_specs(session_ids: Sequence[int], mouse_id: int) -> list[FitSpec]:
    """Return the exact r2 fit-key universe for one mouse."""

    ids = tuple(sorted(map(int, session_ids)))
    if len(ids) < 3:
        raise ValueError("hmm_insufficient_training_sessions")
    specs: list[FitSpec] = []
    for target in ids:
        specs.append(FitSpec(int(mouse_id), 1, (target,), "sensitivity_outer"))
        specs.append(FitSpec(int(mouse_id), 2, (target,), "primary_outer"))
        specs.append(FitSpec(int(mouse_id), 3, (target,), "sensitivity_outer"))
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            specs.append(
                FitSpec(
                    int(mouse_id),
                    PRIMARY_HMM_K,
                    (left, right),
                    "primary_nested",
                )
            )
    return sorted(specs, key=lambda item: (item.k, item.excluded_sessions))


def plan_chunks(manifest_path: Path, *, max_fit_keys: int = 5) -> dict:
    """Build a deterministic GitHub matrix with at most five fits per process."""

    if not 1 <= int(max_fit_keys) <= 5:
        raise ValueError("max_fit_keys must be in [1,5]")
    manifest = pd.read_csv(manifest_path)
    active = manifest[manifest.role.eq("active")].copy()
    if len(active) != 50 or active.mouse_id.nunique() != 10:
        raise ValueError("source_integrity_failure: expected 50 active / 10 mice")
    chunks: list[dict] = []
    for mouse_id, rows in active.groupby("mouse_id", sort=True):
        containers = sorted(rows.ophys_container_id.astype(int).unique())
        if len(containers) != 1:
            raise ValueError(f"mouse {mouse_id} does not map to one container")
        specs = fit_specs(rows.behavior_session_id.astype(int), int(mouse_id))
        for chunk_index, start in enumerate(range(0, len(specs), max_fit_keys)):
            selected = specs[start : start + max_fit_keys]
            chunks.append(
                {
                    "mouse_id": int(mouse_id),
                    "container_id": int(containers[0]),
                    "chunk_id": int(chunk_index),
                    "fit_ids": ",".join(item.fit_id for item in selected),
                }
            )
    return {
        "schema": "neural-dev-v4-hmm-plan-v1",
        "method_revision": HMM_METHOD_REVISION,
        "max_fit_keys_per_chunk": int(max_fit_keys),
        "chunks": chunks,
    }


def _read_behavior(cache: Path, row) -> pd.DataFrame:
    experiment_id = int(row.ophys_experiment_id)
    trials = pd.read_parquet(cache / f"{experiment_id}.trials.parquet")
    return compile_behavior(
        trials,
        pd.read_parquet(cache / f"{experiment_id}.stim.parquet"),
        pd.read_parquet(cache / f"{experiment_id}.licks.parquet"),
        pd.read_parquet(cache / f"{experiment_id}.rewards.parquet"),
        pd.read_parquet(cache / f"{experiment_id}.eye.parquet"),
        pd.read_parquet(cache / f"{experiment_id}.running.parquet"),
        neural_valid=trials.neural_valid,
    )


def load_mouse_behavior(
    cache: Path, manifest_path: Path, mouse_id: int
) -> dict[int, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    rows = manifest[
        manifest.role.eq("active") & manifest.mouse_id.astype(int).eq(int(mouse_id))
    ]
    if len(rows) < 3:
        raise ValueError("hmm_insufficient_training_sessions")
    return {
        int(row.behavior_session_id): _read_behavior(cache, row)
        for row in rows.itertuples(index=False)
    }


def _write_checkpoint(
    destination: Path,
    spec: FitSpec,
    fit,
    heldout: Mapping[int, pd.DataFrame],
    *,
    fixed_shape: tuple[int, int],
    provenance: Mapping[str, str],
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{spec.fit_id}.", dir=destination.parent)
    )
    try:
        order = state_order(fit, fit.params.emissions.weights.shape[-1])
        initial = np.asarray(fit.params.initial.probs, float)[order]
        transition = np.asarray(
            fit.params.transitions.transition_matrix, float
        )[np.ix_(order, order)]
        emissions = np.asarray(fit.params.emissions.weights, float)[order]
        np.savez(
            temporary / "parameters.npz",
            initial_probs=initial,
            transition_matrix=transition,
            emission_weights=emissions,
            state_order=order,
            position_mean=np.asarray(fit.scaler["position_mean"]),
            position_scale=np.asarray(fit.scaler["position_scale"]),
        )
        starts = pd.DataFrame(
            [
                {
                    **row,
                    "fit_id": spec.fit_id,
                    "selected_start": int(row["seed"]) == int(fit.seed),
                }
                for row in fit.all_starts
            ]
        )
        starts.to_parquet(temporary / "starts.parquet", index=False)
        posterior_rows: list[dict] = []
        heldout_scores: list[dict] = []
        for session_id, behavior in sorted(heldout.items()):
            probability = predictive_state_probs(fit, behavior)[:, order]
            score = marginal_loglik(fit, behavior) / len(behavior)
            heldout_scores.append(
                {
                    "session_id": int(session_id),
                    "per_trial_loglik": float(score),
                }
            )
            for trial_index, values in enumerate(probability):
                posterior_rows.append(
                    {
                        "fit_id": spec.fit_id,
                        "predicted_session": int(session_id),
                        "trial_index": int(trial_index),
                        **{
                            f"state_{index}": float(value)
                            for index, value in enumerate(values)
                        },
                    }
                )
        pd.DataFrame(posterior_rows).to_parquet(
            temporary / "predictive.parquet", index=False
        )
        files = ("parameters.npz", "starts.parquet", "predictive.parquet")
        result = {
            "schema": HMM_CHECKPOINT_SCHEMA,
            "method_revision": HMM_METHOD_REVISION,
            "fit_id": spec.fit_id,
            "mouse_id": spec.mouse_id,
            "k": spec.k,
            "role": spec.role,
            "status": "estimable",
            "excluded_sessions": list(spec.excluded_sessions),
            "training_sessions": list(fit.training_session_ids),
            "fixed_shape": list(map(int, fixed_shape)),
            "selected_seed": int(fit.seed),
            "marginal_loglik": float(fit.marginal_loglik),
            "heldout_scores": heldout_scores,
            "files": {name: _sha256(temporary / name) for name in files},
            **dict(provenance),
        }
        (temporary / "fit-manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_failure_checkpoint(
    destination: Path,
    spec: FitSpec,
    *,
    training_sessions: Sequence[int],
    fixed_shape: tuple[int, int],
    reason: str,
    detail: str,
    provenance: Mapping[str, str],
    starts: Sequence | None = None,
) -> dict:
    """Atomically record a registered statistical nonestimability."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{spec.fit_id}.", dir=destination.parent)
    )
    try:
        files: dict[str, str] = {}
        if starts is not None:
            start_rows = [
                {
                    "k": int(start.k),
                    "seed": int(start.seed),
                    "converged": bool(start.converged),
                    "marginal_loglik": float(start.marginal_loglik),
                    "n_iterations": len(start.likelihood_trace),
                    "fit_id": spec.fit_id,
                    "selected_start": False,
                }
                for start in starts
            ]
            pd.DataFrame(start_rows).to_parquet(
                temporary / "starts.parquet", index=False
            )
            files["starts.parquet"] = _sha256(temporary / "starts.parquet")
        result = {
            "schema": HMM_CHECKPOINT_SCHEMA,
            "method_revision": HMM_METHOD_REVISION,
            "fit_id": spec.fit_id,
            "mouse_id": spec.mouse_id,
            "k": spec.k,
            "role": spec.role,
            "status": "nonestimable",
            "reason": reason,
            "detail": detail,
            "excluded_sessions": list(spec.excluded_sessions),
            "training_sessions": list(map(int, training_sessions)),
            "fixed_shape": list(map(int, fixed_shape)),
            "heldout_scores": [
                {
                    "session_id": int(session_id),
                    "per_trial_loglik": None,
                    "status": "nonestimable",
                    "reason": reason,
                }
                for session_id in spec.excluded_sessions
            ],
            "files": files,
            **dict(provenance),
        }
        (temporary / "fit-manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _valid_checkpoint(
    path: Path, expected: FitSpec, provenance: Mapping[str, str]
) -> bool:
    manifest_path = path / "fit-manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != HMM_CHECKPOINT_SCHEMA
            or manifest.get("fit_id") != expected.fit_id
            or int(manifest.get("k")) != expected.k
            or tuple(manifest.get("excluded_sessions", ()))
            != expected.excluded_sessions
        ):
            return False
        if any(manifest.get(key) != value for key, value in provenance.items()):
            return False
        if manifest.get("status", "estimable") not in {
            "estimable",
            "nonestimable",
        }:
            return False
        return all(
            _sha256(path / name) == digest
            for name, digest in manifest["files"].items()
        )
    except Exception:
        return False


def fit_chunk(
    cache: Path,
    manifest_path: Path,
    out: Path,
    *,
    mouse_id: int,
    fit_ids: Iterable[str],
    cache_release: str,
    cache_manifest_sha256: str,
    prereg_sha256: str,
    environment_sha256: str,
    code_commit: str,
    seeds: Sequence[int] = HMM_SEEDS,
    max_iter: int = HMM_MAX_ITER,
) -> dict:
    """Fit or resume one bounded set of exact HMM training keys."""

    sessions = load_mouse_behavior(cache, manifest_path, mouse_id)
    specs = {item.fit_id: item for item in fit_specs(sessions, mouse_id)}
    requested = tuple(value for value in fit_ids if value)
    if not requested or any(value not in specs for value in requested):
        raise ValueError("checkpoint_integrity_failure: unknown fit key")
    fixed_shape = (len(sessions) - 1, max(map(len, sessions.values())))
    provenance = {
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "prereg_sha256": prereg_sha256,
        "environment_sha256": environment_sha256,
        "code_commit": code_commit,
    }
    completed, resumed, nonestimable = [], [], []
    for fit_id in requested:
        spec = specs[fit_id]
        destination = out / fit_id
        if _valid_checkpoint(destination, spec, provenance):
            resumed.append(fit_id)
            existing = json.loads(
                (destination / "fit-manifest.json").read_text()
            )
            if existing.get("status") != "estimable":
                nonestimable.append(
                    {
                        "fit_id": fit_id,
                        "reason": existing.get("reason"),
                    }
                )
            continue
        training = {
            session_id: frame
            for session_id, frame in sessions.items()
            if session_id not in spec.excluded_sessions
        }
        heldout = {
            session_id: sessions[session_id] for session_id in spec.excluded_sessions
        }
        try:
            fit = fit_hmm(
                training,
                k=spec.k,
                seeds=seeds,
                max_iter=max_iter,
                tolerance=HMM_TOL,
                fixed_shape=fixed_shape,
            )
            checkpoint = _write_checkpoint(
                destination,
                spec,
                fit,
                heldout,
                fixed_shape=fixed_shape,
                provenance=provenance,
            )
        except HMMNoConvergence as exc:
            checkpoint = _write_failure_checkpoint(
                destination,
                spec,
                training_sessions=tuple(training),
                fixed_shape=fixed_shape,
                reason="hmm_no_converged_initialization",
                detail=str(exc),
                provenance=provenance,
                starts=exc.starts,
            )
        if checkpoint.get("status") != "estimable":
            nonestimable.append(
                {
                    "fit_id": fit_id,
                    "reason": checkpoint.get("reason"),
                }
            )
        completed.append(fit_id)
    result = {
        "schema": "neural-dev-v4-hmm-chunk-v1",
        "method_revision": HMM_METHOD_REVISION,
        "mouse_id": int(mouse_id),
        "requested": list(requested),
        "completed": completed,
        "resumed": resumed,
        "nonestimable": nonestimable,
        **provenance,
    }
    (out / "chunk-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_release(
    checkpoints: Path,
    manifest_path: Path,
    out: Path,
    *,
    cache_release: str,
    cache_manifest_sha256: str,
    prereg_sha256: str,
    environment_sha256: str,
    code_commit: str,
) -> dict:
    """Validate and consolidate every registered checkpoint into a release."""

    source = pd.read_csv(manifest_path)
    active = source[source.role.eq("active")]
    if (
        len(source) != 70
        or len(active) != 50
        or source[source.role.eq("passive")].shape[0] != 20
        or active.mouse_id.nunique() != 10
    ):
        raise ValueError(
            "source_integrity_failure: expected 50 active + 20 passive / 10 mice"
        )
    expected: dict[str, FitSpec] = {}
    for mouse_id, rows in active.groupby("mouse_id"):
        for spec in fit_specs(rows.behavior_session_id.astype(int), int(mouse_id)):
            expected[spec.fit_id] = spec
    found: dict[str, Path] = {}
    for manifest_path_found in checkpoints.rglob("fit-manifest.json"):
        value = json.loads(manifest_path_found.read_text())
        fit_id = value.get("fit_id")
        if fit_id in found:
            raise ValueError(f"checkpoint_integrity_failure: duplicate {fit_id}")
        found[fit_id] = manifest_path_found.parent
    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise ValueError(
            f"checkpoint_integrity_failure: missing={missing} extra={extra}"
        )
    provenance = {
        "cache_release": cache_release,
        "cache_manifest_sha256": cache_manifest_sha256,
        "prereg_sha256": prereg_sha256,
        "environment_sha256": environment_sha256,
        "code_commit": code_commit,
    }
    out.mkdir(parents=True, exist_ok=True)
    release_rows, behavior_rows = [], []
    for fit_id in sorted(expected):
        source_path = found[fit_id]
        if not _valid_checkpoint(source_path, expected[fit_id], provenance):
            raise ValueError(f"checkpoint_integrity_failure: invalid {fit_id}")
        manifest = json.loads((source_path / "fit-manifest.json").read_text())
        destination = out / "fits" / fit_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, destination)
        release_rows.append(manifest)
        if len(expected[fit_id].excluded_sessions) == 1:
            target = expected[fit_id].excluded_sessions[0]
            heldout = next(
                row
                for row in manifest["heldout_scores"]
                if int(row["session_id"]) == target
            )
            behavior_rows.append(
                {
                    "mouse_id": expected[fit_id].mouse_id,
                    "target_session": target,
                    "k": expected[fit_id].k,
                    "per_trial_loglik": heldout.get("per_trial_loglik"),
                    "status": manifest.get("status", "estimable"),
                    "reason": manifest.get("reason"),
                }
            )
    behavior = pd.DataFrame(behavior_rows)
    score_wide = behavior.pivot(
        index=["mouse_id", "target_session"],
        columns="k",
        values="per_trial_loglik",
    ).reset_index()
    score_wide.columns = [
        value if isinstance(value, str) else f"k{int(value)}_per_trial_loglik"
        for value in score_wide.columns
    ]
    status_wide = behavior.pivot(
        index=["mouse_id", "target_session"],
        columns="k",
        values="status",
    ).reset_index()
    status_wide.columns = [
        value if isinstance(value, str) else f"k{int(value)}_status"
        for value in status_wide.columns
    ]
    wide = score_wide.merge(
        status_wide, on=["mouse_id", "target_session"], validate="one_to_one"
    )
    wide["k2_minus_k1"] = (
        wide["k2_per_trial_loglik"] - wide["k1_per_trial_loglik"]
    )
    wide["k3_minus_k2"] = (
        wide["k3_per_trial_loglik"] - wide["k2_per_trial_loglik"]
    )
    wide.to_parquet(out / "behavior_sensitivity.parquet", index=False)
    result = {
        "schema": HMM_RELEASE_SCHEMA,
        "method_revision": HMM_METHOD_REVISION,
        "primary_k": PRIMARY_HMM_K,
        "k_selection_performed": False,
        "sensitivity_k": [1, 3],
        "n_mice": int(active.mouse_id.nunique()),
        "n_sessions": int(len(active)),
        "n_fits": len(release_rows),
        "n_nonestimable_fits": sum(
            row.get("status", "estimable") != "estimable"
            for row in release_rows
        ),
        "typed_failures": [
            {
                "fit_id": row["fit_id"],
                "mouse_id": row["mouse_id"],
                "k": row["k"],
                "role": row["role"],
                "reason": row.get("reason"),
            }
            for row in release_rows
            if row.get("status", "estimable") != "estimable"
        ],
        "fit_ids": sorted(expected),
        **provenance,
    }
    (out / "hmm-release-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def load_target_posteriors(
    release: Path,
    *,
    mouse_id: int,
    target_session: int,
    tuning_sessions: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load the primary target and nested tuning predictive posteriors."""

    def probability(excluded: Sequence[int], predicted: int) -> np.ndarray:
        fit_id = FitSpec(
            int(mouse_id),
            PRIMARY_HMM_K,
            tuple(sorted(map(int, excluded))),
            "lookup",
        ).fit_id
        path = release / "fits" / fit_id / "predictive.parquet"
        if not path.exists():
            raise ValueError(f"checkpoint_integrity_failure: missing {fit_id}")
        frame = pd.read_parquet(path)
        frame = frame[frame.predicted_session.astype(int).eq(int(predicted))]
        state_columns = sorted(
            (name for name in frame if name.startswith("state_")),
            key=lambda name: int(name.split("_")[1]),
        )
        if frame.empty or len(state_columns) != PRIMARY_HMM_K:
            raise ValueError("checkpoint_integrity_failure: posterior shape")
        frame = frame.sort_values("trial_index")
        trial_index = frame.trial_index.to_numpy(np.int64)
        values = frame[state_columns].to_numpy(float)
        if (
            not np.array_equal(trial_index, np.arange(len(frame)))
            or not np.all(np.isfinite(values))
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError("checkpoint_integrity_failure: posterior alignment")
        return values

    target = probability((target_session,), target_session)
    nested = {
        int(tuning): probability((target_session, int(tuning)), int(tuning))
        for tuning in tuning_sessions
    }
    return target, nested

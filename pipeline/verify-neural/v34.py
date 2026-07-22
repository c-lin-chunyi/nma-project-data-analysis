#!/usr/bin/env python3
"""Frozen CONFIRM v3.4 extraction and analysis entrypoint.

The statistical kernels are imported from v3.3.  This module supplies the
CONFIRM-only population, cache namespace, coverage rule, and typed decision;
it deliberately exposes no state-anchor, threshold-sweep, passive, or v4 path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import neural
import v33


FEATURE_SCHEMA = "neural-confirm-feature-cache-v1"
RESULT_SCHEMA = "neural-confirm-v3.4"
BEHAVIOR_SCHEMA = "behavioral-confirm-v3.4"
EXPECTED_EXPERIMENTS = 130
EXPECTED_MICE = 29
EXPECTED_CONTAINERS = 29
REQUIRED_MICE = 24
SESOI = 0.55
Q1_BOOTSTRAP_SEED = 3305
Q2_BOOTSTRAP_SEED = 3304
q1_kernel = v33._q1_session

IDENTITY_COLUMNS = {
    "ophys_experiment_id", "behavior_session_id", "ophys_container_id",
    "mouse_id", "project_code", "session_type",
}


def access_receipt(path: Path, *, run_id: int, commit: str,
                   freeze_release: str, freeze_sha256: str,
                   split_manifest_sha256: str) -> str:
    """Create a receipt once, or validate an exact same-run recovery."""
    expected = {
        "schema": "confirm-v3.4-access-v1", "run_id": int(run_id),
        "commit": str(commit), "freeze_release": str(freeze_release),
        "freeze_sha256": str(freeze_sha256),
        "split_manifest_sha256": str(split_manifest_sha256),
    }
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise ValueError(
                "receipt exists for a different run/provenance; new dispatch refused")
        return "resume"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(expected, handle, indent=2)
        handle.write("\n")
    return "create"


def write_freeze_manifest(out: Path, *, code_commit: str, prereg: Path,
                          requirements: Path, split_release: str,
                          split_manifest: Path, v33_release: str,
                          v33_manifest: Path, workflow_order: Path) -> int:
    """Build the pre-access manifest that is published in the freeze Release."""
    if split_release != "split-lock":
        raise ValueError("v3.4 freeze requires split_release=split-lock")
    if not v33_release.startswith("neural-dev-v3.3-"):
        raise ValueError("v3.4 freeze requires an exact neural-dev-v3.3-* Release")
    if json.loads(v33_manifest.read_text()).get("schema") != "neural-dev-v3.3":
        raise ValueError("v3.3 analysis manifest schema mismatch")
    order = json.loads(workflow_order.read_text())
    if order.get("schema") != "confirm-v3.4-gh-order-v1":
        raise ValueError("v3.4 GitHub workflow order schema mismatch")
    upstream = order.get("immutable_upstream_releases", {})
    if upstream.get("split") != split_release:
        raise ValueError("workflow order split Release mismatch")
    if upstream.get("dev_v33_analysis") != v33_release:
        raise ValueError("workflow order v3.3 Release mismatch")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "schema": "confirm-v3.4-freeze-v1",
        "code_commit": code_commit,
        "prereg_sha256": digest(prereg),
        "requirements_sha256": digest(requirements),
        "workflow_order_sha256": digest(workflow_order),
        "split_release": split_release,
        "split_manifest_sha256": digest(split_manifest),
        "v33_release": v33_release,
        "v33_manifest_sha256": digest(v33_manifest),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def _require_columns(table: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{name} missing columns {missing}")


def validate_confirm_split(confirm: pd.DataFrame,
                           dev: pd.DataFrame | None = None) -> None:
    _require_columns(confirm, IDENTITY_COLUMNS, "CONFIRM split")
    problems = []
    for column, expected in (
            ("ophys_experiment_id", EXPECTED_EXPERIMENTS),
            ("behavior_session_id", EXPECTED_EXPERIMENTS),
            ("mouse_id", EXPECTED_MICE),
            ("ophys_container_id", EXPECTED_CONTAINERS)):
        count = confirm[column].astype(int).nunique()
        if count != expected:
            problems.append(f"{column}:{count}!={expected}")
    if len(confirm) != EXPECTED_EXPERIMENTS:
        problems.append(f"rows:{len(confirm)}!={EXPECTED_EXPERIMENTS}")
    if confirm.session_type.astype(str).str.contains(
            "passive", case=False, regex=False).any():
        problems.append("passive_experiment")
    if dev is not None:
        overlap = {}
        for column in ("ophys_experiment_id", "behavior_session_id",
                       "ophys_container_id", "mouse_id"):
            if column in dev:
                values = set(confirm[column].dropna().astype(int)) & set(
                    dev[column].dropna().astype(int))
                if values:
                    overlap[column] = sorted(values)
        if overlap:
            problems.append(f"DEV_contamination:{overlap}")
    if problems:
        raise ValueError("invalid CONFIRM split: " + "; ".join(problems))


def build_confirm_manifest(confirm: pd.DataFrame, experiments: pd.DataFrame,
                           dev: pd.DataFrame | None = None) -> pd.DataFrame:
    validate_confirm_split(confirm, dev)
    table = (experiments.reset_index()
             if "ophys_experiment_id" not in experiments.columns
             else experiments.copy())
    _require_columns(table, IDENTITY_COLUMNS, "Allen experiment table")
    expected_ids = set(confirm.ophys_experiment_id.astype(int))
    selected = table[
        table.ophys_experiment_id.astype("Int64").isin(expected_ids)].copy()
    if (len(selected) != EXPECTED_EXPERIMENTS or
            set(selected.ophys_experiment_id.astype(int)) != expected_ids):
        found = set(selected.ophys_experiment_id.astype(int))
        raise ValueError(
            f"active CONFIRM mismatch; missing={sorted(expected_ids-found)}, "
            f"extra={sorted(found-expected_ids)}")
    for column in ("behavior_session_id", "ophys_container_id", "mouse_id"):
        expected_map = confirm.set_index("ophys_experiment_id")[column].astype(int)
        observed = selected.set_index("ophys_experiment_id")[column].astype(int)
        if not observed.sort_index().equals(expected_map.sort_index()):
            raise ValueError(f"CONFIRM identity mismatch for {column}")
    selected["role"] = "active"
    keep = [column for column in (
        "ophys_experiment_id", "behavior_session_id", "ophys_session_id",
        "ophys_container_id", "mouse_id", "project_code", "session_type",
        "equipment_name", "imaging_depth", "targeted_structure", "file_id",
        "role") if column in selected]
    result = selected[keep].sort_values(
        ["ophys_container_id", "ophys_experiment_id"]).reset_index(drop=True)
    validate_confirm_manifest(result, dev)
    return result


def validate_confirm_manifest(manifest: pd.DataFrame,
                              dev: pd.DataFrame | None = None) -> None:
    _require_columns(manifest, IDENTITY_COLUMNS | {"role"}, "CONFIRM manifest")
    validate_confirm_split(manifest, dev)
    if not manifest.role.astype(str).eq("active").all():
        raise ValueError("CONFIRM manifest must contain active experiments only")


def make_manifest(confirm_path: Path, dev_path: Path, out: Path,
                  cache_dir: Path) -> int:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    confirm, dev = pd.read_csv(confirm_path), pd.read_csv(dev_path)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    result = build_confirm_manifest(
        confirm, cache.get_ophys_experiment_table(), dev)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"wrote {len(result)} active experiments / "
          f"{result.ophys_container_id.nunique()} containers")
    return 0


def pull_container(manifest_path: Path, container: int, out: Path,
                   cache_dir: Path, retries: int) -> int:
    manifest = pd.read_csv(manifest_path)
    validate_confirm_manifest(manifest)
    selected = manifest[
        manifest.ophys_container_id.astype(int).eq(int(container))].copy()
    if selected.empty:
        raise ValueError(f"container {container} absent from CONFIRM manifest")
    return neural.pull(selected, [int(container)], out, cache_dir, retries,
                       f"_pull-container-{int(container)}.json")


def _validate_behavior(behavior_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = json.loads((behavior_dir / "behavioral-manifest.json").read_text())
    if (metadata.get("schema") != BEHAVIOR_SCHEMA or
            int(metadata.get("n_sessions", -1)) != EXPECTED_EXPERIMENTS or
            int(metadata.get("n_mice", -1)) != EXPECTED_MICE or
            metadata.get("construct_sweep_performed") is not False or
            metadata.get("threshold_sweep_performed") is not False):
        raise ValueError("behavioral label-only manifest is not frozen v3.4")
    labels = pd.read_parquet(behavior_dir / "_trial_labels.parquet")
    _require_columns(labels, {
        "behavior_session_id", "trial_id", "trial_index", "late_hit", "miss",
        "engaged_B", "keep_B", "is_image_novel",
    }, "behavior labels")
    expected = set(manifest.behavior_session_id.astype(int))
    expected_ids_sha = hashlib.sha256(
        "\n".join(map(str, sorted(expected))).encode()).hexdigest()
    if metadata.get("confirm_ids_sha256") != expected_ids_sha:
        raise ValueError("behavioral manifest CONFIRM ID hash mismatch")
    observed = set(labels.behavior_session_id.astype(int))
    if observed != expected:
        raise ValueError(
            f"behavior session mismatch; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}")
    return labels


def materialize_features(source: Path, manifest_path: Path,
                         behavior_dir: Path, out: Path, container: int, *,
                         data_release: str, data_manifest_sha256: str,
                         behavioral_release: str) -> int:
    manifest = pd.read_csv(manifest_path)
    validate_confirm_manifest(manifest)
    selected = manifest[
        manifest.ophys_container_id.astype(int).eq(int(container))].copy()
    if selected.empty:
        raise ValueError(f"container {container} absent from CONFIRM manifest")
    neural.validate_bundles(source, selected)
    source_failures = neural.appendix_a_failures(source, selected)
    if source_failures:
        raise ValueError(f"source bundle validation failed: {source_failures}")
    labels = _validate_behavior(behavior_dir, manifest)
    out.mkdir(parents=True, exist_ok=True)
    materialized, skipped = [], []
    for row in selected.itertuples(index=False):
        oeid = int(row.ophys_experiment_id)
        if neural.feature_cache_complete(out, oeid):
            skipped.append(oeid)
            continue
        for path in neural.feature_cache_paths(out, oeid):
            path.unlink(missing_ok=True)
        neural._write_feature_cache_experiment(
            source, out, row, labels, data_release=data_release,
            data_manifest_sha256=data_manifest_sha256,
            behavioral_release=behavioral_release,
            feature_schema=FEATURE_SCHEMA)
        materialized.append(oeid)
    provenance = {
        "neural_data_release": data_release,
        "data_manifest_sha256": data_manifest_sha256,
        "behavioral_release": behavioral_release,
    }
    failures = neural.feature_cache_failures(
        out, selected, feature_schema=FEATURE_SCHEMA,
        expected_provenance=provenance)
    report = {
        "schema": FEATURE_SCHEMA, "container_id": int(container),
        "active_experiments": sorted(
            selected.ophys_experiment_id.astype(int).tolist()),
        "materialized": materialized, "skipped": skipped,
        "failures": failures, "neural_data_release": data_release,
        "data_manifest_sha256": data_manifest_sha256,
        "behavioral_release": behavioral_release,
        "allen_nwb_download": False,
    }
    (out / f"_features-container-{int(container)}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    shutil.rmtree(out / ".staging", ignore_errors=True)
    if failures:
        raise ValueError(f"feature cache validation failed: {failures}")
    return 0


def feature_failures(root: Path, manifest: pd.DataFrame, *,
                     expected_provenance: dict | None = None) -> list[dict]:
    failures = []
    try:
        validate_confirm_manifest(manifest)
    except ValueError as exc:
        failures.append({"scope": "manifest", "problems": [str(exc)]})
        return failures
    failures.extend(neural.feature_cache_failures(
        root, manifest, feature_schema=FEATURE_SCHEMA,
        expected_provenance=expected_provenance))
    return failures


def verify_features(root: Path, manifest_path: Path,
                    report_path: Path | None = None, *, data_release: str,
                    data_manifest_sha256: str,
                    behavioral_release: str) -> int:
    manifest = pd.read_csv(manifest_path)
    provenance = {
        "neural_data_release": data_release,
        "data_manifest_sha256": data_manifest_sha256,
        "behavioral_release": behavioral_release,
    }
    failures = feature_failures(
        root, manifest, expected_provenance=provenance)
    report = {
        "schema": FEATURE_SCHEMA,
        "n_experiments": int(len(manifest)),
        "n_containers": int(manifest.ophys_container_id.nunique())
        if "ophys_container_id" in manifest else 0,
        "complete": not failures, "failures": failures,
        "allen_nwb_download": False,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


def _feature_sessions(root: Path, manifest_path: Path, *, data_release: str,
                      data_manifest_sha256: str,
                      behavioral_release: str) -> tuple[list[dict], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    validate_confirm_manifest(manifest)
    failures = neural.feature_cache_failures(
        root, manifest, feature_schema=FEATURE_SCHEMA,
        expected_provenance={
            "neural_data_release": data_release,
            "data_manifest_sha256": data_manifest_sha256,
            "behavioral_release": behavioral_release,
        })
    if failures:
        raise ValueError(f"feature cache validation failed: {failures}")
    sessions = []
    for row in manifest.itertuples(index=False):
        oeid, bsid = int(row.ophys_experiment_id), int(row.behavior_session_id)
        labels = pd.read_parquet(root / f"{oeid}.labels.parquet")
        q2 = pd.read_parquet(root / f"{oeid}.q2.parquet")
        with h5py.File(root / f"{oeid}.features.h5", "r") as h5:
            trial_ids = np.asarray(h5["trial_id"][:], np.int64)
            arrays = {name: np.asarray(h5[name][:], np.float32)
                      for name in neural.FEATURE_DATASETS}
        if (labels.trial_id.astype(int).tolist() != trial_ids.tolist() or
                q2.trial_id.astype(int).tolist() != trial_ids.tolist()):
            raise ValueError(f"feature/label/Q2 trial alignment failed for {oeid}")
        primary = (labels.engaged_B.fillna(False).astype(bool) &
                   labels.keep_B.fillna(False).astype(bool))
        miss_b = int((primary & labels.miss.fillna(False).astype(bool)).sum())
        late_b = int((primary & labels.late_hit.fillna(False).astype(bool)).sum())
        novelty = labels.is_image_novel.dropna().astype(bool).unique()
        sessions.append({
            "arrays": arrays, "labels": labels.reset_index(drop=True),
            "q2": q2.reset_index(drop=True),
            "meta": {
                "ophys_experiment_id": oeid,
                "behavior_session_id": bsid,
                "mouse_id": int(row.mouse_id),
                "project_code": str(row.project_code),
                "novel": bool(novelty[0]) if len(novelty) == 1 else None,
                "miss_B": miss_b, "late_hit_B": late_b,
                "behavioral_eligible": bool(miss_b >= 20 and late_b >= 20),
            },
        })
    return sessions, manifest


def primary_decision(interval: dict, n_mice: int, *,
                     integrity: bool = True) -> dict:
    if not integrity:
        return {"status": "pipeline_failure", "reason": "integrity_failure"}
    if int(n_mice) < REQUIRED_MICE:
        return {
            "status": "nonestimable_coverage",
            "reason": f"estimable_mice_{int(n_mice)}_below_{REQUIRED_MICE}",
        }
    low = interval.get("low")
    if low is None or not np.isfinite(float(low)):
        return {"status": "nonestimable_interval", "reason": "BCa_interval_nonfinite"}
    if float(low) > SESOI:
        return {"status": "confirmatory_supported", "reason": "ci_low_above_SESOI"}
    return {
        "status": "confirmatory_not_supported",
        "reason": "ci_low_not_strictly_above_SESOI",
    }


def bca_interval(values: np.ndarray, *, seed: int) -> dict:
    """Normalize degenerate SciPy BCa bounds to a typed nonestimable interval."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    try:
        result = neural._bca_mean(values, seed=seed)
    except ValueError:
        result = {
            "mean": float(values.mean()) if len(values) else None,
            "low": None, "high": None, "n_mice": int(len(values)),
        }
    for key in ("mean", "low", "high"):
        value = result.get(key)
        if value is not None and not np.isfinite(float(value)):
            result[key] = None
    return result


def complete_q1_result(metrics: dict, error: str | None) -> tuple[dict, str | None]:
    """Enforce the preregistered all-ten-seeds estimability rule."""
    if error is not None:
        return {}, error
    if int(metrics.get("n_seeds", 0)) != neural.N_SEEDS:
        return {}, "partial_seed_completion"
    return metrics, None


def _assert_exact_table(computed: pd.DataFrame, expected: pd.DataFrame,
                        keys: list[str], name: str) -> None:
    missing = sorted(set(computed.columns) - set(expected.columns))
    if missing:
        raise ValueError(f"{name} expected table missing columns {missing}")
    left = computed.sort_values(keys).reset_index(drop=True)
    right = expected[computed.columns].sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError(f"v3.4 Q1 drift in {name}: {exc}") from exc


def verify_dev_q1_equivalence(features: Path, manifest_path: Path,
                              expected_dir: Path, report_path: Path) -> int:
    """Re-run the v3.4 Q1 kernel on DEV and match the v3.3 tables exactly."""
    sessions, _ = v33._feature_sessions(features, manifest_path)
    rows, folds = [], []
    for item in sessions:
        meta = item["meta"]
        if meta["late_hit_B"] < 20 or meta["miss_B"] < 20:
            continue
        metrics, session_folds, error = q1_kernel(item)
        metrics, error = complete_q1_result(metrics, error)
        folds.extend(session_folds)
        rows.append({
            **meta, **metrics, "K": neural.PRIMARY_K, "C": neural.FROZEN_C50,
            "decoder_estimability": error or "estimable",
        })
    computed_sessions = pd.DataFrame(rows)
    valid = (computed_sessions[
        computed_sessions.decoder_estimability.eq("estimable") &
        np.isfinite(computed_sessions.auc)]
        if "auc" in computed_sessions else computed_sessions.iloc[0:0])
    computed_mice, _, _ = neural._mouse_summary(valid)
    computed_folds = pd.DataFrame(folds)
    expected_sessions = pd.read_parquet(expected_dir / "q1_sessions.parquet")
    expected_mice = pd.read_parquet(expected_dir / "q1_mice.parquet")
    expected_folds = pd.read_parquet(
        expected_dir / "q1_fold_diagnostics.parquet")
    selected_sessions = set(computed_sessions.behavior_session_id.astype(int))
    expected_folds = expected_folds[
        expected_folds.behavior_session_id.astype(int).isin(selected_sessions)]
    _assert_exact_table(
        computed_sessions, expected_sessions,
        ["ophys_experiment_id"], "session results")
    _assert_exact_table(computed_mice, expected_mice, ["mouse_id"], "mouse results")
    _assert_exact_table(
        computed_folds, expected_folds,
        ["ophys_experiment_id", "seed", "fold"], "fold results")
    report = {
        "schema": "neural-confirm-v3.4-dev-equivalence-v1",
        "exact": True, "kernel_is_v33": q1_kernel is v33._q1_session,
        "n_sessions": int(len(computed_sessions)),
        "n_mice": int(len(computed_mice)),
        "n_fold_rows": int(len(computed_folds)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _q2_mouse_summary(q2_sessions: pd.DataFrame) -> pd.DataFrame:
    if q2_sessions.empty or "delta_log_loss" not in q2_sessions:
        return pd.DataFrame(columns=["mouse_id"])
    valid = q2_sessions[np.isfinite(q2_sessions.delta_log_loss)].copy()
    excluded = {
        "ophys_experiment_id", "behavior_session_id", "mouse_id",
        "project_code", "novel", "miss_B", "late_hit_B",
        "behavioral_eligible", "q2_estimability",
    }
    metrics = [column for column in valid.columns
               if column not in excluded and
               pd.api.types.is_numeric_dtype(valid[column])]
    rows = []
    for mouse, group in valid.groupby("mouse_id"):
        weights = np.maximum(group.miss_B.to_numpy(float), 1)
        row = {"mouse_id": mouse}
        for column in metrics:
            values = group[column].to_numpy(float)
            finite = np.isfinite(values)
            row[column] = (float(np.average(values[finite], weights=weights[finite]))
                           if finite.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def scan(features: Path, manifest_path: Path, out: Path, *,
         feature_release: str, feature_manifest_sha256: str) -> int:
    sessions, manifest = _feature_sessions(
        features, manifest_path, data_release=feature_release,
        data_manifest_sha256=feature_manifest_sha256,
        behavioral_release=feature_release)
    out.mkdir(parents=True, exist_ok=True)

    q1_rows, q1_folds, coverage_rows = [], [], []
    for item in sessions:
        meta = item["meta"]
        if not meta["behavioral_eligible"]:
            error = "behavioral_ineligible"
            metrics, folds = {}, []
        else:
            metrics, folds, error = q1_kernel(item)
            metrics, error = complete_q1_result(metrics, error)
        q1_folds.extend(folds)
        q1_rows.append({
            **meta, **metrics, "K": neural.PRIMARY_K, "C": neural.FROZEN_C50,
            "decoder_estimability": error or "estimable",
        })
        coverage_rows.append({
            **meta, "q1_estimability": error or "estimable",
        })
    q1_sessions = pd.DataFrame(q1_rows)
    valid_q1 = (q1_sessions[
        q1_sessions.decoder_estimability.eq("estimable") &
        np.isfinite(q1_sessions.auc)].copy()
        if "auc" in q1_sessions else q1_sessions.iloc[0:0].copy())
    q1_mice, _, _ = neural._mouse_summary(valid_q1)
    q1_interval = bca_interval(
        q1_mice.auc.to_numpy(), seed=Q1_BOOTSTRAP_SEED)
    decision = primary_decision(q1_interval, len(q1_mice))

    q2_rows, q2_selection, q2_folds = [], [], []
    q2_estimability = {
        int(item["meta"]["behavior_session_id"]): "behavioral_ineligible"
        for item in sessions if not item["meta"]["behavioral_eligible"]
    }
    for item in sessions:
        meta = item["meta"]
        if not meta["behavioral_eligible"]:
            continue
        metrics, selection, folds, error = v33._q2_session(item)
        q2_selection.extend(selection)
        q2_folds.extend(folds)
        q2_rows.append({
            **meta, **metrics, "q2_estimability": error or "estimable",
        })
        q2_estimability[int(meta["behavior_session_id"])] = error or "estimable"
    q2_sessions = pd.DataFrame(q2_rows)
    q2_mice = _q2_mouse_summary(q2_sessions)
    q2_coverage = int(len(q2_mice))
    q2_interval_metrics = (
        "delta_log_loss", "delta_brier", "delta_auc",
        "raw_delta_log_loss", "raw_delta_auc",
        "v32_C1_delta_log_loss", "v32_C1_delta_auc",
    )
    q2_intervals = {
        metric: (bca_interval(
            q2_mice[metric].to_numpy() if metric in q2_mice else np.array([]),
            seed=Q2_BOOTSTRAP_SEED)
            if q2_coverage >= REQUIRED_MICE else
            {"mean": None, "low": None, "high": None,
             "n_mice": q2_coverage})
        for metric in q2_interval_metrics
    }
    q2_interval = q2_intervals["delta_log_loss"]
    if q2_coverage < REQUIRED_MICE:
        q2_status = "secondary_nonestimable_coverage"
    elif any(q2_intervals[metric].get("low") is None
             for metric in ("delta_log_loss", "delta_brier", "delta_auc")):
        q2_status = "secondary_nonestimable_interval"
    else:
        q2_status = "descriptive_estimable"

    q1_sessions.to_parquet(out / "q1_sessions.parquet", index=False)
    q1_mice.to_parquet(out / "q1_mice.parquet", index=False)
    pd.DataFrame(q1_folds).to_parquet(
        out / "q1_fold_diagnostics.parquet", index=False)
    q2_sessions.to_parquet(out / "q2_sessions.parquet", index=False)
    q2_mice.to_parquet(out / "q2_mice.parquet", index=False)
    pd.DataFrame(q2_selection).to_parquet(
        out / "q2_C_selection.parquet", index=False)
    pd.DataFrame(q2_folds).to_parquet(
        out / "q2_fold_diagnostics.parquet", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage["q2_estimability"] = coverage.behavior_session_id.astype(int).map(
        q2_estimability)
    coverage.to_parquet(
        out / "coverage_failures.parquet", index=False)

    result = {
        "schema": RESULT_SCHEMA,
        "confirm_data_accessed": True,
        "population": {
            "expected_experiments": EXPECTED_EXPERIMENTS,
            "expected_mice": EXPECTED_MICE,
            "expected_containers": EXPECTED_CONTAINERS,
            "active_only": True, "observed_experiments": int(len(manifest)),
        },
        "feature_source": {
            "feature_release": feature_release,
            "feature_manifest_sha256": feature_manifest_sha256,
            "schema": FEATURE_SCHEMA, "cache_only": True,
            "n_active_experiments": EXPECTED_EXPERIMENTS,
            "allen_nwb_download": False,
        },
        "primary": {
            "question": "Q1", "target": "late-hit-vs-miss",
            "signal": "events", "window": [neural.FIT_START, neural.FIT_END],
            "K": neural.PRIMARY_K, "C": neural.FROZEN_C50,
            "cell_seeds": neural.N_SEEDS, "contiguous_folds": neural.N_BLOCKS,
            "purge_raw_trials": neural.GAP_RAW_TRIALS,
            "session_estimator": "mean_of_10_seed_pooled_oof_auc",
            "mouse_weight": "engaged_miss", "mice_equal_weight": True,
            "required_mice": REQUIRED_MICE, "SESOI": SESOI,
            "n_estimable_sessions": int(len(valid_q1)),
            "n_estimable_mice": int(len(q1_mice)),
            "SESOI_source": "external_fixed_pre_access",
            "bootstrap": {"method": "BCa", "sides": 2, "level": 0.95,
                          "resamples": 2000, "seed": Q1_BOOTSTRAP_SEED},
            "mouse_interval": q1_interval, "decision": decision,
            "diagnostics_can_change_decision": False,
        },
        "secondary": {
            "question": "Q2", "role": "descriptive_secondary",
            "C_grid": list(v33.C_GRID),
            "selection": "nested_one_SE_smallest_C_separate_M0_M1",
            "calibration": "training_only_inner_OOF_sigmoid",
            "required_mice_for_group_summary": REQUIRED_MICE,
            "n_estimable_sessions": int(
                q2_sessions.q2_estimability.eq("estimable").sum())
            if "q2_estimability" in q2_sessions else 0,
            "n_mice": q2_coverage, "status": q2_status,
            "delta_log_loss_mouse_interval": q2_interval,
            "mouse_intervals": q2_intervals,
            "bootstrap": {"method": "BCa", "sides": 2, "level": 0.95,
                          "resamples": 2000, "seed": Q2_BOOTSTRAP_SEED},
            "SESOI": None, "changes_primary_decision": False,
        },
        "outcome": decision["status"],
        "outcome_taxonomy": [
            "pipeline_failure", "nonestimable_coverage",
            "nonestimable_interval", "confirmatory_not_supported",
            "confirmatory_supported",
        ],
        "integrity": {
            "exact_feature_cache": True, "exact_active_manifest": True,
            "DEV_overlap": False, "state_analysis_run": False,
            "threshold_sweep_run": False, "v4_run": False,
        },
    }
    (out / "analysis-manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--confirm", type=Path, required=True)
    manifest.add_argument("--dev", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--cache", type=Path, default=Path("/tmp/allen-meta-v34"))
    pull = sub.add_parser("pull")
    pull.add_argument("--manifest", type=Path, required=True)
    pull.add_argument("--container", type=int, required=True)
    pull.add_argument("--out", type=Path, required=True)
    pull.add_argument("--cache", type=Path, default=Path("/tmp/allen-neural-v34"))
    pull.add_argument("--retries", type=int, default=3)
    features = sub.add_parser("features")
    features.add_argument("--manifest", type=Path, required=True)
    features.add_argument("--neural", type=Path, required=True)
    features.add_argument("--behavior", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--container", type=int, required=True)
    features.add_argument("--data-release", required=True)
    features.add_argument("--data-manifest-sha256", required=True)
    features.add_argument("--behavioral-release", required=True)
    verify = sub.add_parser("feature-verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--features", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    verify.add_argument("--data-release", required=True)
    verify.add_argument("--data-manifest-sha256", required=True)
    verify.add_argument("--behavioral-release", required=True)
    analyze = sub.add_parser("scan")
    analyze.add_argument("--features", type=Path, required=True)
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--feature-release", required=True)
    analyze.add_argument("--feature-manifest-sha256", required=True)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--path", type=Path, required=True)
    receipt.add_argument("--run-id", type=int, required=True)
    receipt.add_argument("--commit", required=True)
    receipt.add_argument("--freeze-release", required=True)
    receipt.add_argument("--freeze-sha256", required=True)
    receipt.add_argument("--split-manifest-sha256", required=True)
    equivalence = sub.add_parser("dev-q1-equivalence")
    equivalence.add_argument("--features", type=Path, required=True)
    equivalence.add_argument("--manifest", type=Path, required=True)
    equivalence.add_argument("--expected", type=Path, required=True)
    equivalence.add_argument("--report", type=Path, required=True)
    freeze = sub.add_parser("freeze-manifest")
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--code-commit", required=True)
    freeze.add_argument("--prereg", type=Path, default=Path("docs/prereg_v3.4.md"))
    freeze.add_argument("--requirements", type=Path,
                        default=Path("requirements-pipeline.txt"))
    freeze.add_argument("--workflow-order", type=Path,
                        default=Path("docs/confirm_v3.4_gh_order.json"))
    freeze.add_argument("--split-release", default="split-lock")
    freeze.add_argument("--split-manifest", type=Path, required=True)
    freeze.add_argument("--v33-release", required=True)
    freeze.add_argument("--v33-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "manifest":
        return make_manifest(args.confirm, args.dev, args.out, args.cache)
    if args.cmd == "pull":
        return pull_container(
            args.manifest, args.container, args.out, args.cache, args.retries)
    if args.cmd == "features":
        return materialize_features(
            args.neural, args.manifest, args.behavior, args.out, args.container,
            data_release=args.data_release,
            data_manifest_sha256=args.data_manifest_sha256,
            behavioral_release=args.behavioral_release)
    if args.cmd == "feature-verify":
        return verify_features(
            args.features, args.manifest, args.report,
            data_release=args.data_release,
            data_manifest_sha256=args.data_manifest_sha256,
            behavioral_release=args.behavioral_release)
    if args.cmd == "receipt":
        print(access_receipt(
            args.path, run_id=args.run_id, commit=args.commit,
            freeze_release=args.freeze_release,
            freeze_sha256=args.freeze_sha256,
            split_manifest_sha256=args.split_manifest_sha256))
        return 0
    if args.cmd == "dev-q1-equivalence":
        return verify_dev_q1_equivalence(
            args.features, args.manifest, args.expected, args.report)
    if args.cmd == "freeze-manifest":
        return write_freeze_manifest(
            args.out, code_commit=args.code_commit, prereg=args.prereg,
            requirements=args.requirements, split_release=args.split_release,
            split_manifest=args.split_manifest, v33_release=args.v33_release,
            v33_manifest=args.v33_manifest, workflow_order=args.workflow_order)
    return scan(
        args.features, args.manifest, args.out,
        feature_release=args.feature_release,
        feature_manifest_sha256=args.feature_manifest_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

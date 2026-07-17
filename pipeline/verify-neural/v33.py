#!/usr/bin/env python3
"""Cache-only DEV v3.3 analysis.

This entrypoint consumes the immutable active-session feature cache.  It cannot
construct an Allen cache, download an NWB, or read a neural bundle.  Q1 keeps
the frozen pooled-OOF estimator; the state anchor uses within-test-fold AUCs;
Q2 selects and calibrates nuisance models strictly inside each outer training
partition.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import neural


C_GRID = tuple(float(10.0**power) for power in range(-4, 3))
MIN_STATE_FOLDS = 3


def _feature_sessions(root: Path, manifest_path: Path) -> tuple[list[dict], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    if (len(manifest) != 70 or manifest.ophys_experiment_id.astype(int).nunique() != 70 or
            manifest.ophys_container_id.astype(int).nunique() != 10 or
            int(manifest.role.eq("active").sum()) != 50 or
            int(manifest.role.eq("passive").sum()) != 20):
        raise ValueError("v3.3 requires the exact 50-active + 20-passive DEV manifest")
    failures = neural.feature_cache_failures(root, manifest)
    if failures:
        raise ValueError(f"feature cache validation failed: {failures}")
    sessions = []
    for row in manifest.loc[manifest.role.eq("active")].itertuples(index=False):
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
            "arrays": arrays,
            "labels": labels.reset_index(drop=True),
            "q2": q2.reset_index(drop=True),
            "meta": {
                "ophys_experiment_id": oeid,
                "behavior_session_id": bsid,
                "mouse_id": int(row.mouse_id),
                "project_code": str(row.project_code),
                "novel": bool(novelty[0]) if len(novelty) == 1 else None,
                "miss_B": miss_b,
                "late_hit_B": late_b,
                "behavioral_eligible": bool(miss_b >= 20 and late_b >= 20),
            },
        })
    return sessions, manifest


def _fit_decoder(X: np.ndarray, y: np.ndarray, train: np.ndarray, *,
                 C: float, seed: int, class_weight):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[train])
    model = LogisticRegression(
        C=C, class_weight=class_weight, solver="liblinear",
        random_state=seed, max_iter=2000)
    model.fit(scaler.transform(X[train]), y[train])
    return scaler, model


def _blocked_auc(X: np.ndarray, y: np.ndarray, raw: np.ndarray, *,
                 C: float, seed: int, class_weight,
                 minimum_estimable_folds: int = 1) -> tuple[dict, list[dict], str | None]:
    from sklearn.metrics import roc_auc_score
    if len(raw) < neural.N_BLOCKS:
        return {}, [], "insufficient_trials_for_folds"
    scores = np.full(len(y), np.nan)
    fold_rows = []
    estimable = []
    for fold, (train, test) in enumerate(neural._folds(raw), start=1):
        train_pos, test_pos = int(y[train].sum()), int(y[test].sum())
        train_neg, test_neg = len(train)-train_pos, len(test)-test_pos
        base = {
            "fold": fold,
            "n_train": int(len(train)), "n_test": int(len(test)),
            "train_positive": train_pos, "train_negative": train_neg,
            "test_positive": test_pos, "test_negative": test_neg,
            "train_prevalence": float(y[train].mean()) if len(train) else np.nan,
            "test_prevalence": float(y[test].mean()) if len(test) else np.nan,
        }
        if len(np.unique(y[train])) < 2:
            fold_rows.append({**base, "auc": np.nan, "comparable_pairs": 0,
                              "score_mean": np.nan, "score_sd": np.nan,
                              "estimability": "training_class_missing"})
            return {}, fold_rows, "temporal_support_nonestimable"
        scaler, model = _fit_decoder(
            X, y, train, C=C, seed=seed, class_weight=class_weight)
        test_score = model.decision_function(scaler.transform(X[test]))
        scores[test] = test_score
        pairs = int(test_pos * test_neg)
        if pairs:
            auc = float(roc_auc_score(y[test], test_score))
            estimable.append((auc, pairs))
            reason = "estimable"
        else:
            auc, reason = np.nan, "test_class_missing"
        fold_rows.append({
            **base, "auc": auc, "comparable_pairs": pairs,
            "score_mean": float(np.mean(test_score)),
            "score_sd": float(np.std(test_score, ddof=0)),
            "estimability": reason,
        })
    if not np.isfinite(scores).all() or len(np.unique(y)) < 2:
        return {}, fold_rows, "score_nonestimable"
    pooled = float(roc_auc_score(y, scores))
    if len(estimable) < minimum_estimable_folds:
        return {"pooled_auc": pooled}, fold_rows, "conditional_fold_support"
    weight = np.asarray([item[1] for item in estimable], float)
    values = np.asarray([item[0] for item in estimable], float)
    return {
        "pooled_auc": pooled,
        "conditional_auc": float(np.average(values, weights=weight)),
        "mean_fold_auc": float(values.mean()),
        "n_estimable_folds": int(len(estimable)),
        "comparable_pairs": int(weight.sum()),
    }, fold_rows, None


def _random_auc(X: np.ndarray, y: np.ndarray, *, C: float, seed: int,
                class_weight) -> tuple[float, str | None]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    scores = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2 or int(np.bincount(y).min()) < neural.N_BLOCKS:
        return np.nan, "random_class_nonestimable"
    for train, test in StratifiedKFold(
            neural.N_BLOCKS, shuffle=True, random_state=seed).split(X, y):
        scaler, model = _fit_decoder(
            X, y, train, C=C, seed=seed, class_weight=class_weight)
        scores[test] = model.decision_function(scaler.transform(X[test]))
    return float(roc_auc_score(y, scores)), None


def _q1_session(item: dict) -> tuple[dict, list[dict], str | None]:
    labels, meta = item["labels"], item["meta"]
    mask = (labels.engaged_B.fillna(False).astype(bool) &
            labels.keep_B.fillna(False).astype(bool) &
            (labels.late_hit.fillna(False).astype(bool) |
             labels.miss.fillna(False).astype(bool)))
    y = labels.loc[mask, "late_hit"].astype(int).to_numpy()
    raw = labels.loc[mask, "trial_index"].astype(int).to_numpy()
    rows, folds, errors = [], [], []
    for seed in range(neural.N_SEEDS):
        X = neural._subset_cells(
            item["arrays"]["events_baselined_post"][mask.to_numpy()],
            neural.PRIMARY_K, seed, meta["ophys_experiment_id"])
        if X is None:
            return {}, folds, "low_cells"
        metrics, seed_folds, error = _blocked_auc(
            X, y, raw, C=neural.FROZEN_C50, seed=seed,
            class_weight="balanced")
        for fold in seed_folds:
            folds.append({**meta, "seed": seed, "analysis": "q1", **fold})
        if error:
            errors.append(error)
        else:
            random_auc, random_error = _random_auc(
                X, y, C=neural.FROZEN_C50, seed=seed,
                class_weight="balanced")
            dff = neural._subset_cells(
                item["arrays"]["dff_baselined_post"][mask.to_numpy()],
                neural.PRIMARY_K, seed, meta["ophys_experiment_id"])
            dff_metrics, _, dff_error = _blocked_auc(
                dff, y, raw, C=neural.FROZEN_C50, seed=seed,
                class_weight="balanced")
            rows.append({
                **metrics,
                "random_auc": random_auc if not random_error else np.nan,
                "dff_pooled_auc": (dff_metrics.get("pooled_auc", np.nan)
                                   if not dff_error else np.nan),
            })
    if not rows:
        return {}, folds, ";".join(sorted(set(errors))) or "q1_nonestimable"
    frame = pd.DataFrame(rows)
    return {
        "auc": float(frame.pooled_auc.mean()),
        "conditional_auc": float(frame.conditional_auc.mean()),
        "mean_fold_auc": float(frame.mean_fold_auc.mean()),
        "random_auc": float(frame.random_auc.mean()),
        "dff_auc": float(frame.dff_pooled_auc.mean()),
        "n_seeds": int(len(frame)),
        "n_trials": int(len(y)),
    }, folds, None


def _balanced_state_indices(labels: pd.DataFrame, *, guarded: bool,
                            seed: int, oeid: int) -> tuple[np.ndarray, int]:
    eligible = ((labels.late_hit.fillna(False).astype(bool) |
                 labels.miss.fillna(False).astype(bool)) &
                ~labels.first_ten.fillna(False).astype(bool))
    if guarded:
        eligible &= labels.keep_B.fillna(False).astype(bool)
    candidates = labels.loc[eligible]
    rng = np.random.default_rng(seed + oeid)
    selected = []
    limiting = 0
    for outcome in ("late_hit", "miss"):
        groups = [
            candidates.index[
                candidates[outcome].fillna(False).astype(bool) &
                candidates.engaged_B.fillna(False).astype(bool).eq(state)
            ].to_numpy()
            for state in (False, True)
        ]
        n = min(map(len, groups))
        limiting += n
        if n:
            selected.extend(rng.choice(group, n, replace=False) for group in groups)
    flat = np.concatenate(selected).astype(int) if selected else np.array([], dtype=int)
    return flat, int(limiting)


def _sigmoid_fit(logits: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    if (not np.isfinite(logits).all() or len(np.unique(y)) < 2 or
            float(np.ptp(logits)) == 0):
        raise ValueError("calibrator_nonestimable")
    model = LogisticRegression(
        C=1e6, class_weight=None, solver="liblinear", max_iter=2000)
    model.fit(np.asarray(logits, float).reshape(-1, 1), y)
    return model


def _sigmoid_apply(model, logits: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(
        np.asarray(logits, float).reshape(-1, 1))[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("calibrator_probability_nonfinite")
    return np.clip(probability, 1e-6, 1-1e-6)


def _state_seed(X: np.ndarray, y: np.ndarray, raw: np.ndarray, *,
                seed: int) -> tuple[dict, list[dict], str | None]:
    from sklearn.metrics import log_loss, roc_auc_score
    if len(raw) < neural.N_BLOCKS:
        return {}, [], "state_insufficient_trials_for_folds"
    outer_logits = np.full(len(y), np.nan)
    raw_probability = np.full(len(y), np.nan)
    calibrated = np.full(len(y), np.nan)
    training_null = np.full(len(y), np.nan)
    folds, estimable = [], []
    for fold, (outer_train, outer_test) in enumerate(neural._folds(raw), start=1):
        train_y = y[outer_train]
        train_pos, test_pos = int(train_y.sum()), int(y[outer_test].sum())
        train_neg, test_neg = len(train_y)-train_pos, len(outer_test)-test_pos
        base = {
            "fold": fold, "n_train": int(len(outer_train)),
            "n_test": int(len(outer_test)),
            "train_positive": train_pos, "train_negative": train_neg,
            "test_positive": test_pos, "test_negative": test_neg,
            "train_prevalence": float(train_y.mean()),
            "test_prevalence": float(y[outer_test].mean()),
        }
        if len(np.unique(train_y)) < 2:
            folds.append({**base, "auc": np.nan, "comparable_pairs": 0,
                          "score_mean": np.nan, "score_sd": np.nan,
                          "estimability": "training_class_missing"})
            return {}, folds, "state_outer_training_class_missing"
        inner_logits = np.full(len(outer_train), np.nan)
        inner_raw = raw[outer_train]
        for inner_train_local, inner_test_local in neural._folds(
                inner_raw, n_blocks=4):
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            if len(np.unique(y[inner_train])) < 2:
                return {}, folds, "state_inner_training_class_missing"
            scaler, model = _fit_decoder(
                X, y, inner_train, C=neural.FROZEN_C50,
                seed=seed, class_weight=None)
            inner_logits[inner_test_local] = model.decision_function(
                scaler.transform(X[inner_test]))
        if not np.isfinite(inner_logits).all():
            return {}, folds, "state_inner_score_incomplete"
        try:
            calibrator = _sigmoid_fit(inner_logits, train_y)
        except ValueError as exc:
            return {}, folds, f"state_{exc}"
        scaler, model = _fit_decoder(
            X, y, outer_train, C=neural.FROZEN_C50,
            seed=seed, class_weight=None)
        logits = model.decision_function(scaler.transform(X[outer_test]))
        outer_logits[outer_test] = logits
        raw_probability[outer_test] = np.clip(
            model.predict_proba(scaler.transform(X[outer_test]))[:, 1],
            1e-6, 1-1e-6)
        calibrated[outer_test] = _sigmoid_apply(calibrator, logits)
        training_null[outer_test] = np.clip(train_y.mean(), 1e-6, 1-1e-6)
        pairs = int(test_pos * test_neg)
        if pairs:
            auc = float(roc_auc_score(y[outer_test], logits))
            estimable.append((auc, pairs))
            reason = "estimable"
        else:
            auc, reason = np.nan, "test_class_missing"
        folds.append({
            **base, "auc": auc, "comparable_pairs": pairs,
            "score_mean": float(np.mean(logits)),
            "score_sd": float(np.std(logits, ddof=0)),
            "estimability": reason,
        })
    if not (np.isfinite(outer_logits).all() and np.isfinite(calibrated).all() and
            np.isfinite(raw_probability).all()):
        return {}, folds, "state_outer_score_incomplete"
    if len(estimable) < MIN_STATE_FOLDS:
        return {}, folds, "state_conditional_fold_support"
    pairs = np.asarray([value[1] for value in estimable], float)
    aucs = np.asarray([value[0] for value in estimable], float)
    fixed_null_loss = float(log_loss(y, np.full(len(y), .5), labels=[0, 1]))
    calibrated_loss = float(log_loss(y, calibrated, labels=[0, 1]))
    return {
        "conditional_auc": float(np.average(aucs, weights=pairs)),
        "mean_fold_auc": float(aucs.mean()),
        "pooled_auc_v32": float(roc_auc_score(y, outer_logits)),
        "n_estimable_folds": int(len(estimable)),
        "comparable_pairs": int(pairs.sum()),
        "calibrated_log_loss": calibrated_loss,
        "raw_log_loss": float(log_loss(y, raw_probability, labels=[0, 1])),
        "fixed_half_null_log_loss": fixed_null_loss,
        "training_prevalence_null_log_loss": float(
            log_loss(y, training_null, labels=[0, 1])),
        "calibrated_logloss_gain": fixed_null_loss-calibrated_loss,
    }, folds, None


def _state_anchor(
        sessions: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    session_rows, fold_rows, representation_rows = [], [], []
    for item in sessions:
        labels, meta = item["labels"], item["meta"]
        for guarded in (True, False):
            seed_rows, errors, limiting = [], [], []
            for seed in range(neural.N_SEEDS):
                indices, limit = _balanced_state_indices(
                    labels, guarded=guarded, seed=seed,
                    oeid=meta["ophys_experiment_id"])
                limiting.append(limit)
                if not len(indices):
                    errors.append("state_balance_nonestimable")
                    continue
                y = labels.loc[indices, "engaged_B"].astype(int).to_numpy()
                raw = labels.loc[indices, "trial_index"].astype(int).to_numpy()
                X = neural._subset_cells(
                    item["arrays"]["events_unbaselined_pre"][indices],
                    neural.PRIMARY_K, seed, meta["ophys_experiment_id"])
                if X is None:
                    errors.append("low_cells")
                    continue
                metrics, folds, error = _state_seed(X, y, raw, seed=seed)
                for fold in folds:
                    fold_rows.append({
                        **meta, "guarded": guarded, "seed": seed, **fold})
                if error:
                    errors.append(error)
                else:
                    random_auc, random_error = _random_auc(
                        X, y, C=neural.FROZEN_C50, seed=seed,
                        class_weight=None)
                    shifted = np.roll(y, 1 + seed % max(1, len(y)-1))
                    shift_metrics, _, shift_error = _blocked_auc(
                        X, shifted, raw, C=neural.FROZEN_C50, seed=seed,
                        class_weight=None, minimum_estimable_folds=MIN_STATE_FOLDS)
                    seed_rows.append({
                        **metrics,
                        "random_auc": random_auc if not random_error else np.nan,
                        "circular_shift_auc": (
                            shift_metrics.get("conditional_auc", np.nan)
                            if not shift_error else np.nan),
                    })
                    if guarded:
                        for representation in (
                                "events_unbaselined_post",
                                "events_baselined_post",
                                "events_baselined_full_pre"):
                            diagnostic_X = neural._subset_cells(
                                item["arrays"][representation][indices],
                                neural.PRIMARY_K, seed,
                                meta["ophys_experiment_id"])
                            diagnostic, _, diagnostic_error = _blocked_auc(
                                diagnostic_X, y, raw, C=neural.FROZEN_C50,
                                seed=seed, class_weight=None,
                                minimum_estimable_folds=MIN_STATE_FOLDS)
                            representation_rows.append({
                                **meta, "seed": seed,
                                "representation": representation,
                                "auc_state_conditional": (
                                    diagnostic.get("conditional_auc", np.nan)
                                    if not diagnostic_error else np.nan),
                                "estimability": diagnostic_error or "estimable",
                            })
            if seed_rows:
                frame = pd.DataFrame(seed_rows)
                session_rows.append({
                    **meta, "guarded": guarded,
                    "auc_state_conditional": float(frame.conditional_auc.mean()),
                    "auc_state_mean_fold": float(frame.mean_fold_auc.mean()),
                    "auc_state_pooled_v32": float(frame.pooled_auc_v32.mean()),
                    "state_calibrated_logloss_gain": float(
                        frame.calibrated_logloss_gain.mean()),
                    "state_calibrated_log_loss": float(
                        frame.calibrated_log_loss.mean()),
                    "state_raw_log_loss": float(frame.raw_log_loss.mean()),
                    "state_fixed_half_null_log_loss": float(
                        frame.fixed_half_null_log_loss.mean()),
                    "state_training_prevalence_null_log_loss": float(
                        frame.training_prevalence_null_log_loss.mean()),
                    "random_auc": float(frame.random_auc.mean()),
                    "circular_shift_auc": float(frame.circular_shift_auc.mean()),
                    "limiting_state_n": int(min(limiting)),
                    "n_seeds": int(len(frame)),
                    "estimability": "estimable",
                })
            else:
                session_rows.append({
                    **meta, "guarded": guarded,
                    "auc_state_conditional": np.nan,
                    "state_calibrated_logloss_gain": np.nan,
                    "limiting_state_n": int(min(limiting) if limiting else 0),
                    "n_seeds": 0,
                    "estimability": ";".join(sorted(set(errors))) or
                                    "state_nonestimable",
                })
    table = pd.DataFrame(session_rows)
    guarded = table[table.guarded].copy()
    auc_mice, _, _ = neural._mouse_summary(
        guarded.rename(columns={"auc_state_conditional": "auc"}),
        weight="limiting_state_n")
    gain_mice, _, _ = neural._mouse_summary(
        guarded.rename(columns={"state_calibrated_logloss_gain": "gain"}),
        value="gain", weight="limiting_state_n")
    auc_bca = neural._bca_mean(
        auc_mice.auc.to_numpy() if len(auc_mice) else np.array([]), seed=3301)
    gain_bca = neural._bca_mean(
        gain_mice.gain.to_numpy() if len(gain_mice) else np.array([]), seed=3302)
    summary = {
        "guarded": {
            "auc_state_conditional": auc_bca,
            "state_calibrated_logloss_gain": gain_bca,
            "n_mice": int(len(auc_mice)),
            "session_weight": "limiting outcome-balanced state class",
        },
        "unguarded_diagnostic": {},
    }
    unguarded = table[~table.guarded]
    if len(unguarded):
        mice, _, _ = neural._mouse_summary(
            unguarded.rename(columns={"auc_state_conditional": "auc"}),
            weight="limiting_state_n")
        summary["unguarded_diagnostic"] = {
            "auc_state_conditional": neural._bca_mean(
                mice.auc.to_numpy() if len(mice) else np.array([]), seed=3303),
            "n_mice": int(len(mice)),
        }
    return (table, pd.DataFrame(fold_rows),
            pd.DataFrame(representation_rows), summary)


def _nuisance_model(C: float, include_neural: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    continuous = [
        "flashes_before_change", "time_since_previous_change",
        "time_since_previous_lick", "time_since_previous_reward",
        "session_position", "pre_change_pupil", "pre_change_running",
    ]
    categorical = ["transition", "preceding_omission", "previous_outcome"]
    cont = continuous + (["neural_score"] if include_neural else [])
    pre = ColumnTransformer([
        ("continuous", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), cont),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(
            C=C, class_weight=None, solver="liblinear", max_iter=2000)),
    ])


def _inner_nuisance(frame: pd.DataFrame, y: np.ndarray, raw: np.ndarray, *,
                    C: float, include_neural: bool) -> tuple[np.ndarray, list[float], str | None]:
    from sklearn.metrics import log_loss
    probability = np.full(len(y), np.nan)
    losses = []
    for train, test in neural._folds(raw, n_blocks=4):
        if len(np.unique(y[train])) < 2:
            return probability, losses, "inner_training_class_missing"
        model = _nuisance_model(C, include_neural)
        model.fit(frame.iloc[train], y[train])
        predicted = model.predict_proba(frame.iloc[test])[:, 1]
        if not np.isfinite(predicted).all():
            return probability, losses, "inner_probability_nonfinite"
        probability[test] = np.clip(predicted, 1e-6, 1-1e-6)
        losses.append(float(log_loss(y[test], predicted, labels=[0, 1])))
    if not np.isfinite(probability).all():
        return probability, losses, "inner_probability_incomplete"
    return probability, losses, None


def _select_c(frame: pd.DataFrame, y: np.ndarray, raw: np.ndarray, *,
              include_neural: bool) -> tuple[float | None, np.ndarray, list[dict], str | None]:
    candidates = []
    probabilities = {}
    for C in C_GRID:
        probability, losses, error = _inner_nuisance(
            frame, y, raw, C=C, include_neural=include_neural)
        if error:
            return None, np.array([]), candidates, error
        mean = float(np.mean(losses))
        se = float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0
        candidates.append({"C": C, "mean_log_loss": mean, "se_log_loss": se})
        probabilities[C] = probability
    best = min(candidates, key=lambda row: row["mean_log_loss"])
    threshold = best["mean_log_loss"] + best["se_log_loss"]
    selected = min(row["C"] for row in candidates
                   if row["mean_log_loss"] <= threshold)
    for row in candidates:
        row["selected"] = bool(row["C"] == selected)
        row["one_se_threshold"] = threshold
    regenerated, _, error = _inner_nuisance(
        frame, y, raw, C=selected, include_neural=include_neural)
    if error:
        return None, np.array([]), candidates, error
    return selected, regenerated, candidates, None


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (float(np.corrcoef(a, b)[0, 1])
            if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else np.nan)


def _q2_session(item: dict) -> tuple[dict, list[dict], list[dict], str | None]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    labels = item["labels"]
    joined = labels.join(
        item["q2"].drop(columns=["trial_id", "change_time"], errors="ignore"))
    mask = (joined.engaged_B.fillna(False).astype(bool) &
            joined.keep_B.fillna(False).astype(bool) &
            (joined.late_hit.fillna(False).astype(bool) |
             joined.miss.fillna(False).astype(bool)) &
            joined.q2_covariates_complete.fillna(False).astype(bool))
    use = joined.loc[mask].reset_index(drop=True)
    if len(use) < neural.N_BLOCKS or use.late_hit.nunique() < 2:
        return {}, [], [], "q2_class_nonestimable"
    y = use.late_hit.astype(int).to_numpy()
    raw = use.trial_index.astype(int).to_numpy()
    X_all = item["arrays"]["events_baselined_post"][mask.to_numpy()]
    feature_columns = [
        "flashes_before_change", "time_since_previous_change",
        "time_since_previous_lick", "time_since_previous_reward",
        "session_position", "pre_change_pupil", "pre_change_running",
        "transition", "preceding_omission", "previous_outcome",
    ]
    seed_metrics, selection_rows, fold_rows = [], [], []
    oeid = item["meta"]["ophys_experiment_id"]
    for seed in range(neural.N_SEEDS):
        X = neural._subset_cells(X_all, neural.PRIMARY_K, seed, oeid)
        if X is None:
            return {}, selection_rows, fold_rows, "low_cells"
        predictions = {
            name: np.full(len(y), np.nan) for name in (
                "m0", "m1", "m0_raw", "m1_raw", "m0_c1", "m1_c1",
                "neural_raw", "q1_score")
        }
        orth_score, orth_residual = [], []
        for outer_fold, (outer_train, outer_test) in enumerate(
                neural._folds(raw), start=1):
            if len(np.unique(y[outer_train])) < 2:
                return {}, selection_rows, fold_rows, "q2_outer_training_class_missing"
            inner_score = np.full(len(outer_train), np.nan)
            inner_raw = raw[outer_train]
            for inner_train_local, inner_test_local in neural._folds(
                    inner_raw, n_blocks=4):
                inner_train = outer_train[inner_train_local]
                inner_test = outer_train[inner_test_local]
                if len(np.unique(y[inner_train])) < 2:
                    return {}, selection_rows, fold_rows, "q2_neural_inner_class_missing"
                scaler, decoder = _fit_decoder(
                    X, y, inner_train, C=neural.FROZEN_C50,
                    seed=seed, class_weight=None)
                inner_score[inner_test_local] = decoder.decision_function(
                    scaler.transform(X[inner_test]))
            if not np.isfinite(inner_score).all():
                return {}, selection_rows, fold_rows, "q2_neural_inner_score_incomplete"
            scaler, decoder = _fit_decoder(
                X, y, outer_train, C=neural.FROZEN_C50,
                seed=seed, class_weight=None)
            test_score = decoder.decision_function(scaler.transform(X[outer_test]))
            center, scale = float(inner_score.mean()), float(inner_score.std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                return {}, selection_rows, fold_rows, "q2_neural_score_zero_variance"
            train_frame = use.iloc[outer_train][feature_columns].copy()
            test_frame = use.iloc[outer_test][feature_columns].copy()
            train_frame["neural_score"] = (inner_score-center)/scale
            test_frame["neural_score"] = (test_score-center)/scale
            selected = {}
            for model_name, include_neural in (("m0", False), ("m1", True)):
                C, inner_probability, candidates, error = _select_c(
                    train_frame, y[outer_train], inner_raw,
                    include_neural=include_neural)
                for candidate in candidates:
                    selection_rows.append({
                        **item["meta"], "seed": seed,
                        "outer_fold": outer_fold, "model": model_name,
                        **candidate,
                    })
                if error or C is None:
                    return {}, selection_rows, fold_rows, (
                        f"q2_{model_name}_{error or 'C_nonestimable'}")
                try:
                    calibrator = _sigmoid_fit(
                        np.log(inner_probability/(1-inner_probability)),
                        y[outer_train])
                except ValueError as exc:
                    return {}, selection_rows, fold_rows, f"q2_{model_name}_{exc}"
                model = _nuisance_model(C, include_neural)
                model.fit(train_frame, y[outer_train])
                raw_probability = np.clip(
                    model.predict_proba(test_frame)[:, 1], 1e-6, 1-1e-6)
                logits = np.log(raw_probability/(1-raw_probability))
                predictions[model_name][outer_test] = _sigmoid_apply(
                    calibrator, logits)
                predictions[f"{model_name}_raw"][outer_test] = raw_probability
                comparator = _nuisance_model(1.0, include_neural)
                comparator.fit(train_frame, y[outer_train])
                predictions[f"{model_name}_c1"][outer_test] = np.clip(
                    comparator.predict_proba(test_frame)[:, 1], 1e-6, 1-1e-6)
                selected[model_name] = (C, logits)
            predictions["neural_raw"][outer_test] = np.clip(
                decoder.predict_proba(scaler.transform(X[outer_test]))[:, 1],
                1e-6, 1-1e-6)
            q1_scaler, q1 = _fit_decoder(
                X, y, outer_train, C=neural.FROZEN_C50,
                seed=seed, class_weight="balanced")
            predictions["q1_score"][outer_test] = q1.decision_function(
                q1_scaler.transform(X[outer_test]))
            m0_logit = selected["m0"][1]
            orth_score.append(_correlation(test_frame.neural_score, m0_logit))
            orth_residual.append(_correlation(
                test_frame.neural_score,
                y[outer_test]-predictions["m0"][outer_test]))
            fold_rows.append({
                **item["meta"], "seed": seed, "outer_fold": outer_fold,
                "m0_C": selected["m0"][0], "m1_C": selected["m1"][0],
                "n_train": int(len(outer_train)), "n_test": int(len(outer_test)),
                "test_prevalence": float(y[outer_test].mean()),
                "corr_neural_m0_logit": orth_score[-1],
                "corr_neural_m0_residual": orth_residual[-1],
            })
        if not all(np.isfinite(value).all() for value in predictions.values()):
            return {}, selection_rows, fold_rows, "q2_outer_probability_incomplete"
        m0_ci, m0_slope = neural._calibration_summary(y, predictions["m0"])
        m1_ci, m1_slope = neural._calibration_summary(y, predictions["m1"])
        seed_metrics.append({
            "n_trials": int(len(y)), "prevalence": float(y.mean()),
            "m0_log_loss": float(log_loss(y, predictions["m0"], labels=[0, 1])),
            "m1_log_loss": float(log_loss(y, predictions["m1"], labels=[0, 1])),
            "delta_log_loss": float(
                log_loss(y, predictions["m0"], labels=[0, 1]) -
                log_loss(y, predictions["m1"], labels=[0, 1])),
            "m0_auc": float(roc_auc_score(y, predictions["m0"])),
            "m1_auc": float(roc_auc_score(y, predictions["m1"])),
            "delta_auc": float(
                roc_auc_score(y, predictions["m1"]) -
                roc_auc_score(y, predictions["m0"])),
            "m0_brier": float(brier_score_loss(y, predictions["m0"])),
            "m1_brier": float(brier_score_loss(y, predictions["m1"])),
            "delta_brier": float(
                brier_score_loss(y, predictions["m0"]) -
                brier_score_loss(y, predictions["m1"])),
            "raw_delta_log_loss": float(
                log_loss(y, predictions["m0_raw"], labels=[0, 1]) -
                log_loss(y, predictions["m1_raw"], labels=[0, 1])),
            "raw_delta_auc": float(
                roc_auc_score(y, predictions["m1_raw"]) -
                roc_auc_score(y, predictions["m0_raw"])),
            "v32_C1_delta_log_loss": float(
                log_loss(y, predictions["m0_c1"], labels=[0, 1]) -
                log_loss(y, predictions["m1_c1"], labels=[0, 1])),
            "v32_C1_delta_auc": float(
                roc_auc_score(y, predictions["m1_c1"]) -
                roc_auc_score(y, predictions["m0_c1"])),
            "neural_only_auc": float(roc_auc_score(y, predictions["neural_raw"])),
            "q1_auc_same_trials": float(roc_auc_score(y, predictions["q1_score"])),
            "m0_calibration_intercept": m0_ci,
            "m0_calibration_slope": m0_slope,
            "m1_calibration_intercept": m1_ci,
            "m1_calibration_slope": m1_slope,
            "corr_neural_m0_logit": float(np.nanmean(orth_score)),
            "corr_neural_m0_residual": float(np.nanmean(orth_residual)),
        })
    frame = pd.DataFrame(seed_metrics)
    metrics = {
        column: (int(frame[column].iloc[0]) if column == "n_trials"
                 else float(frame[column].mean()))
        for column in frame.columns
    }
    return metrics, selection_rows, fold_rows, None


def _typed_status(interval: dict, *, null: float, coverage: bool) -> dict:
    if (not coverage or interval.get("low") is None or
            interval.get("high") is None):
        return {"status": "nonestimable", "reason": "coverage_or_interval"}
    if float(interval["low"]) > null:
        return {"status": "usable_positive", "reason": "ci_above_null"}
    if float(interval["high"]) < null:
        return {"status": "invalid_direction", "reason": "ci_below_null"}
    return {"status": "inconclusive", "reason": "ci_includes_null"}


def scan(features: Path, manifest_path: Path, out: Path, *,
         feature_release: str, feature_manifest_sha256: str) -> int:
    sessions, manifest = _feature_sessions(features, manifest_path)
    out.mkdir(parents=True, exist_ok=True)

    q1_rows, q1_fold_rows = [], []
    for item in sessions:
        meta = item["meta"]
        if meta["late_hit_B"] < 20 or meta["miss_B"] < 10:
            continue
        metrics, folds, error = _q1_session(item)
        q1_fold_rows.extend(folds)
        q1_rows.append({
            **meta, **metrics,
            "K": neural.PRIMARY_K, "C": neural.FROZEN_C50,
            "decoder_estimability": error or "estimable",
        })
    q1_all = pd.DataFrame(q1_rows)
    q1_primary = q1_all[q1_all.miss_B.ge(20)].copy()
    q1_mice, _, _ = neural._mouse_summary(q1_primary)

    threshold_sessions, threshold_summary = [], []
    for threshold in (10, 15, 20, 25, 30):
        selected = q1_all[q1_all.miss_B.ge(threshold)].copy()
        selected["miss_threshold"] = threshold
        threshold_sessions.append(selected)
        mice, _, _ = neural._mouse_summary(selected)
        interval = neural._bca_mean(
            mice.auc.to_numpy() if len(mice) else np.array([]),
            seed=3300+threshold)
        threshold_summary.append({
            "miss_threshold": threshold,
            "n_behavioral_sessions": int(len(selected)),
            "n_estimable_sessions": int(np.isfinite(selected.auc).sum()),
            "n_mice": int(len(mice)),
            "mouse_mean_auc": interval["mean"],
            "ci_low": interval["low"], "ci_high": interval["high"],
        })

    state_sessions, state_folds, state_representations, anchor = _state_anchor(
        sessions)
    q2_rows, q2_selection, q2_folds = [], [], []
    for item in sessions:
        if not item["meta"]["behavioral_eligible"]:
            continue
        metrics, selection, folds, error = _q2_session(item)
        q2_selection.extend(selection)
        q2_folds.extend(folds)
        q2_rows.append({
            **item["meta"], **metrics,
            "q2_estimability": error or "estimable",
        })
    q2_sessions = pd.DataFrame(q2_rows)
    q2_mouse_rows = []
    metric_columns = [
        column for column in q2_sessions.columns
        if column not in {
            "ophys_experiment_id", "behavior_session_id", "mouse_id",
            "project_code", "novel", "miss_B", "late_hit_B",
            "behavioral_eligible", "q2_estimability",
        } and pd.api.types.is_numeric_dtype(q2_sessions[column])
    ]
    valid_q2 = (q2_sessions[q2_sessions.delta_log_loss.notna()]
                if "delta_log_loss" in q2_sessions else q2_sessions.iloc[0:0])
    for mouse, group in valid_q2.groupby("mouse_id"):
        weights = np.maximum(group.miss_B.to_numpy(float), 1)
        row = {"mouse_id": mouse}
        for column in metric_columns:
            values = group[column].to_numpy(float)
            finite = np.isfinite(values)
            row[column] = (float(np.average(values[finite], weights=weights[finite]))
                           if finite.any() else np.nan)
        q2_mouse_rows.append(row)
    q2_mice = pd.DataFrame(q2_mouse_rows)

    auc_interval = anchor["guarded"]["auc_state_conditional"]
    gain_interval = anchor["guarded"]["state_calibrated_logloss_gain"]
    anchor_coverage = int(anchor["guarded"]["n_mice"]) >= 8
    auc_status = _typed_status(auc_interval, null=.5, coverage=anchor_coverage)
    gain_status = _typed_status(gain_interval, null=0.0, coverage=anchor_coverage)
    q1_margin = (.2*(float(auc_interval["mean"])-.5)
                 if auc_status["status"] == "usable_positive" else np.nan)
    q2_sesoi = (.2*float(gain_interval["mean"])
                if gain_status["status"] == "usable_positive" else np.nan)
    q1_sd = float(q1_mice.auc.std(ddof=1)) if len(q1_mice) > 1 else np.nan
    q2_sd = (float(q2_mice.delta_log_loss.std(ddof=1))
             if len(q2_mice) > 1 and "delta_log_loss" in q2_mice else np.nan)
    gates = neural._precision_gates(
        appendix_complete=True, q1_mice=len(q1_mice),
        anchor_mice=anchor["guarded"]["n_mice"], q2_mice=len(q2_mice),
        q1_sd=q1_sd, q2_sd=q2_sd, q1_margin=q1_margin,
        q2_sesoi=q2_sesoi)
    gates["confirm_ready"] = bool(
        gates["confirm_ready"] and
        auc_status["status"] == "usable_positive" and
        gain_status["status"] == "usable_positive")

    q1_primary.to_parquet(out / "q1_sessions.parquet", index=False)
    q1_mice.to_parquet(out / "q1_mice.parquet", index=False)
    pd.DataFrame(q1_fold_rows).to_parquet(
        out / "q1_fold_diagnostics.parquet", index=False)
    pd.DataFrame(threshold_summary).to_parquet(
        out / "threshold_sweep.parquet", index=False)
    (pd.concat(threshold_sessions, ignore_index=True)
     if threshold_sessions else pd.DataFrame()).to_parquet(
         out / "threshold_sweep_sessions.parquet", index=False)
    state_sessions.to_parquet(out / "state_sessions.parquet", index=False)
    state_folds.to_parquet(out / "state_fold_diagnostics.parquet", index=False)
    state_representations.to_parquet(
        out / "state_representation_diagnostics.parquet", index=False)
    q2_sessions.to_parquet(out / "q2_sessions.parquet", index=False)
    q2_mice.to_parquet(out / "q2_mice.parquet", index=False)
    pd.DataFrame(q2_selection).to_parquet(
        out / "q2_C_selection.parquet", index=False)
    pd.DataFrame(q2_folds).to_parquet(
        out / "q2_fold_diagnostics.parquet", index=False)

    result = {
        "schema": "neural-dev-v3.3",
        "feature_source": {
            "feature_release": feature_release,
            "feature_manifest_sha256": feature_manifest_sha256,
            "schema": neural.FEATURE_CACHE_SCHEMA,
            "n_active_experiments": 50,
            "n_source_experiments": int(len(manifest)),
            "cache_only": True,
            "neural_bundle_download": False,
            "allen_nwb_download": False,
        },
        "primary": {
            "target": "late-hit-vs-miss", "signal": "events",
            "window": [neural.FIT_START, neural.FIT_END],
            "K": neural.PRIMARY_K, "C": neural.FROZEN_C50,
            "session_estimator": "pooled_oof_auc",
            "conditional_auc_role": "diagnostic",
        },
        "anchor": {
            "authoritative": "guarded_pair_weighted_within_test_fold_auc",
            **anchor,
            "auc_status": auc_status,
            "state_logloss_status": gain_status,
        },
        "q2": {
            "C_grid": list(C_GRID),
            "selection": "nested one-SE smallest C separately for M0 and M1",
            "calibration": "training-only inner-OOF sigmoid",
            "primary": "calibrated_delta_log_loss",
            "mouse_bca": neural._bca_mean(
                q2_mice.delta_log_loss.to_numpy()
                if len(q2_mice) and "delta_log_loss" in q2_mice
                else np.array([]), seed=3304),
            "raw_C1_comparator": True,
        },
        "sesoi": {
            "q1_auc_boundary": (
                .5+q1_margin if np.isfinite(q1_margin) else None),
            "q1_margin": q1_margin if np.isfinite(q1_margin) else None,
            "q1_status": auc_status,
            "q2_delta_logloss": (
                q2_sesoi if np.isfinite(q2_sesoi) else None),
            "q2_status": gain_status,
        },
        "gates": {
            **gates, "q1_between_mouse_sd": q1_sd,
            "q2_between_mouse_sd": q2_sd, "required_mice": 8,
        },
        "q1_mouse_bca": neural._bca_mean(
            q1_mice.auc.to_numpy() if len(q1_mice) else np.array([]),
            seed=3305),
        "selection_sweep": {
            "miss_thresholds": [10, 15, 20, 25, 30],
            "late_hit_min": 20, "decoder_frozen": True,
        },
        "integrity": {
            "exact_feature_cache": True,
            "baseline_subtraction_all_passed": True,
        },
    }
    (out / "analysis-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", nargs="?")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-release", required=True)
    parser.add_argument("--feature-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.scan not in (None, "scan"):
        parser.error("only the scan command is supported")
    return scan(
        args.features, args.manifest, args.out,
        feature_release=args.feature_release,
        feature_manifest_sha256=args.feature_manifest_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

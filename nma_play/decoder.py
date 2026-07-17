"""A typed, single-experiment Q1 decoder for the educational Colab notebook."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .release import FeatureMatrix


@dataclass(frozen=True)
class DecoderConfig:
    k: int | None = 50
    C: float = 1e-4
    n_seeds: int = 10
    n_folds: int = 5
    purge: int = 10
    cv: str = "blocked"

    @property
    def exploratory(self) -> bool:
        return not (
            self.k == 50
            and self.C == 1e-4
            and self.n_seeds == 10
            and self.n_folds == 5
            and self.purge == 10
            and self.cv == "blocked"
        )

    def validate(self) -> None:
        if self.k is not None and self.k < 1:
            raise ValueError("k must be positive or None for all cells")
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be positive")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.purge < 0:
            raise ValueError("purge must be non-negative")
        if self.cv not in {"blocked", "random"}:
            raise ValueError("cv must be 'blocked' or 'random'")


@dataclass
class DecoderResult:
    status: str
    reason: str | None
    config: DecoderConfig
    experiment_id: int
    feature_name: str
    seed_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    oof: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def mean_auc(self) -> float:
        if self.seed_metrics.empty:
            return float("nan")
        return float(self.seed_metrics["auc"].mean())


def contiguous_purged_folds(
    raw_index: np.ndarray, n_blocks: int = 5, purge: int = 10
) -> list[tuple[np.ndarray, np.ndarray]]:
    raw_index = np.asarray(raw_index, dtype=int)
    order = np.argsort(raw_index)
    result = []
    for test in np.array_split(order, n_blocks):
        if not len(test):
            continue
        low, high = int(raw_index[test].min()), int(raw_index[test].max())
        train = np.setdiff1d(order, test)
        adjacent = (
            ((raw_index[train] >= low - purge) & (raw_index[train] < low))
            | ((raw_index[train] > high) & (raw_index[train] <= high + purge))
        )
        result.append((train[~adjacent], test))
    return result


def deterministic_cell_indices(
    n_cells: int, k: int | None, seed: int, experiment_id: int
) -> np.ndarray | None:
    if k is None:
        return np.arange(n_cells)
    if n_cells < k:
        return None
    digest = hashlib.sha256(f"{int(experiment_id)}:{int(k)}:{int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(n_cells, int(k), replace=False))


def _empty_result(
    reason: str,
    config: DecoderConfig,
    matrix: FeatureMatrix,
) -> DecoderResult:
    return DecoderResult(
        status="nonestimable",
        reason=reason,
        config=config,
        experiment_id=matrix.experiment_id,
        feature_name=matrix.name,
    )


def run_q1_decoder(
    matrix: FeatureMatrix,
    labels: pd.DataFrame,
    config: DecoderConfig | None = None,
) -> DecoderResult:
    """Fit the registered single-session late-hit-vs-miss decoder.

    This follows the v3.2/v3.3 Q1 session defaults but does not perform the
    authoritative mouse-level aggregation.
    """
    config = config or DecoderConfig()
    config.validate()
    required = {
        "trial_id", "trial_index", "engaged_B", "keep_B", "late_hit", "miss",
    }
    missing = required - set(labels.columns)
    if missing:
        return _empty_result(f"missing_label_columns:{','.join(sorted(missing))}", config, matrix)
    aligned = labels.set_index("trial_id").reindex(matrix.trial_ids).reset_index()
    if aligned["trial_index"].isna().any():
        return _empty_result("trial_alignment_failed", config, matrix)
    mask = (
        aligned["engaged_B"].fillna(False).astype(bool)
        & aligned["keep_B"].fillna(False).astype(bool)
        & (
            aligned["late_hit"].fillna(False).astype(bool)
            | aligned["miss"].fillna(False).astype(bool)
        )
    )
    X = np.asarray(matrix.values[mask.to_numpy()], dtype=float)
    selected_labels = aligned.loc[mask].reset_index(drop=True)
    y = selected_labels["late_hit"].astype(int).to_numpy()
    raw = selected_labels["trial_index"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return _empty_result("insufficient_classes", config, matrix)
    if not np.isfinite(X).all():
        return _empty_result("nonfinite_features", config, matrix)
    if config.k is not None and X.shape[1] < config.k:
        return _empty_result("low_cells", config, matrix)

    seed_rows: list[dict] = []
    fold_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    oof_columns: dict[str, np.ndarray] = {}
    selected_by_seed: dict[int, np.ndarray] = {}

    for seed in range(config.n_seeds):
        cell_index = deterministic_cell_indices(
            X.shape[1], config.k, seed, matrix.experiment_id
        )
        if cell_index is None:
            return _empty_result("low_cells", config, matrix)
        selected_by_seed[seed] = cell_index
        X_seed = X[:, cell_index]
        scores = np.full(len(y), np.nan)
        if config.cv == "blocked":
            splits = contiguous_purged_folds(raw, config.n_folds, config.purge)
        else:
            class_counts = np.bincount(y, minlength=2)
            if int(class_counts.min()) < config.n_folds:
                return _empty_result("random_cv_class_support_nonestimable", config, matrix)
            splits = list(
                StratifiedKFold(
                    config.n_folds, shuffle=True, random_state=seed
                ).split(X_seed, y)
            )
        if len(splits) != config.n_folds:
            return _empty_result("fold_construction_failed", config, matrix)

        for fold, (train, test) in enumerate(splits):
            train_classes = np.bincount(y[train], minlength=2)
            test_classes = np.bincount(y[test], minlength=2)
            fold_row = {
                "seed": seed,
                "fold": fold,
                "cv": config.cv,
                "n_train": len(train),
                "n_test": len(test),
                "train_negative": int(train_classes[0]),
                "train_positive": int(train_classes[1]),
                "test_negative": int(test_classes[0]),
                "test_positive": int(test_classes[1]),
                "test_raw_min": int(raw[test].min()),
                "test_raw_max": int(raw[test].max()),
                "estimable": bool((train_classes > 0).all()),
            }
            fold_rows.append(fold_row)
            if not fold_row["estimable"]:
                return _empty_result("temporal_support_nonestimable", config, matrix)
            scaler = StandardScaler().fit(X_seed[train])
            model = LogisticRegression(
                C=config.C,
                penalty="l2",
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
                max_iter=2000,
            )
            model.fit(scaler.transform(X_seed[train]), y[train])
            scores[test] = model.decision_function(scaler.transform(X_seed[test]))
            for local_cell, coefficient in zip(cell_index, model.coef_[0]):
                coefficient_rows.append({
                    "seed": seed,
                    "fold": fold,
                    "cell_index": int(local_cell),
                    "cell_id": int(matrix.cell_ids[local_cell]),
                    "coefficient": float(coefficient),
                })
        if not np.isfinite(scores).all():
            return _empty_result("score_nonestimable", config, matrix)
        seed_rows.append({
            "seed": seed,
            "auc": float(roc_auc_score(y, scores)),
            "n_trials": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "n_cells": len(cell_index),
        })
        oof_columns[f"score_seed_{seed}"] = scores

    oof = selected_labels[["trial_id", "trial_index", "late_hit", "miss"]].copy()
    oof["y"] = y
    for name, values in oof_columns.items():
        oof[name] = values
    score_columns = [name for name in oof.columns if name.startswith("score_seed_")]
    oof["mean_score"] = oof[score_columns].mean(axis=1)

    coefficients = pd.DataFrame(coefficient_rows)
    cell_summary_rows = []
    for index, cell_id in enumerate(matrix.cell_ids):
        selected_seeds = sum(index in indices for indices in selected_by_seed.values())
        values = coefficients.loc[
            coefficients["cell_index"].eq(index), "coefficient"
        ]
        cell_summary_rows.append({
            "cell_index": index,
            "cell_id": int(cell_id),
            "selection_frequency": selected_seeds / config.n_seeds,
            "median_coefficient": float(values.median()) if len(values) else np.nan,
            "mean_abs_coefficient": float(values.abs().mean()) if len(values) else np.nan,
        })
    return DecoderResult(
        status="estimable",
        reason=None,
        config=config,
        experiment_id=matrix.experiment_id,
        feature_name=matrix.name,
        seed_metrics=pd.DataFrame(seed_rows),
        fold_metrics=pd.DataFrame(fold_rows),
        oof=oof,
        cell_summary=pd.DataFrame(cell_summary_rows),
    )

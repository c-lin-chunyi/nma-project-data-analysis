#!/usr/bin/env python3
"""Render preregistered v3.4 results as APA-style publication figures.

This command consumes only the public, aggregate assets from an immutable
``neural-confirm-v3.4-*`` GitHub Release.  It never fits a model or recomputes
the registered BCa intervals stored in ``analysis-manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


PLOT_SCHEMA = "neural-confirm-v3.4-plots-v1"
SOURCE_SCHEMA = "neural-confirm-v3.4"
SOURCE_TAG = re.compile(r"^neural-confirm-v3\.4-[0-9a-f]{64}$")
SOURCE_FILES = (
    "analysis-manifest.json",
    "q1_sessions.parquet",
    "q1_mice.parquet",
    "q1_fold_diagnostics.parquet",
    "q2_mice.parquet",
    "q2_C_selection.parquet",
    "coverage_failures.parquet",
)
FONT_FILES = (
    "STIXTwoText-Regular.otf",
    "STIXTwoText-Italic.otf",
    "STIXTwoText-Bold.otf",
    "STIXTwoText-BoldItalic.otf",
    "STIXTwoMath-Regular.otf",
)
FIGURES = (
    ("fig01_q1_mouse_auc", "Figure 1", "Q1 neural decoding performance"),
    ("fig02_q1_session_auc_by_mouse", "Figure 2", "Q1 session-level AUC by mouse"),
    ("fig03_q1_decoder_robustness", "Figure 3", "Q1 decoder robustness"),
    ("fig04_q1_temporal_fold_stability", "Figure 4", "Q1 temporal fold stability"),
    (
        "fig05_q2_incremental_performance",
        "Figure 5",
        "Q2 incremental predictive performance",
    ),
    (
        "fig06_q2_m0_m1_performance",
        "Figure 6",
        "Q2 behavioral and neural model performance",
    ),
    (
        "supp_fig01_q2_regularization_selection",
        "Figure S1",
        "Q2 regularization selection",
    ),
    (
        "supp_fig02_estimability_coverage",
        "Figure S2",
        "Analysis estimability and coverage",
    ),
)
C_GRID = tuple(10.0**power for power in range(-4, 3))
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#C9C9C9"
BLACK = "#111111"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    *,
    allow_schema_less_empty: bool = True,
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing and not (allow_schema_less_empty and frame.empty):
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def validate_interval(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    result = {key: value.get(key) for key in ("mean", "low", "high", "n_mice")}
    points = [result[key] for key in ("mean", "low", "high")]
    all_missing = all(item is None for item in points)
    if all_missing:
        return result
    if any(item is None for item in points):
        raise ValueError(f"{label} has an incomplete interval")
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in points):
        raise ValueError(f"{label} contains a nonfinite interval")
    mean, low, high = (float(result[key]) for key in ("mean", "low", "high"))
    if not low <= mean <= high:
        raise ValueError(f"{label} is not ordered low <= mean <= high")
    result.update(mean=mean, low=low, high=high)
    return result


def load_inputs(root: Path, source_release: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    if not SOURCE_TAG.fullmatch(source_release):
        raise ValueError(
            "source release must match neural-confirm-v3.4-[0-9a-f]{64}"
        )
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source assets: {missing}")
    manifest = json.loads((root / "analysis-manifest.json").read_text())
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"source manifest schema must be {SOURCE_SCHEMA}")
    if manifest.get("confirm_data_accessed") is not True:
        raise ValueError("source manifest must describe the completed CONFIRM analysis")
    if manifest.get("primary", {}).get("SESOI") != 0.55:
        raise ValueError("source manifest has an unexpected Q1 SESOI")
    validate_interval(
        manifest.get("primary", {}).get("mouse_interval"), "Q1 mouse interval"
    )
    secondary_intervals = manifest.get("secondary", {}).get("mouse_intervals")
    if not isinstance(secondary_intervals, dict):
        raise ValueError("source manifest is missing Q2 mouse intervals")
    for metric in ("delta_log_loss", "delta_brier", "delta_auc"):
        validate_interval(secondary_intervals.get(metric), f"Q2 {metric} interval")

    frames = {
        "q1_sessions": pd.read_parquet(root / "q1_sessions.parquet"),
        "q1_mice": pd.read_parquet(root / "q1_mice.parquet"),
        "q1_folds": pd.read_parquet(root / "q1_fold_diagnostics.parquet"),
        "q2_mice": pd.read_parquet(root / "q2_mice.parquet"),
        "q2_selection": pd.read_parquet(root / "q2_C_selection.parquet"),
        "coverage": pd.read_parquet(root / "coverage_failures.parquet"),
    }
    require_columns(frames["q1_sessions"], ("mouse_id", "auc", "miss_B"), "q1_sessions")
    require_columns(frames["q1_mice"], ("mouse_id", "auc"), "q1_mice")
    require_columns(
        frames["q1_folds"],
        ("mouse_id", "behavior_session_id", "seed", "fold", "auc", "miss_B"),
        "q1_fold_diagnostics",
    )
    require_columns(
        frames["q2_mice"],
        (
            "mouse_id",
            "delta_log_loss",
            "delta_brier",
            "delta_auc",
            "m0_log_loss",
            "m1_log_loss",
            "m0_brier",
            "m1_brier",
            "m0_auc",
            "m1_auc",
        ),
        "q2_mice",
    )
    require_columns(
        frames["q2_selection"],
        ("mouse_id", "model", "C", "selected"),
        "q2_C_selection",
    )
    require_columns(
        frames["coverage"],
        ("mouse_id", "behavioral_eligible", "q1_estimability", "q2_estimability"),
        "coverage_failures",
    )
    return manifest, frames


def validate_fonts(font_dir: Path) -> dict:
    source_path = font_dir / "SOURCE.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"missing font provenance: {source_path}")
    source = json.loads(source_path.read_text())
    expected = source.get("files")
    if source.get("tag") != "v2.13b171" or not isinstance(expected, dict):
        raise ValueError("font provenance must pin STIX Two v2.13b171")
    for name in (*FONT_FILES, "OFL.txt"):
        path = font_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored font asset: {path}")
        if expected.get(name) != sha256(path):
            raise ValueError(f"font checksum mismatch: {name}")
    return source


def mouse_mapping(frames: Mapping[str, pd.DataFrame]) -> tuple[dict[object, str], str]:
    all_ids: set[object] = set()
    for frame in frames.values():
        if "mouse_id" in frame:
            all_ids.update(frame["mouse_id"].dropna().tolist())
    q1 = frames["q1_mice"]
    scores = (
        q1.set_index("mouse_id")["auc"].to_dict()
        if {"mouse_id", "auc"} <= set(q1.columns)
        else {}
    )

    def key(mouse: object) -> tuple[int, float, str]:
        value = scores.get(mouse, np.nan)
        finite = pd.notna(value) and np.isfinite(float(value))
        return (0 if finite else 1, float(value) if finite else math.inf, str(mouse))

    ordered = sorted(all_ids, key=key)
    mapping = {mouse: f"M{index:02d}" for index, mouse in enumerate(ordered, start=1)}
    private_pairs = [(str(mouse), mapping[mouse]) for mouse in ordered]
    return mapping, _json_digest(private_pairs)


def weighted_mouse_metric(
    frame: pd.DataFrame, value: str, *, weight: str = "miss_B"
) -> pd.DataFrame:
    require_columns(frame, ("mouse_id", value, weight), f"metric {value}")
    use = frame[
        np.isfinite(pd.to_numeric(frame[value], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame[weight], errors="coerce"))
    ].copy()
    rows = []
    for mouse, group in use.groupby("mouse_id", sort=False):
        weights = np.maximum(group[weight].to_numpy(float), 1.0)
        rows.append(
            {"mouse_id": mouse, value: float(np.average(group[value], weights=weights))}
        )
    return pd.DataFrame(rows, columns=["mouse_id", value])


def fold_mouse_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ("mouse_id", "behavior_session_id", "seed", "fold", "auc", "miss_B"),
        "q1 folds",
    )
    if frame.empty:
        return pd.DataFrame(columns=["mouse_id", "fold", "auc"])
    use = frame.copy()
    if "estimability" in use:
        use = use[use["estimability"].eq("estimable")]
    use = use[np.isfinite(pd.to_numeric(use["auc"], errors="coerce"))]
    session_fold = (
        use.groupby(["mouse_id", "behavior_session_id", "fold"], as_index=False)
        .agg(auc=("auc", "mean"), miss_B=("miss_B", "first"))
    )
    rows = []
    for (mouse, fold), group in session_fold.groupby(["mouse_id", "fold"]):
        weights = np.maximum(group["miss_B"].to_numpy(float), 1.0)
        rows.append(
            {
                "mouse_id": mouse,
                "fold": int(fold),
                "auc": float(np.average(group["auc"], weights=weights)),
            }
        )
    return pd.DataFrame(rows, columns=["mouse_id", "fold", "auc"])


def coverage_counts(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ("behavioral_eligible", "q1_estimability", "q2_estimability"),
        "coverage",
    )
    rows = []
    for analysis, status_column in (
        ("Q1", "q1_estimability"),
        ("Q2", "q2_estimability"),
    ):
        for row in frame.itertuples(index=False):
            eligible = bool(getattr(row, "behavioral_eligible"))
            status = getattr(row, status_column)
            if not eligible:
                category = "Behaviorally ineligible"
            elif str(status) == "estimable":
                category = "Estimable"
            else:
                category = "Eligible, not estimable"
            rows.append({"analysis": analysis, "category": category})
    counts = (
        pd.DataFrame(rows)
        .value_counts(["analysis", "category"])
        .rename("count")
        .reset_index()
    )
    return counts


def _configure_matplotlib(font_dir: Path) -> tuple[object, object, object]:
    import matplotlib

    matplotlib.use("pgf")
    escaped = font_dir.resolve().as_posix().replace("#", r"\#") + "/"
    preamble = "\n".join(
        (
            r"\usepackage{fontspec}",
            r"\usepackage{unicode-math}",
            rf"\setmainfont[Path={{{escaped}}},"
            r"UprightFont=STIXTwoText-Regular.otf,"
            r"ItalicFont=STIXTwoText-Italic.otf,"
            r"BoldFont=STIXTwoText-Bold.otf,"
            r"BoldItalicFont=STIXTwoText-BoldItalic.otf]{STIX Two Text}",
            rf"\setmathfont[Path={{{escaped}}}]{{STIXTwoMath-Regular.otf}}",
        )
    )
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "xelatex",
            "pgf.rcfonts": False,
            "pgf.preamble": preamble,
            "font.family": "serif",
            "font.serif": ["STIX Two Text"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "svg.hashsalt": "neural-confirm-v3.4-plots-v1",
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", rc=dict(matplotlib.rcParams))
    return plt, sns, FuncFormatter


def _apa_number(value: float, _position: int | None = None) -> str:
    if not math.isfinite(float(value)):
        return ""
    if value == 0:
        return "0"
    text = f"{value:.2f}"
    if 0 < value < 1:
        return text[1:]
    if -1 < value < 0:
        return "-" + text[2:]
    return text


def _style_axis(ax: object, sns: object) -> None:
    sns.despine(ax=ax)
    ax.grid(False)
    ax.tick_params(width=0.8, length=3, color=BLACK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLACK)


def _panel(ax: object, label: str) -> None:
    ax.text(
        0.0,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=10,
        clip_on=False,
    )


def _empty(ax: object) -> None:
    ax.text(0.5, 0.5, "Not estimable", transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


def _auc_limits(values: Iterable[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], float)
    upper = max(0.65, float(finite.max()) + 0.04) if finite.size else 0.65
    lower = min(0.46, float(finite.min()) - 0.04) if finite.size else 0.46
    return max(0.0, lower), min(1.0, upper)


def _title(ax: object, text: str) -> None:
    ax.set_title(text, loc="left", pad=9)


def plot_q1_mouse_auc(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    manifest: dict,
    _: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _title(ax, "Q1 neural decoding performance")
    table = frames["q1_mice"]
    values = (
        table.loc[np.isfinite(table["auc"]), "auc"].to_numpy(float)
        if "auc" in table
        else np.array([])
    )
    interval = validate_interval(manifest["primary"]["mouse_interval"], "Q1 interval")
    if values.size:
        rng = np.random.default_rng(3401)
        ax.scatter(
            rng.normal(0.0, 0.035, len(values)),
            values,
            s=28,
            facecolor=BLUE,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
            zorder=3,
            label="Mouse",
        )
    if interval["mean"] is not None:
        ax.errorbar(
            1.0,
            interval["mean"],
            yerr=[
                [interval["mean"] - interval["low"]],
                [interval["high"] - interval["mean"]],
            ],
            fmt="D",
            color=BLACK,
            markerfacecolor=VERMILLION,
            markeredgecolor=BLACK,
            capsize=4,
            linewidth=1.4,
            zorder=4,
            label="Mean (95\\% BCa CI)",
        )
    else:
        ax.text(1.0, 0.575, "Not estimable", ha="center", va="center")
    ax.axhline(0.50, color=GRAY, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.axhline(0.55, color=VERMILLION, linewidth=0.9, linestyle=(0, (6, 2)))
    ax.text(1.45, 0.50, "Chance", ha="right", va="bottom", color=GRAY)
    ax.text(1.45, 0.55, r"SESOI $=.55$", ha="right", va="bottom", color=VERMILLION)
    ax.set_xlim(-0.45, 1.5)
    ax.set_xticks([0, 1], ["Mice", "Mean"])
    limits = _auc_limits(
        [*values, *(item for item in (interval["low"], interval["high"]) if item is not None)]
    )
    ax.set_ylim(*limits)
    ax.set_ylabel("AUC")
    ax.yaxis.set_major_formatter(formatter(_apa_number))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left")
    _style_axis(ax, sns)
    fig.tight_layout()
    return fig


def plot_q1_sessions(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _title(ax, "Q1 session-level AUC by mouse")
    sessions, mice = frames["q1_sessions"], frames["q1_mice"]
    order = [mouse for mouse, _label in sorted(mouse_map.items(), key=lambda item: item[1])]
    x_positions = {mouse: index for index, mouse in enumerate(order)}
    if "auc" in sessions:
        use = sessions[np.isfinite(pd.to_numeric(sessions["auc"], errors="coerce"))]
        rng = np.random.default_rng(3402)
        for mouse, group in use.groupby("mouse_id"):
            x = x_positions.get(mouse)
            if x is None:
                continue
            ax.scatter(
                rng.normal(x, 0.075, len(group)),
                group["auc"],
                s=15,
                facecolor=LIGHT_GRAY,
                edgecolor=GRAY,
                linewidth=0.35,
                zorder=2,
            )
    if {"mouse_id", "auc"} <= set(mice.columns):
        use_mice = mice[np.isfinite(pd.to_numeric(mice["auc"], errors="coerce"))]
        ax.scatter(
            [x_positions[mouse] for mouse in use_mice["mouse_id"]],
            use_mice["auc"],
            s=32,
            marker="D",
            facecolor=BLUE,
            edgecolor=BLACK,
            linewidth=0.45,
            zorder=4,
            label="Mouse estimate",
        )
    ax.axhline(0.50, color=GRAY, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axhline(0.55, color=VERMILLION, linewidth=0.8, linestyle=(0, (6, 2)))
    ax.set_xticks(
        range(len(order)), [mouse_map[mouse] for mouse in order], rotation=90
    )
    ax.set_xlim(-0.7, max(len(order) - 0.3, 0.7))
    auc_values = []
    if "auc" in sessions:
        auc_values.extend(pd.to_numeric(sessions["auc"], errors="coerce").dropna())
    if "auc" in mice:
        auc_values.extend(pd.to_numeric(mice["auc"], errors="coerce").dropna())
    ax.set_ylim(*_auc_limits(auc_values))
    ax.set_xlabel("Mouse")
    ax.set_ylabel("AUC")
    ax.yaxis.set_major_formatter(formatter(_apa_number))
    ax.legend(loc="upper left")
    _style_axis(ax, sns)
    fig.tight_layout()
    return fig


def plot_q1_robustness(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _title(ax, "Q1 decoder robustness")
    source = frames["q1_sessions"]
    specifications = (
        ("auc", "Primary events"),
        ("conditional_auc", "Conditional"),
        ("random_auc", "Random CV"),
        ("dff_auc", r"$\mathrm{dF/F}$"),
    )
    merged: pd.DataFrame | None = None
    for column, label in specifications:
        if column not in source:
            metric = pd.DataFrame(columns=["mouse_id", label])
        else:
            metric = weighted_mouse_metric(source, column).rename(columns={column: label})
        merged = metric if merged is None else merged.merge(metric, on="mouse_id", how="outer")
    assert merged is not None
    xs = np.arange(len(specifications))
    for _index, row in merged.iterrows():
        values = np.asarray([row[label] for _column, label in specifications], float)
        finite = np.isfinite(values)
        ax.plot(xs[finite], values[finite], color=LIGHT_GRAY, linewidth=0.65, zorder=1)
        ax.scatter(xs[finite], values[finite], color=BLUE, s=17, alpha=0.7, zorder=2)
    means = np.asarray(
        [pd.to_numeric(merged[label], errors="coerce").mean() for _column, label in specifications]
    )
    ax.plot(xs, means, color=BLACK, linewidth=1.5, zorder=3)
    ax.scatter(xs, means, marker="D", s=34, color=VERMILLION, edgecolor=BLACK, zorder=4)
    ax.axhline(0.50, color=GRAY, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks(xs, [label for _column, label in specifications])
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_ylabel("AUC")
    ax.yaxis.set_major_formatter(formatter(_apa_number))
    ax.set_ylim(*_auc_limits(merged.drop(columns=["mouse_id"]).to_numpy().ravel()))
    _style_axis(ax, sns)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    return fig


def plot_q1_folds(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _title(ax, "Q1 temporal fold stability")
    table = fold_mouse_metrics(frames["q1_folds"])
    if table.empty:
        _empty(ax)
    else:
        for _mouse, group in table.groupby("mouse_id"):
            group = group.sort_values("fold")
            ax.plot(
                group["fold"],
                group["auc"],
                color=LIGHT_GRAY,
                linewidth=0.7,
                marker="o",
                markersize=2.5,
                alpha=0.85,
            )
        mean = table.groupby("fold", as_index=False)["auc"].mean()
        ax.plot(
            mean["fold"],
            mean["auc"],
            color=BLACK,
            linewidth=1.6,
            marker="D",
            markerfacecolor=VERMILLION,
            markeredgecolor=BLACK,
            markersize=5,
            label="Mouse mean",
            zorder=4,
        )
        ax.axhline(0.50, color=GRAY, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Temporal test fold")
        ax.set_ylabel("AUC")
        ax.yaxis.set_major_formatter(formatter(_apa_number))
        ax.set_ylim(*_auc_limits(table["auc"]))
        ax.legend(loc="upper left")
    _style_axis(ax, sns)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    return fig


def _delta_limits(values: Iterable[float], interval: dict) -> tuple[float, float]:
    candidates = [float(value) for value in values if np.isfinite(value)]
    candidates.extend(
        float(value)
        for value in (interval["low"], interval["high"])
        if value is not None
    )
    bound = max((abs(value) for value in candidates), default=0.01)
    margin = max(bound * 0.16, 0.002)
    return -bound - margin, bound + margin


def plot_q2_incremental(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    manifest: dict,
    mouse_map: Mapping[object, str],
) -> object:
    specifications = (
        ("delta_log_loss", r"$\Delta$ log loss"),
        ("delta_brier", r"$\Delta$ Brier score"),
        ("delta_auc", r"$\Delta$ AUC"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8))
    fig.suptitle(
        "Q2 incremental predictive performance",
        x=0.075,
        y=1.02,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    table = frames["q2_mice"]
    intervals = manifest["secondary"]["mouse_intervals"]
    for index, (ax, (metric, ylabel)) in enumerate(zip(axes, specifications)):
        _panel(ax, chr(ord("A") + index))
        interval = validate_interval(intervals[metric], f"Q2 {metric}")
        values = (
            pd.to_numeric(table[metric], errors="coerce").dropna().to_numpy(float)
            if metric in table
            else np.array([])
        )
        if values.size:
            rng = np.random.default_rng(3410 + index)
            ax.scatter(
                rng.normal(0.0, 0.035, len(values)),
                values,
                s=22,
                color=BLUE,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.8,
            )
        if interval["mean"] is not None:
            ax.errorbar(
                1,
                interval["mean"],
                yerr=[
                    [interval["mean"] - interval["low"]],
                    [interval["high"] - interval["mean"]],
                ],
                fmt="D",
                color=BLACK,
                markerfacecolor=VERMILLION,
                capsize=3,
                zorder=4,
            )
        else:
            ax.text(1, 0, "Not estimable", ha="center", va="bottom", rotation=90)
        ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_xticks([0, 1], ["Mice", "Mean"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(*_delta_limits(values, interval))
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(formatter(_apa_number))
        _style_axis(ax, sns)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    return fig


def plot_q2_models(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    specifications = (
        ("m0_log_loss", "m1_log_loss", "Log loss"),
        ("m0_brier", "m1_brier", "Brier score"),
        ("m0_auc", "m1_auc", "AUC"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8))
    fig.suptitle(
        "Q2 behavioral and neural model performance",
        x=0.075,
        y=1.02,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    table = frames["q2_mice"]
    for index, (ax, (m0, m1, ylabel)) in enumerate(zip(axes, specifications)):
        _panel(ax, chr(ord("A") + index))
        if {m0, m1} <= set(table.columns):
            use = table[
                np.isfinite(pd.to_numeric(table[m0], errors="coerce"))
                & np.isfinite(pd.to_numeric(table[m1], errors="coerce"))
            ]
        else:
            use = pd.DataFrame()
        if use.empty:
            _empty(ax)
        else:
            for row in use.itertuples(index=False):
                values = [float(getattr(row, m0)), float(getattr(row, m1))]
                ax.plot([0, 1], values, color=LIGHT_GRAY, linewidth=0.7, zorder=1)
                ax.scatter(
                    [0, 1],
                    values,
                    c=[GRAY, BLUE],
                    s=17,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=2,
                )
            means = [float(use[m0].mean()), float(use[m1].mean())]
            ax.plot([0, 1], means, color=BLACK, linewidth=1.5, zorder=3)
            ax.scatter(
                [0, 1],
                means,
                marker="D",
                s=32,
                c=[GRAY, VERMILLION],
                edgecolor=BLACK,
                linewidth=0.45,
                zorder=4,
            )
            values = use[[m0, m1]].to_numpy(float)
            low, high = float(values.min()), float(values.max())
            margin = max((high - low) * 0.13, 0.01)
            ax.set_ylim(low - margin, high + margin)
            ax.set_xticks([0, 1], ["M0", "M1"])
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(formatter(_apa_number))
        _style_axis(ax, sns)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    return fig


def plot_q2_selection(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    _title(ax, "Q2 regularization selection")
    table = frames["q2_selection"]
    if table.empty:
        _empty(ax)
    else:
        selected = table[table["selected"].fillna(False).astype(bool)].copy()
        selected["model"] = selected["model"].astype(str).str.upper()
        rows = []
        for model in ("M0", "M1"):
            values = pd.to_numeric(
                selected.loc[selected["model"].eq(model), "C"], errors="coerce"
            )
            denominator = int(values.notna().sum())
            for candidate in C_GRID:
                count = int(np.isclose(values.to_numpy(float), candidate).sum())
                rows.append(
                    {
                        "model": model,
                        "C": candidate,
                        "frequency": count / denominator if denominator else 0.0,
                    }
                )
        summary = pd.DataFrame(rows)
        x = np.arange(len(C_GRID))
        width = 0.36
        for offset, model, color, hatch in (
            (-width / 2, "M0", GRAY, "//"),
            (width / 2, "M1", BLUE, ""),
        ):
            values = summary.loc[summary["model"].eq(model), "frequency"]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=color,
                edgecolor=BLACK,
                linewidth=0.45,
                hatch=hatch,
                label=model,
            )
        ax.set_xticks(x, [rf"$10^{{{power}}}$" for power in range(-4, 3)])
        ax.set_xlabel(r"Selected $C$")
        ax.set_ylabel("Selection frequency")
        ax.yaxis.set_major_formatter(formatter(_apa_number))
        ax.set_ylim(0, max(0.2, float(summary["frequency"].max()) * 1.15))
        ax.legend(loc="upper right")
    _style_axis(ax, sns)
    fig.tight_layout()
    return fig


def plot_coverage(
    plt: object,
    sns: object,
    formatter: object,
    frames: Mapping[str, pd.DataFrame],
    _: dict,
    mouse_map: Mapping[object, str],
) -> object:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    _title(ax, "Analysis estimability and coverage")
    counts = coverage_counts(frames["coverage"])
    analyses = ("Q1", "Q2")
    categories = (
        ("Estimable", BLUE),
        ("Eligible, not estimable", VERMILLION),
        ("Behaviorally ineligible", LIGHT_GRAY),
    )
    left = np.zeros(len(analyses), float)
    for category, color in categories:
        values = np.asarray(
            [
                counts.loc[
                    counts["analysis"].eq(analysis)
                    & counts["category"].eq(category),
                    "count",
                ].sum()
                for analysis in analyses
            ],
            float,
        )
        bars = ax.barh(
            analyses,
            values,
            left=left,
            color=color,
            edgecolor=BLACK,
            linewidth=0.45,
            label=category,
        )
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color=BLACK,
                    fontsize=8,
                )
        left += values
    ax.set_xlabel("Sessions")
    ax.invert_yaxis()
    maximum = float(left.max()) if len(left) else 1.0
    ax.set_xlim(0, maximum * 1.35)
    ax.legend(loc="upper right")
    _style_axis(ax, sns)
    fig.tight_layout()
    return fig


def _write_captions(path: Path) -> None:
    notes = {
        "Figure 1": "Dots represent mice; the diamond and interval are the registered equal-mouse mean and 95% BCa confidence interval.",
        "Figure 2": "Circles represent sessions; diamonds represent registered mouse estimates.",
        "Figure 3": "Lines connect diagnostic estimates from the same mouse.",
        "Figure 4": "Gray lines represent mice; the black line represents the equal-mouse mean.",
        "Figure 5": "Positive values favor M1; intervals are the registered descriptive 95% BCa confidence intervals.",
        "Figure 6": "Lines connect M0 and M1 estimates from the same mouse.",
        "Figure S1": "Bars show the frequency of registered one-standard-error selections.",
        "Figure S2": "Counts are based on the released session-level estimability table.",
    }
    lines = []
    for _stem, number, title in FIGURES:
        lines.extend(
            (
                f"**{number}**",
                "",
                f"*{title}*",
                "",
                f"*Note.* {notes[number]}",
                "",
            )
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def _write_preamble(path: Path, font_dir: Path) -> None:
    root = font_dir.resolve().as_posix() + "/"
    path.write_text(
        "\n".join(
            (
                r"\usepackage{fontspec}",
                r"\usepackage{unicode-math}",
                rf"\setmainfont[Path={{{root}}},",
                r"  UprightFont=STIXTwoText-Regular.otf,",
                r"  ItalicFont=STIXTwoText-Italic.otf,",
                r"  BoldFont=STIXTwoText-Bold.otf,",
                r"  BoldItalicFont=STIXTwoText-BoldItalic.otf]{STIX Two Text}",
                rf"\setmathfont[Path={{{root}}}]{{STIXTwoMath-Regular.otf}}",
                "",
            )
        )
    )


def _render_figure(fig: object, stem: str, output: Path, plt: object) -> list[Path]:
    pgf = output / f"{stem}.pgf"
    pdf = output / f"{stem}.pdf"
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    fig.savefig(pgf, bbox_inches="tight")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "NMA Project Data Analysis",
            "Subject": "Neural CONFIRM v3.4 publication figure",
            "Creator": "Matplotlib PGF and XeLaTeX",
        },
    )
    plt.close(fig)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
    png_root = output / f".{stem}-600dpi"
    generated = png_root.with_suffix(".png")
    if generated.exists():
        generated.unlink()
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-r",
            "600",
            str(pdf),
            str(png_root),
        ],
        check=True,
    )
    generated.replace(png)
    for path in (pgf, pdf, svg, png):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"rendering did not produce {path}")
    return [pgf, pdf, svg, png]


def render(
    input_dir: Path,
    output_dir: Path,
    source_release: str,
    font_dir: Path,
) -> dict:
    if shutil.which("xelatex") is None:
        raise RuntimeError("xelatex is required for PGF rendering")
    if shutil.which("pdftocairo") is None:
        raise RuntimeError("pdftocairo is required for SVG and PNG rendering")
    manifest, frames = load_inputs(input_dir, source_release)
    font_source = validate_fonts(font_dir)
    mouse_map, mouse_map_hash = mouse_mapping(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, sns, formatter = _configure_matplotlib(font_dir)
    np.random.seed(3400)
    renderers: tuple[Callable[..., object], ...] = (
        plot_q1_mouse_auc,
        plot_q1_sessions,
        plot_q1_robustness,
        plot_q1_folds,
        plot_q2_incremental,
        plot_q2_models,
        plot_q2_selection,
        plot_coverage,
    )
    rendered: list[Path] = []
    for (stem, _number, _title_text), renderer in zip(FIGURES, renderers):
        figure = renderer(plt, sns, formatter, frames, manifest, mouse_map)
        rendered.extend(_render_figure(figure, stem, output_dir, plt))
    captions = output_dir / "captions.md"
    preamble = output_dir / "pgf-preamble.tex"
    _write_captions(captions)
    _write_preamble(preamble, font_dir)
    rendered.extend((captions, preamble))

    q1_interval = validate_interval(manifest["primary"]["mouse_interval"], "Q1 interval")
    q2_intervals = {
        metric: validate_interval(
            manifest["secondary"]["mouse_intervals"][metric], f"Q2 {metric}"
        )
        for metric in ("delta_log_loss", "delta_brier", "delta_auc")
    }
    result = {
        "schema": PLOT_SCHEMA,
        "source": {
            "release": source_release,
            "analysis_manifest_sha256": sha256(input_dir / "analysis-manifest.json"),
            "schema": SOURCE_SCHEMA,
            "outcome": manifest.get("outcome"),
            "input_sha256": {
                name: sha256(input_dir / name) for name in SOURCE_FILES
            },
        },
        "plot_code_commit": os.environ.get("GITHUB_SHA"),
        "requirements_sha256": (
            sha256(Path("requirements-plot.txt"))
            if Path("requirements-plot.txt").is_file()
            else None
        ),
        "fonts": {
            "family_text": "STIX Two Text",
            "family_math": "STIX Two Math",
            "version": font_source["version"],
            "source_tag": font_source["tag"],
            "sha256": {name: sha256(font_dir / name) for name in FONT_FILES},
        },
        "mouse_anonymization": {
            "scheme": "Q1-AUC-ordered M01-MNN",
            "n_mice": len(mouse_map),
            "mapping_sha256": mouse_map_hash,
            "original_ids_exported": False,
        },
        "registered_intervals_used_without_recomputation": {
            "q1_auc": q1_interval,
            "q2": q2_intervals,
        },
        "figures": [
            {
                "stem": stem,
                "number": number,
                "title": title,
                "formats": {
                    extension: {
                        "sha256": sha256(output_dir / f"{stem}.{extension}"),
                        "bytes": (output_dir / f"{stem}.{extension}").stat().st_size,
                    }
                    for extension in ("pgf", "pdf", "svg", "png")
                },
            }
            for stem, number, title in FIGURES
        ],
        "supporting_files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (captions, preamble)
        },
        "confirm_raw_data_accessed": False,
        "models_refit": False,
        "registered_intervals_recomputed": False,
    }
    (output_dir / "plot-manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--font-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.input, args.output, args.source_release, args.font_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

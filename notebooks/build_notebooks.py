#!/usr/bin/env python3
"""Build the two self-contained, output-free Colab notebooks.

The generated notebooks embed the release loader (and, for notebook 02, the
decoder) so a reader never needs to clone this repository or install a package.
"""

from __future__ import annotations

import json
import hashlib
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def markdown(text: str) -> dict:
    source = clean(text)
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(("markdown:" + source).encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(text: str, *, hidden: bool = False) -> dict:
    source = clean(text)
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    return {
        "cell_type": "code",
        "id": hashlib.sha1(("code:" + source).encode()).hexdigest()[:12],
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def embedded_source(path: Path, *, strip: tuple[str, ...] = ()) -> str:
    text = path.read_text()
    for item in strip:
        text = text.replace(item, "")
    return text


RELEASE_SOURCE = embedded_source(ROOT / "nma_play" / "release.py")
DECODER_SOURCE = embedded_source(
    ROOT / "nma_play" / "decoder.py",
    strip=("from .release import FeatureMatrix\n",),
)
MODEL_WORKFLOW_SVG = embedded_source(ROOT / "docs" / "v33-model-workflow.svg")

DEPENDENCY_SETUP = r"""
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

requirements = [
    ("pandas", "pandas", "pandas"),
    ("pyarrow", "pyarrow", "pyarrow"),
    ("h5py", "h5py", "h5py"),
    ("sklearn", "scikit-learn", "scikit-learn"),
    ("plotly", "plotly", "plotly"),
    ("ipywidgets", "ipywidgets", "ipywidgets"),
]
install = []
for module, package, distribution in requirements:
    if find_spec(module) is None:
        install.append(package)

if install:
    print("Installing missing Colab dependencies:", ", ".join(install))
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", *install
    ])
else:
    print("All notebook dependencies are already available.")

versions = {}
for _, _, distribution in requirements:
    try:
        versions[distribution] = version(distribution)
    except PackageNotFoundError:
        versions[distribution] = "installed in this cell; restart only if import fails"
print("Runtime versions:", versions)
"""

NEURAL_DEPENDENCY_SETUP = DEPENDENCY_SETUP.replace(
    '    ("plotly", "plotly", "plotly"),\n',
    '    ("matplotlib", "matplotlib", "matplotlib"),\n'
    '    ("seaborn", "seaborn", "seaborn"),\n',
).replace('    ("ipywidgets", "ipywidgets", "ipywidgets"),\n', "")


BEHAVIOR_LOAD = r"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display

try:
    from google.colab import output as colab_output
    colab_output.enable_custom_widget_manager()
except ImportError:
    pass

scan = load_behavioral_scan()
trials = scan.trial_labels.copy()
sessions = scan.session_scan.copy()
eligibility = scan.tables["eligibility"].copy()
persistence = scan.tables["persistence"].copy()
guard = scan.tables["guard_diagnostics"].copy()
yield_sweep = scan.tables["yield_sweep"].copy()

display(HTML(
    "<div style='padding:12px;border-left:5px solid #d97706;background:#fff7ed'>"
    "<b>DEV-only playground.</b> This notebook reads the immutable behavioral "
    f"release <code>{scan.tag}</code>. It never requests CONFIRM data."
    "</div>"
))
print(
    f"Loaded {sessions.behavior_session_id.nunique():,} sessions, "
    f"{trials.mouse_id.nunique():,} mice, and {len(trials):,} trials."
)
"""


BEHAVIOR_OVERVIEW = r"""
def metric_cards(items):
    cards = "".join(
        f"<div style='flex:1;min-width:150px;padding:14px;border:1px solid #e5e7eb;"
        f"border-radius:10px;background:white'><div style='font-size:12px;color:#6b7280'>"
        f"{label}</div><div style='font-size:26px;font-weight:700'>{value}</div></div>"
        for label, value in items
    )
    display(HTML(f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{cards}</div>"))

metric_cards([
    ("DEV mice", f"{trials.mouse_id.nunique():,}"),
    ("Sessions", f"{sessions.behavior_session_id.nunique():,}"),
    ("Trials", f"{len(trials):,}"),
    ("Eligible sessions", f"{int(sessions.behavioral_eligible.sum()):,}"),
])

cohort = px.scatter(
    sessions,
    x="miss_B",
    y="late_hit_B",
    color="project_code",
    symbol="behavioral_eligible",
    size="n_trials",
    hover_data=["behavior_session_id", "mouse_id", "session_type", "eligibility_reasons"],
    labels={"miss_B": "B-engaged misses", "late_hit_B": "B-engaged late hits"},
    title="Behavioral cohort: registered B-engaged support",
    template="plotly_white",
)
cohort.add_vline(x=20, line_dash="dash", line_color="#6b7280")
cohort.add_hline(y=20, line_dash="dash", line_color="#6b7280")
cohort.show()

heat = yield_sweep.pivot(
    index="construct", columns="miss_threshold", values="n_sessions"
).sort_index()
fig = px.imshow(
    heat,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={"x": "Minimum misses", "y": "Construct", "color": "Sessions"},
    title="Session yield under registered selection sweep",
)
fig.update_layout(template="plotly_white")
fig.show()

diag = persistence.merge(
    sessions[["behavior_session_id", "abort_frac", "behavioral_eligible"]],
    on="behavior_session_id",
)
px.scatter(
    diag,
    x="impulsive_frac",
    y="abort_frac",
    size="max_run",
    color="null_p_max_run",
    symbol="behavioral_eligible",
    hover_data=["behavior_session_id", "mouse_id", "n_runs"],
    color_continuous_scale="Viridis_r",
    title="Persistence and abort diagnostics",
    template="plotly_white",
).show()
"""


BEHAVIOR_WIDGETS = r"""
CONSTRUCTS = {
    "B — registered": ("engaged_B", "keep_B"),
    "A — exploratory": ("engaged_A", "keep_A"),
    "A hysteretic — exploratory": ("engaged_A_hysteretic", "keep_A_hysteretic"),
}
OUTCOMES = {
    "All outcomes": None,
    "Late hit": "late_hit",
    "Early hit": "early_hit",
    "Miss": "miss",
    "Abort": "aborted",
}

project_dd = widgets.Dropdown(
    options=["All"] + sorted(sessions.project_code.astype(str).unique()),
    description="Project:",
    style={"description_width": "initial"},
)
mouse_dd = widgets.Dropdown(description="Mouse:", style={"description_width": "initial"})
session_dd = widgets.Dropdown(description="Session:", style={"description_width": "initial"})
construct_dd = widgets.Dropdown(
    options=list(CONSTRUCTS), value="B — registered", description="Engagement:",
    style={"description_width": "initial"},
)
outcome_dd = widgets.Dropdown(
    options=list(OUTCOMES), value="All outcomes", description="Outcome:",
    style={"description_width": "initial"},
)
view = widgets.Output()
download_button = widgets.Button(description="Download filtered CSV", icon="download")
current = {"table": pd.DataFrame()}


def project_sessions():
    frame = sessions
    if project_dd.value != "All":
        frame = frame[frame.project_code.astype(str).eq(project_dd.value)]
    return frame


def sync_mice(*_):
    frame = project_sessions()
    options = sorted(frame.mouse_id.astype(str).unique())
    previous = mouse_dd.value
    mouse_dd.options = options
    if previous in options:
        mouse_dd.value = previous


def sync_sessions(*_):
    frame = project_sessions()
    if mouse_dd.value is not None:
        frame = frame[frame.mouse_id.astype(str).eq(str(mouse_dd.value))]
    options = [
        (f"{int(row.behavior_session_id)} · {row.session_type}", int(row.behavior_session_id))
        for row in frame.sort_values("behavior_session_id").itertuples()
    ]
    previous = session_dd.value
    session_dd.options = options
    values = [value for _, value in options]
    if previous in values:
        session_dd.value = previous


def contiguous_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return []
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def render_session(*_):
    if session_dd.value is None:
        return
    session_id = int(session_dd.value)
    state_col, keep_col = CONSTRUCTS[construct_dd.value]
    frame = trials[trials.behavior_session_id.eq(session_id)].sort_values("trial_index").copy()
    selected = frame.copy()
    outcome_col = OUTCOMES[outcome_dd.value]
    if outcome_col:
        selected = selected[selected[outcome_col].fillna(False)]
    current["table"] = selected
    outcome = np.select(
        [frame.late_hit, frame.early_hit, frame.miss, frame.aborted],
        ["late hit", "early hit", "miss", "abort"],
        default="other",
    )
    frame["outcome"] = outcome
    with view:
        clear_output(wait=True)
        row = sessions[sessions.behavior_session_id.eq(session_id)].iloc[0]
        badge = "eligible" if bool(row.behavioral_eligible) else "not eligible"
        display(HTML(
            f"<h3>Session {session_id}</h3><p><b>{badge}</b> · mouse {row.mouse_id} · "
            f"{row.session_type} · {construct_dd.value}</p>"
        ))
        timeline = go.Figure()
        timeline.add_trace(go.Scatter(
            x=frame.trial_index, y=frame.reward_rate, mode="lines",
            name="reward rate", line={"color": "#2563eb"},
        ))
        timeline.add_trace(go.Scatter(
            x=frame.trial_index, y=frame.bout_rate, mode="lines",
            name="bout rate", line={"color": "#f59e0b"}, yaxis="y2",
        ))
        max_reward = float(np.nanmax(frame.reward_rate)) if frame.reward_rate.notna().any() else 1.0
        palette = {"late hit": "#16a34a", "early hit": "#84cc16", "miss": "#dc2626",
                   "abort": "#7c3aed", "other": "#9ca3af"}
        for label, group in frame.groupby("outcome"):
            timeline.add_trace(go.Scatter(
                x=group.trial_index,
                y=np.full(len(group), max_reward * 1.04),
                mode="markers",
                marker={"size": 6, "color": palette[label]},
                name=label,
                hovertext=[f"trial {int(value)}" for value in group.trial_id],
            ))
        engaged = frame[state_col].fillna(False).to_numpy(bool)
        for start, stop in contiguous_segments(engaged):
            timeline.add_vrect(
                x0=float(frame.trial_index.iloc[start]) - 0.5,
                x1=float(frame.trial_index.iloc[stop - 1]) + 0.5,
                fillcolor="#22c55e", opacity=0.08, line_width=0,
            )
        removed = frame[~frame[keep_col].fillna(False) & frame.go.fillna(False)]
        if len(removed):
            timeline.add_trace(go.Scatter(
                x=removed.trial_index,
                y=np.full(len(removed), -0.03 * max_reward),
                mode="markers",
                marker={"symbol": "x", "color": "#111827", "size": 7},
                name="guard-excluded go",
            ))
        timeline.update_layout(
            title="Trial timeline (green shading = engaged)",
            xaxis_title="Raw trial index",
            yaxis={"title": "Reward rate"},
            yaxis2={"title": "Bout rate", "overlaying": "y", "side": "right"},
            template="plotly_white",
            legend={"orientation": "h"},
        )
        timeline.show()

        composition = (
            frame.groupby([state_col, "outcome"], dropna=False).size()
            .rename("trials").reset_index()
        )
        composition[state_col] = composition[state_col].map(
            {True: "engaged", False: "disengaged"}
        )
        px.bar(
            composition, x=state_col, y="trials", color="outcome",
            barmode="stack", color_discrete_map=palette,
            title="Outcome composition by engagement state",
            template="plotly_white",
        ).show()

        reason = eligibility.loc[
            eligibility.behavior_session_id.eq(session_id), "eligibility_reasons"
        ]
        reason_text = reason.iloc[0] if len(reason) and str(reason.iloc[0]) else "none"
        display(HTML(
            f"<p><b>Eligibility reasons:</b> {reason_text}<br>"
            f"<b>Visible rows after outcome filter:</b> {len(selected):,}</p>"
        ))
        display(selected.head(30))


def download_filtered(_):
    path = f"behavior-session-{int(session_dd.value)}-filtered.csv"
    current["table"].to_csv(path, index=False)
    try:
        from google.colab import files
        files.download(path)
    except ImportError:
        print(f"Saved {path}")


project_dd.observe(sync_mice, names="value")
mouse_dd.observe(sync_sessions, names="value")
session_dd.observe(render_session, names="value")
construct_dd.observe(render_session, names="value")
outcome_dd.observe(render_session, names="value")
download_button.on_click(download_filtered)
sync_mice()
sync_sessions()
render_session()

display(widgets.VBox([
    widgets.HTML("<h2>Interactive session explorer</h2>"),
    widgets.HBox([project_dd, mouse_dd, session_dd]),
    widgets.HBox([construct_dd, outcome_dd, download_button]),
    view,
]))
"""


NEURAL_LOAD = r"""
from contextlib import nullcontext
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook")


def display_matplotlib_figure(figure):
    # Emit a native PNG payload that is reliable in Colab.
    try:
        figure.tight_layout()
    except Exception:
        pass
    display(figure)
    plt.close(figure)


cache = load_feature_cache()
index = cache.index.copy()
display(HTML(
    "<div style='padding:12px;border-left:5px solid #d97706;background:#fff7ed'>"
    "<b>DEV-only educational explorer.</b> It uses fold-independent features from "
    f"<code>{FEATURE_TAG}</code>, never downloads raw neural bundles, and never "
    "requests CONFIRM data.</div>"
))
print(
    f"Loaded {len(index):,} active experiments across "
    f"{index.ophys_container_id.nunique():,} containers and "
    f"{index.mouse_id.nunique():,} mice."
)


@lru_cache(maxsize=24)
def experiment_data(experiment_id, feature):
    experiment_id = int(experiment_id)
    matrix = cache.matrix(experiment_id, feature)
    labels = cache.labels(experiment_id).set_index("trial_id").reindex(
        matrix.trial_ids
    ).reset_index()
    q2 = cache.q2(experiment_id).set_index("trial_id").reindex(
        matrix.trial_ids
    ).reset_index()
    return matrix, labels, q2


def outcome_labels(labels):
    return np.select(
        [labels.late_hit, labels.early_hit, labels.miss, labels.aborted],
        ["late hit", "early hit", "miss", "abort"],
        default="other",
    )
"""


NEURAL_COHORT = r"""
def metric_cards(items):
    cards = "".join(
        f"<div style='flex:1;min-width:150px;padding:14px;border:1px solid #e5e7eb;"
        f"border-radius:10px;background:white'><div style='font-size:12px;color:#6b7280'>"
        f"{label}</div><div style='font-size:26px;font-weight:700'>{value}</div></div>"
        for label, value in items
    )
    display(HTML(f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{cards}</div>"))

metric_cards([
    ("Active experiments", f"{len(index):,}"),
    ("DEV mice", f"{index.mouse_id.nunique():,}"),
    ("Aligned trials", f"{index.n_trials.sum():,}"),
    ("Cells per experiment", f"{index.n_cells.min()}–{index.n_cells.max()}"),
])

figure, axis = plt.subplots(figsize=(11, 6))
sns.scatterplot(
    data=index,
    x="n_trials",
    y="n_cells",
    hue="project_code",
    style="session_type",
    size="n_cells",
    sizes=(70, 360),
    alpha=0.8,
    ax=axis,
)
axis.set(
    title="Feature-cache coverage",
    xlabel="Aligned trials",
    ylabel="Cells",
)
axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
display_matplotlib_figure(figure)

hierarchy = index.sort_values(
    ["mouse_id", "ophys_container_id", "ophys_experiment_id"]
).copy()
hierarchy["row_label"] = (
    hierarchy.mouse_id.astype(str)
    + " / "
    + hierarchy.ophys_container_id.astype(str)
)
hierarchy["experiment_order"] = (
    hierarchy.groupby(["mouse_id", "ophys_container_id"]).cumcount() + 1
)
figure, axis = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=hierarchy,
    x="experiment_order",
    y="row_label",
    hue="session_type",
    style="session_type",
    size="n_trials",
    sizes=(80, 360),
    alpha=0.85,
    ax=axis,
)
axis.set(
    title="Mouse → container → experiment hierarchy (size = trials)",
    xlabel="Experiment order within container",
    ylabel="Mouse / container",
)
axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
display_matplotlib_figure(figure)

coverage = pd.crosstab(index.mouse_id.astype(str), index.session_type)
figure, axis = plt.subplots(figsize=(12, 5))
sns.heatmap(
    coverage,
    annot=True,
    fmt="d",
    cmap="crest",
    linewidths=0.5,
    cbar_kws={"label": "Experiments"},
    ax=axis,
)
axis.set(
    title="Session-type coverage by mouse",
    xlabel="Session type",
    ylabel="Mouse",
)
display_matplotlib_figure(figure)
"""


NEURAL_STATIC_ANALYSIS = r"""
FEATURE_LABELS = {
    "events_baselined_post": "Events · baseline-subtracted · [0, 0.30)s",
    "events_unbaselined_pre": "Events · unbaselined · [-1, 0)s",
    "events_unbaselined_post": "Events · unbaselined · [0, 0.30)s",
    "events_baselined_full_pre": "Events · baseline-subtracted · full pre",
    "dff_baselined_post": "dF/F · baseline-subtracted · [0, 0.30)s",
}

# Edit this compact configuration block, then rerun the cell.
SELECTED_EXPERIMENT_ID = int(index.iloc[0].ophys_experiment_id)
REPRESENTATION = "events_baselined_post"
HEATMAP_SCALE = "z"  # "z" or "raw"
CELL_ORDER = "effect"  # "effect" or "variance"
PCA_COLOR = "outcome"  # "outcome", "engaged_B", or "session_position"

DECODER_REPRESENTATION = "events_baselined_post"
DECODER_K = 50
DECODER_C = 1e-4
DECODER_SEEDS = 10
DECODER_CV = "blocked"  # "blocked" registered; "random" exploratory

if SELECTED_EXPERIMENT_ID not in set(index.ophys_experiment_id.astype(int)):
    raise ValueError(
        f"SELECTED_EXPERIMENT_ID={SELECTED_EXPERIMENT_ID} is not in the DEV cache."
    )
if REPRESENTATION not in FEATURE_LABELS:
    raise ValueError(f"Unknown REPRESENTATION: {REPRESENTATION}")


def selected_effect(values, labels):
    late = labels.late_hit.fillna(False).to_numpy(bool)
    miss = labels.miss.fillna(False).to_numpy(bool)
    engaged = labels.engaged_B.fillna(False).to_numpy(bool)
    keep = labels.keep_B.fillna(False).to_numpy(bool)
    left = values[late & engaged & keep]
    right = values[miss & engaged & keep]
    if not len(left) or not len(right):
        return np.zeros(values.shape[1])
    pooled = np.sqrt((np.nanvar(left, axis=0) + np.nanvar(right, axis=0)) / 2)
    return np.divide(
        np.nanmean(left, axis=0) - np.nanmean(right, axis=0),
        pooled,
        out=np.zeros(values.shape[1]),
        where=np.isfinite(pooled) & (pooled > 0),
    )


def bootstrap_population(frame, groups, n_boot=500):
    rng = np.random.default_rng(20260717)
    rows = []
    for label in ["late hit", "miss", "early hit", "abort", "other"]:
        values = frame[np.asarray(groups) == label]
        if not len(values):
            continue
        means = np.array([
            rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)
        ])
        rows.append({
            "outcome": label,
            "mean": float(np.mean(values)),
            "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)),
            "n": len(values),
        })
    return pd.DataFrame(rows)


def render_matrix():
    oeid = SELECTED_EXPERIMENT_ID
    matrix, labels, _ = experiment_data(oeid, REPRESENTATION)
    values = matrix.values.astype(float)
    effect = selected_effect(values, labels)
    order = (
        np.argsort(np.abs(effect))[::-1]
        if CELL_ORDER == "effect"
        else np.argsort(np.nanvar(values, axis=0))[::-1]
    )
    order = order[: min(200, len(order))]
    trial_outcome = outcome_labels(labels)
    trial_order = np.lexsort((labels.trial_index.to_numpy(), trial_outcome))
    shown = values[trial_order][:, order]
    if HEATMAP_SCALE == "z":
        mean = np.nanmean(shown, axis=0)
        sd = np.nanstd(shown, axis=0)
        shown = np.divide(shown - mean, sd, out=np.zeros_like(shown), where=sd > 0)
    population = np.nanmean(values, axis=1)
    summary = bootstrap_population(population, trial_outcome)
    with nullcontext():
        display(HTML(
            f"<h3>Experiment {oeid}</h3><p>{FEATURE_LABELS[REPRESENTATION]} · "
            f"{len(values):,} trials × {values.shape[1]:,} cells. "
            f"Heatmap shows the top {len(order)} cells.</p>"
        ))
        heat_frame = pd.DataFrame(
            shown,
            index=labels.trial_index.to_numpy()[trial_order],
            columns=[str(int(matrix.cell_ids[i])) for i in order],
        )
        figure, axis = plt.subplots(figsize=(15, 8))
        sns.heatmap(
            heat_frame,
            cmap="vlag",
            center=0,
            xticklabels=max(1, len(order) // 20),
            yticklabels=max(1, len(heat_frame) // 25),
            cbar_kws={
                "label": "Cell z-score" if HEATMAP_SCALE == "z" else "Response"
            },
            ax=axis,
        )
        axis.set(
            title="Trial × cell response matrix",
            xlabel="Cell specimen ID",
            ylabel="Raw trial index (grouped by outcome)",
        )
        display_matplotlib_figure(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        lower = summary["mean"] - summary.low
        upper = summary.high - summary["mean"]
        axis.bar(
            summary.outcome,
            summary["mean"],
            yerr=np.vstack([lower, upper]),
            color=sns.color_palette("deep", len(summary)),
            capsize=4,
        )
        for position, row in enumerate(summary.itertuples()):
            axis.annotate(
                f"n={row.n}",
                (position, row.high),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
            )
        axis.set(
            title="Population response by outcome (bootstrap 95% CI)",
            xlabel="Outcome",
            ylabel="Mean across cells",
        )
        display_matplotlib_figure(figure)

        effect_table = pd.DataFrame({
            "cell_id": matrix.cell_ids,
            "effect": effect,
            "variance": np.nanvar(values, axis=0),
        }).sort_values("effect")
        effect_table["rank"] = np.arange(len(effect_table))
        figure, axis = plt.subplots(figsize=(11, 5))
        points = axis.scatter(
            effect_table["rank"],
            effect_table["effect"],
            c=effect_table["variance"],
            cmap="viridis",
            s=28,
            alpha=0.8,
        )
        figure.colorbar(points, ax=axis, label="Variance")
        axis.axhline(0, color="0.45", linewidth=1)
        axis.set(
            title="Per-cell standardized late-hit − miss effect",
            xlabel="Effect rank",
            ylabel="Standardized effect",
        )
        display_matplotlib_figure(figure)

        distribution = pd.DataFrame({
            "population_response": population,
            "outcome": trial_outcome,
        })
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.violinplot(
            data=distribution,
            x="outcome",
            y="population_response",
            hue="outcome",
            inner="box",
            cut=0,
            legend=False,
            ax=axis,
        )
        axis.set(
            title="Trial-level population-response distributions",
            xlabel="Outcome",
            ylabel="Population response",
        )
        display_matplotlib_figure(figure)

        comparisons = [
            ("events_baselined_post", "dff_baselined_post"),
            ("events_unbaselined_pre", "events_unbaselined_post"),
        ]
        titles = ["Events post vs dF/F post", "Events pre vs events post"]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        for axis, title, (left_name, right_name) in zip(axes, titles, comparisons):
            left = experiment_data(oeid, left_name)[0].values.mean(axis=1)
            right = experiment_data(oeid, right_name)[0].values.mean(axis=1)
            correlation = np.corrcoef(left, right)[0, 1]
            sns.scatterplot(x=left, y=right, s=25, alpha=0.55, ax=axis)
            axis.set(
                title=f"{title} · r={correlation:.2f}",
                xlabel="Left population response",
                ylabel="Right population response",
            )
        figure.suptitle("Representation comparisons")
        display_matplotlib_figure(figure)


def render_geometry():
    oeid = SELECTED_EXPERIMENT_ID
    matrix, labels, q2 = experiment_data(oeid, REPRESENTATION)
    values = matrix.values.astype(float)
    finite = np.isfinite(values).all(axis=1)
    if finite.sum() < 3:
        print("PCA nonestimable: fewer than three finite trials.")
        return
    scaled = StandardScaler().fit_transform(values[finite])
    n_components = min(10, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components).fit(scaled)
    scores = pca.transform(scaled)
    plot = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
    plot["outcome"] = outcome_labels(labels)[finite]
    plot["engaged_B"] = labels.engaged_B.fillna(False).to_numpy()[finite].astype(str)
    plot["session_position"] = q2.session_position.to_numpy()[finite]
    population = np.nanmean(values, axis=1)
    joint = q2.copy()
    joint["population_response"] = population
    joint["outcome"] = outcome_labels(labels)
    with nullcontext():
        figure, axis = plt.subplots(figsize=(9, 6))
        color_column = PCA_COLOR
        sns.scatterplot(
            data=plot,
            x="PC1",
            y="PC2",
            hue=color_column,
            palette="viridis" if color_column == "session_position" else "deep",
            s=42,
            alpha=0.75,
            ax=axis,
        )
        axis.set_title(f"Trial geometry · {FEATURE_LABELS[REPRESENTATION]}")
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        display_matplotlib_figure(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x=np.arange(1, n_components + 1),
            y=pca.explained_variance_ratio_,
            color=sns.color_palette()[0],
            ax=axis,
        )
        axis.set(
            title="PCA scree plot",
            xlabel="Principal component",
            ylabel="Explained variance ratio",
        )
        display_matplotlib_figure(figure)

        covariates = [
            ("pre_change_pupil", "Pre-change pupil"),
            ("pre_change_running", "Pre-change running"),
            ("session_position", "Session position"),
        ]
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, (name, label_text) in zip(axes, covariates):
            good = joint[name].notna() & joint.population_response.notna()
            sns.regplot(
                data=joint.loc[good],
                x=name,
                y="population_response",
                scatter_kws={"s": 18, "alpha": 0.45},
                line_kws={"color": "#c44e52"},
                ax=axis,
            )
            axis.set_title(label_text)
        figure.suptitle("Behavior–neural relationships (no imputation)")
        display_matplotlib_figure(figure)

        missing = (
            q2.isna().mean().sort_values(ascending=False).rename("missing_fraction")
            .reset_index().rename(columns={"index": "covariate"})
        )
        figure_height = max(5, 0.3 * len(missing))
        figure, axis = plt.subplots(figsize=(10, figure_height))
        sns.barplot(
            data=missing,
            x="missing_fraction",
            y="covariate",
            color=sns.color_palette()[0],
            ax=axis,
        )
        axis.set(
            title="Q2 covariate missingness",
            xlabel="Missing fraction",
            ylabel="Covariate",
        )
        display_matplotlib_figure(figure)


def render_decoder():
    oeid = SELECTED_EXPERIMENT_ID
    feature = DECODER_REPRESENTATION
    k = DECODER_K
    C = DECODER_C
    n_seeds = DECODER_SEEDS
    cv = DECODER_CV
    config = DecoderConfig(k=k, C=C, n_seeds=n_seeds, cv=cv)
    label = "REGISTERED DEFAULTS" if not config.exploratory else "EXPLORATORY"
    color = "#166534" if not config.exploratory else "#b45309"
    display(HTML(
        f"<div style='padding:8px;border-left:4px solid {color}'><b>{label}</b> · "
        "Single-experiment educational analysis; not an authoritative mouse-level result."
        "</div>"
    ))
    print("Fitting train-only scalers and logistic models…")
    matrix, labels, _ = experiment_data(oeid, feature)
    result = run_q1_decoder(matrix, labels, config)
    with nullcontext():
        display(HTML(
            f"<div style='padding:8px;border-left:4px solid {color}'><b>{label}</b> · "
            "Single-experiment educational analysis; not an authoritative mouse-level result."
            "</div>"
        ))
        if result.status != "estimable":
            display(HTML(
                f"<h3>Decoder nonestimable</h3><code>{result.reason}</code>"
            ))
            return
        display(HTML(
            f"<h3>Mean seed AUC: {result.mean_auc:.3f}</h3>"
            f"<p>{len(result.oof):,} registered eligible trials · "
            f"{len(result.seed_metrics):,} deterministic seeds.</p>"
        ))
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=result.seed_metrics,
            x="seed",
            y="auc",
            color=sns.color_palette()[0],
            ax=axis,
        )
        axis.axhline(0.5, linestyle="--", color="0.4")
        axis.set(
            title="Seed-level OOF AUC",
            xlabel="Seed",
            ylabel="OOF AUC",
            ylim=(0, 1),
        )
        display_matplotlib_figure(figure)

        fpr, tpr, _ = roc_curve(result.oof.y, result.oof.mean_score)
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(fpr, tpr, linewidth=2, label="Mean OOF score")
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
        axis.set(
            title="OOF ROC across mean seed scores",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(frameon=False)
        display_matplotlib_figure(figure)

        fold = result.fold_metrics[
            result.fold_metrics.seed.eq(0)
        ].sort_values("fold")
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(
            fold.fold,
            fold.test_negative,
            label="miss",
            color=sns.color_palette("deep")[0],
        )
        axis.bar(
            fold.fold,
            fold.test_positive,
            bottom=fold.test_negative,
            label="late hit",
            color=sns.color_palette("deep")[1],
        )
        axis.set(
            title="Seed 0 temporal test-fold support",
            xlabel="Fold",
            ylabel="Trials",
        )
        axis.legend(frameon=False)
        display_matplotlib_figure(figure)

        temporal = result.oof.sort_values("trial_index").copy()
        temporal["outcome"] = temporal.y.map({0: "miss", 1: "late hit"})
        figure, axis = plt.subplots(figsize=(12, 5))
        sns.scatterplot(
            data=temporal,
            x="trial_index",
            y="mean_score",
            hue="outcome",
            style="outcome",
            s=32,
            alpha=0.75,
            ax=axis,
        )
        axis.axhline(0.5, linestyle="--", color="0.5")
        axis.set(
            title="OOF decision score over raw trial order",
            xlabel="Raw trial index",
            ylabel="Mean OOF score",
        )
        axis.legend(frameon=False)
        display_matplotlib_figure(figure)

        cells = result.cell_summary[
            result.cell_summary.selection_frequency.gt(0)
        ].sort_values(
            ["selection_frequency", "mean_abs_coefficient"], ascending=False
        ).head(60)
        figure, axis = plt.subplots(figsize=(10, 6))
        max_abs = max(float(cells.median_coefficient.abs().max()), 1e-12)
        sns.scatterplot(
            data=cells,
            x="selection_frequency",
            y="median_coefficient",
            size="mean_abs_coefficient",
            hue="median_coefficient",
            palette="vlag",
            hue_norm=(-max_abs, max_abs),
            sizes=(40, 320),
            alpha=0.8,
            ax=axis,
        )
        for row in cells.head(10).itertuples():
            axis.annotate(
                str(int(row.cell_id)),
                (row.selection_frequency, row.median_coefficient),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axhline(0, color="0.45", linewidth=1)
        axis.set(
            title="Cell selection frequency and standardized coefficients",
            xlabel="Selection frequency",
            ylabel="Median standardized coefficient",
        )
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        display_matplotlib_figure(figure)


selected_row = index.loc[
    index.ophys_experiment_id.astype(int).eq(SELECTED_EXPERIMENT_ID)
].iloc[0]
display(HTML(
    "<h2>Static single-experiment analysis</h2>"
    f"<p>Mouse {int(selected_row.mouse_id)} · "
    f"container {int(selected_row.ophys_container_id)} · "
    f"experiment {SELECTED_EXPERIMENT_ID} · "
    f"{selected_row.session_type}</p>"
    "<p>Edit the configuration block at the top of this cell and rerun it "
    "to inspect a different experiment or representation.</p>"
))

render_matrix()
render_geometry()
render_decoder()
"""


behavior_notebook = notebook([
    markdown(
        """
        # Behavioral DEV Playground

        **Data:** immutable DEV behavioral scan only; no CONFIRM access.

        [Release provenance](https://github.com/c-lin-chunyi/nma-project-data-analysis/releases/tag/behavioral-v3.1-29482141350)

        Run the cells from top to bottom. This notebook is self-contained: it
        does not clone a repository. Its first code cell installs only packages
        that are completely missing and never upgrades Colab's scientific stack.
        """
    ),
    markdown("## 0. Prepare the Colab runtime"),
    code(DEPENDENCY_SETUP),
    markdown("## 1. Verified release loader\n\nThe cell below is embedded for one-file Colab use."),
    code(RELEASE_SOURCE, hidden=True),
    markdown("## 2. Load the compact behavioral scan"),
    code(BEHAVIOR_LOAD),
    markdown("## 3. Cohort overview"),
    code(BEHAVIOR_OVERVIEW),
    markdown("## 4. Interactive session explorer"),
    code(BEHAVIOR_WIDGETS),
])


neural_notebook = notebook([
    markdown(
        """
        # Neural Feature Explorer & Decoder Lab

        **Data:** immutable, fold-independent DEV feature cache only; no raw
        neural bundle and no CONFIRM access.

        [Release provenance](https://github.com/c-lin-chunyi/nma-project-data-analysis/releases/tag/neural-dev-features-v1-29482249873)

        Run the cells from top to bottom. This notebook is self-contained: it
        does not clone a repository. Its first code cell installs only missing
        dependencies. The complete feature cache is about 24.3 MiB compressed.
        """
    ),
    markdown("## 0. Prepare the Colab runtime"),
    code(NEURAL_DEPENDENCY_SETUP),
    markdown("## 1. Verified release loader\n\nEmbedded for one-file Colab use."),
    code(RELEASE_SOURCE, hidden=True),
    markdown("## 2. Typed single-experiment decoder\n\nEmbedded for one-file Colab use."),
    code(DECODER_SOURCE, hidden=True),
    markdown("## 3. Load and verify all ten feature shards"),
    code(NEURAL_LOAD),
    markdown("## 4. Cohort browser"),
    code(NEURAL_COHORT),
    markdown(
        "## 5. Logistic-model workflow\n\n"
        "The registered analysis uses regularized logistic models with "
        "temporally isolated evaluation and training-only calibration. "
        "The diagram emphasizes model inputs, fitting, and metrics."
    ),
    markdown(MODEL_WORKFLOW_SVG),
    markdown("## 6. Static single-experiment analysis"),
    code(NEURAL_STATIC_ANALYSIS),
])


def main() -> None:
    outputs = {
        "01_behavioral_playground.ipynb": behavior_notebook,
        "02_neural_feature_explorer.ipynb": neural_notebook,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

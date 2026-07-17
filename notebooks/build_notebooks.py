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
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

try:
    from google.colab import output as colab_output
    colab_output.enable_custom_widget_manager()
except ImportError:
    pass

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

px.scatter(
    index,
    x="n_trials",
    y="n_cells",
    color="project_code",
    symbol="session_type",
    size="n_cells",
    hover_data=["ophys_experiment_id", "ophys_container_id", "mouse_id"],
    title="Feature-cache coverage",
    template="plotly_white",
).show()

hierarchy = index.copy()
for column in ["mouse_id", "ophys_container_id", "ophys_experiment_id"]:
    hierarchy[column] = hierarchy[column].astype(str)
px.sunburst(
    hierarchy,
    path=["mouse_id", "ophys_container_id", "ophys_experiment_id"],
    values="n_trials",
    color="session_type",
    title="Mouse → container → experiment hierarchy (area = trials)",
).show()

coverage = pd.crosstab(index.mouse_id.astype(str), index.session_type)
fig = px.imshow(
    coverage,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Teal",
    labels={"x": "Session type", "y": "Mouse", "color": "Experiments"},
    title="Session-type coverage by mouse",
)
fig.update_layout(template="plotly_white")
fig.show()
"""


NEURAL_WIDGETS = r"""
FEATURE_LABELS = {
    "events_baselined_post": "Events · baseline-subtracted · [0, 0.30)s",
    "events_unbaselined_pre": "Events · unbaselined · [-1, 0)s",
    "events_unbaselined_post": "Events · unbaselined · [0, 0.30)s",
    "events_baselined_full_pre": "Events · baseline-subtracted · full pre",
    "dff_baselined_post": "dF/F · baseline-subtracted · [0, 0.30)s",
}

mouse_dd = widgets.Dropdown(
    options=sorted(index.mouse_id.astype(int).unique()),
    description="Mouse:", style={"description_width": "initial"},
)
container_dd = widgets.Dropdown(
    description="Container:", style={"description_width": "initial"}
)
experiment_dd = widgets.Dropdown(
    description="Experiment:", style={"description_width": "initial"}
)
feature_dd = widgets.Dropdown(
    options=[(label, name) for name, label in FEATURE_LABELS.items()],
    value="events_baselined_post",
    description="Representation:", style={"description_width": "initial"},
    layout={"width": "430px"},
)
scale_dd = widgets.ToggleButtons(
    options=[("Cell z-score", "z"), ("Raw", "raw")],
    value="z", description="Heatmap:",
)
cell_sort_dd = widgets.Dropdown(
    options=[("Late-hit effect", "effect"), ("Variance", "variance")],
    value="effect", description="Cell order:", style={"description_width": "initial"},
)
matrix_out = widgets.Output()
geometry_out = widgets.Output()
decoder_out = widgets.Output()


def sync_containers(*_):
    rows = index[index.mouse_id.astype(int).eq(int(mouse_dd.value))]
    options = [int(value) for value in sorted(rows.ophys_container_id.astype(int).unique())]
    previous = container_dd.value
    container_dd.options = options
    container_dd.value = previous if previous in options else (options[0] if options else None)


def sync_experiments(*_):
    if container_dd.value is None:
        experiment_dd.options = []
        return
    rows = index[index.ophys_container_id.astype(int).eq(int(container_dd.value))]
    options = [
        (f"{int(row.ophys_experiment_id)} · {row.session_type}", int(row.ophys_experiment_id))
        for row in rows.sort_values("ophys_experiment_id").itertuples()
    ]
    previous = experiment_dd.value
    experiment_dd.options = options
    values = [value for _, value in options]
    experiment_dd.value = previous if previous in values else (values[0] if values else None)
    update_k_options()


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


def render_matrix(*_):
    if experiment_dd.value is None:
        return
    oeid = int(experiment_dd.value)
    matrix, labels, _ = experiment_data(oeid, feature_dd.value)
    values = matrix.values.astype(float)
    effect = selected_effect(values, labels)
    order = (
        np.argsort(np.abs(effect))[::-1]
        if cell_sort_dd.value == "effect"
        else np.argsort(np.nanvar(values, axis=0))[::-1]
    )
    order = order[: min(200, len(order))]
    trial_outcome = outcome_labels(labels)
    trial_order = np.lexsort((labels.trial_index.to_numpy(), trial_outcome))
    shown = values[trial_order][:, order]
    if scale_dd.value == "z":
        mean = np.nanmean(shown, axis=0)
        sd = np.nanstd(shown, axis=0)
        shown = np.divide(shown - mean, sd, out=np.zeros_like(shown), where=sd > 0)
    population = np.nanmean(values, axis=1)
    summary = bootstrap_population(population, trial_outcome)
    with matrix_out:
        clear_output(wait=True)
        display(HTML(
            f"<h3>Experiment {oeid}</h3><p>{FEATURE_LABELS[feature_dd.value]} · "
            f"{len(values):,} trials × {values.shape[1]:,} cells. "
            f"Heatmap shows the top {len(order)} cells.</p>"
        ))
        heat = go.Figure(go.Heatmap(
            z=shown,
            x=[str(int(matrix.cell_ids[i])) for i in order],
            y=labels.trial_index.to_numpy()[trial_order],
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "z" if scale_dd.value == "z" else "response"},
        ))
        heat.update_layout(
            title="Trial × cell response matrix",
            xaxis_title="Cell specimen ID",
            yaxis_title="Raw trial index (grouped by outcome)",
            template="plotly_white",
            height=620,
        )
        heat.show()

        error = go.Figure()
        error.add_trace(go.Bar(
            x=summary.outcome,
            y=summary["mean"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": summary.high - summary["mean"],
                "arrayminus": summary["mean"] - summary.low,
            },
            customdata=summary[["n"]],
            hovertemplate="%{x}<br>mean=%{y:.3f}<br>n=%{customdata[0]}<extra></extra>",
        ))
        error.update_layout(
            title="Population response by outcome (bootstrap 95% CI)",
            yaxis_title="Mean across cells",
            template="plotly_white",
        )
        error.show()

        effect_table = pd.DataFrame({
            "cell_id": matrix.cell_ids,
            "effect": effect,
            "variance": np.nanvar(values, axis=0),
        }).sort_values("effect")
        effect_table["rank"] = np.arange(len(effect_table))
        px.scatter(
            effect_table,
            x="rank",
            y="effect",
            color="variance",
            hover_data=["cell_id"],
            color_continuous_scale="Viridis",
            title="Per-cell standardized late-hit − miss effect",
            template="plotly_white",
        ).show()
        distribution = pd.DataFrame({
            "population_response": population,
            "outcome": trial_outcome,
        })
        px.violin(
            distribution,
            x="outcome",
            y="population_response",
            color="outcome",
            box=True,
            points="outliers",
            title="Trial-level population-response distributions",
            template="plotly_white",
        ).show()

        comparisons = [
            ("events_baselined_post", "dff_baselined_post"),
            ("events_unbaselined_pre", "events_unbaselined_post"),
        ]
        compare = make_subplots(rows=1, cols=2, subplot_titles=[
            "Events post vs dF/F post", "Events pre vs events post"
        ])
        for col, (left_name, right_name) in enumerate(comparisons, start=1):
            left = experiment_data(oeid, left_name)[0].values.mean(axis=1)
            right = experiment_data(oeid, right_name)[0].values.mean(axis=1)
            correlation = np.corrcoef(left, right)[0, 1]
            compare.add_trace(go.Scatter(
                x=left, y=right, mode="markers",
                marker={"size": 5, "opacity": 0.55},
                name=f"r={correlation:.2f}", showlegend=True,
            ), row=1, col=col)
        compare.update_xaxes(title_text="Left population response")
        compare.update_yaxes(title_text="Right population response")
        compare.update_layout(
            title="Representation comparisons", template="plotly_white"
        )
        compare.show()


geometry_color_dd = widgets.Dropdown(
    options=["outcome", "engaged_B", "session_position"],
    value="outcome", description="PCA color:", style={"description_width": "initial"},
)


def render_geometry(*_):
    if experiment_dd.value is None:
        return
    oeid = int(experiment_dd.value)
    matrix, labels, q2 = experiment_data(oeid, feature_dd.value)
    values = matrix.values.astype(float)
    finite = np.isfinite(values).all(axis=1)
    if finite.sum() < 3:
        with geometry_out:
            clear_output()
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
    with geometry_out:
        clear_output(wait=True)
        px.scatter(
            plot,
            x="PC1",
            y="PC2",
            color=geometry_color_dd.value,
            hover_data=["outcome", "engaged_B", "session_position"],
            title=f"Trial geometry · {FEATURE_LABELS[feature_dd.value]}",
            template="plotly_white",
        ).show()
        px.bar(
            x=np.arange(1, n_components + 1),
            y=pca.explained_variance_ratio_,
            labels={"x": "Principal component", "y": "Explained variance ratio"},
            title="PCA scree plot",
            template="plotly_white",
        ).show()
        covariates = [
            ("pre_change_pupil", "Pre-change pupil"),
            ("pre_change_running", "Pre-change running"),
            ("session_position", "Session position"),
        ]
        relation = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in covariates])
        for col, (name, _) in enumerate(covariates, start=1):
            good = joint[name].notna() & joint.population_response.notna()
            relation.add_trace(go.Scatter(
                x=joint.loc[good, name],
                y=joint.loc[good, "population_response"],
                mode="markers",
                marker={"size": 5, "opacity": 0.5},
                text=joint.loc[good, "outcome"],
                showlegend=False,
            ), row=1, col=col)
        relation.update_yaxes(title_text="Population response", row=1, col=1)
        relation.update_layout(
            title="Behavior–neural relationships (no imputation)",
            template="plotly_white",
        )
        relation.show()
        missing = (
            q2.isna().mean().sort_values(ascending=False).rename("missing_fraction")
            .reset_index(names="covariate")
        )
        px.bar(
            missing,
            x="missing_fraction",
            y="covariate",
            orientation="h",
            title="Q2 covariate missingness",
            template="plotly_white",
        ).show()


decoder_feature_dd = widgets.Dropdown(
    options=[(label, name) for name, label in FEATURE_LABELS.items()],
    value="events_baselined_post",
    description="Representation:", style={"description_width": "initial"},
    layout={"width": "430px"},
)
k_dd = widgets.Dropdown(description="K:", style={"description_width": "initial"})
c_dd = widgets.Dropdown(
    options=[("1e-5", 1e-5), ("1e-4 · registered", 1e-4), ("1e-3", 1e-3),
             ("1e-2", 1e-2), ("1e-1", 1e-1), ("1", 1.0), ("10", 10.0)],
    value=1e-4, description="C:", style={"description_width": "initial"},
)
seed_dd = widgets.Dropdown(
    options=[1, 5, 10], value=10, description="Seeds:",
    style={"description_width": "initial"},
)
cv_dd = widgets.Dropdown(
    options=[("Blocked + purge · registered", "blocked"),
             ("Random stratified · diagnostic", "random")],
    value="blocked", description="CV:", style={"description_width": "initial"},
)
run_button = widgets.Button(
    description="Run decoder", button_style="primary", icon="play"
)
decoder_cache = {}


def update_k_options(*_):
    if experiment_dd.value is None:
        return
    n_cells = int(
        index.loc[
            index.ophys_experiment_id.astype(int).eq(int(experiment_dd.value)), "n_cells"
        ].iloc[0]
    )
    options = [(str(k), k) for k in (20, 50, 100) if k <= n_cells]
    options.append((f"All ({n_cells}) · exploratory", None))
    k_dd.options = options
    if 50 <= n_cells:
        k_dd.value = 50


def decoder_key():
    return (
        int(experiment_dd.value), decoder_feature_dd.value, k_dd.value,
        float(c_dd.value), int(seed_dd.value), cv_dd.value,
    )


def render_decoder(_=None):
    key = decoder_key()
    oeid, feature, k, C, n_seeds, cv = key
    config = DecoderConfig(k=k, C=C, n_seeds=n_seeds, cv=cv)
    with decoder_out:
        clear_output(wait=True)
        label = "REGISTERED DEFAULTS" if not config.exploratory else "EXPLORATORY"
        color = "#166534" if not config.exploratory else "#b45309"
        display(HTML(
            f"<div style='padding:8px;border-left:4px solid {color}'><b>{label}</b> · "
            "Single-experiment educational analysis; not an authoritative mouse-level result."
            "</div>"
        ))
        print("Fitting train-only scalers and logistic models…")
    if key not in decoder_cache:
        matrix, labels, _ = experiment_data(oeid, feature)
        decoder_cache[key] = run_q1_decoder(matrix, labels, config)
    result = decoder_cache[key]
    with decoder_out:
        clear_output(wait=True)
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
        px.bar(
            result.seed_metrics,
            x="seed",
            y="auc",
            range_y=[0, 1],
            title="Seed-level OOF AUC",
            template="plotly_white",
        ).add_hline(y=0.5, line_dash="dash").show()

        fpr, tpr, _ = roc_curve(result.oof.y, result.oof.mean_score)
        roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="Mean OOF score"))
        roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"dash": "dash", "color": "#6b7280"}, name="Chance",
        ))
        roc.update_layout(
            title="OOF ROC across mean seed scores",
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
            template="plotly_white",
        )
        roc.show()

        fold = result.fold_metrics[result.fold_metrics.seed.eq(0)].melt(
            id_vars=["fold"],
            value_vars=["test_negative", "test_positive"],
            var_name="class",
            value_name="trials",
        )
        px.bar(
            fold, x="fold", y="trials", color="class", barmode="stack",
            title="Seed 0 temporal test-fold support",
            template="plotly_white",
        ).show()
        temporal = result.oof.sort_values("trial_index")
        px.scatter(
            temporal,
            x="trial_index",
            y="mean_score",
            color=temporal.y.map({0: "miss", 1: "late hit"}),
            labels={"color": "Outcome"},
            title="OOF decision score over raw trial order",
            template="plotly_white",
        ).show()
        cells = result.cell_summary[
            result.cell_summary.selection_frequency.gt(0)
        ].sort_values(
            ["selection_frequency", "mean_abs_coefficient"], ascending=False
        ).head(60)
        px.scatter(
            cells,
            x="selection_frequency",
            y="median_coefficient",
            size="mean_abs_coefficient",
            color="median_coefficient",
            hover_data=["cell_id"],
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Cell selection frequency and standardized coefficients",
            template="plotly_white",
        ).show()


mouse_dd.observe(sync_containers, names="value")
container_dd.observe(sync_experiments, names="value")
experiment_dd.observe(update_k_options, names="value")
feature_dd.observe(render_matrix, names="value")
feature_dd.observe(render_geometry, names="value")
scale_dd.observe(render_matrix, names="value")
cell_sort_dd.observe(render_matrix, names="value")
geometry_color_dd.observe(render_geometry, names="value")
experiment_dd.observe(render_matrix, names="value")
experiment_dd.observe(render_geometry, names="value")
run_button.on_click(render_decoder)

sync_containers()
sync_experiments()
update_k_options()
render_matrix()
render_geometry()

matrix_controls = widgets.VBox([
    widgets.HBox([feature_dd, scale_dd, cell_sort_dd]),
    matrix_out,
])
geometry_controls = widgets.VBox([
    geometry_color_dd,
    geometry_out,
])
decoder_controls = widgets.VBox([
    widgets.HBox([decoder_feature_dd, k_dd, c_dd]),
    widgets.HBox([seed_dd, cv_dd, run_button]),
    decoder_out,
])
tabs = widgets.Tab(children=[matrix_controls, geometry_controls, decoder_controls])
for i, title in enumerate(["Matrix Explorer", "Trial Geometry", "Decoder Lab"]):
    tabs.set_title(i, title)

display(widgets.VBox([
    widgets.HTML("<h2>Select an experiment</h2>"),
    widgets.HBox([mouse_dd, container_dd, experiment_dd]),
    tabs,
]))
"""


behavior_notebook = notebook([
    markdown(
        """
        # Behavioral DEV Playground

        **Audience:** public teaching and research exploration.  
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

        **Audience:** public teaching and research exploration.  
        **Data:** immutable, fold-independent DEV feature cache only; no raw
        neural bundle and no CONFIRM access.

        [Release provenance](https://github.com/c-lin-chunyi/nma-project-data-analysis/releases/tag/neural-dev-features-v1-29482249873)

        Run the cells from top to bottom. This notebook is self-contained: it
        does not clone a repository. Its first code cell installs only missing
        dependencies. The complete feature cache is about 24.3 MiB compressed.
        """
    ),
    markdown("## 0. Prepare the Colab runtime"),
    code(DEPENDENCY_SETUP),
    markdown("## 1. Verified release loader\n\nEmbedded for one-file Colab use."),
    code(RELEASE_SOURCE, hidden=True),
    markdown("## 2. Typed single-experiment decoder\n\nEmbedded for one-file Colab use."),
    code(DECODER_SOURCE, hidden=True),
    markdown("## 3. Load and verify all ten feature shards"),
    code(NEURAL_LOAD),
    markdown("## 4. Cohort browser"),
    code(NEURAL_COHORT),
    markdown("## 5. Interactive feature explorer"),
    code(NEURAL_WIDGETS),
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

# NMA Project Data Analysis

Immutable DEV data releases and preregistered behavioral/neural analyses for the
Allen Visual Behavior Ophys project.

## Interactive Colab notebooks

The notebooks are standalone files: open one in Colab and run from top to
bottom. They do not clone this repository. The first code cell installs only
missing or outdated dependencies, and the data cells download checksummed,
exactly pinned GitHub Release assets.

### 01 — Behavioral DEV Playground

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/c-lin-chunyi/nma-project-data-analysis/blob/main/notebooks/01_behavioral_playground.ipynb)

Explore 50 DEV sessions and 31,997 trial labels: engagement constructs, outcome
support, guard diagnostics, session timelines, persistence, and eligibility.
The notebook downloads only the compact behavioral scan (about 1.5 MB).

### 02 — Neural Feature Explorer & Decoder Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/c-lin-chunyi/nma-project-data-analysis/blob/main/notebooks/02_neural_feature_explorer.ipynb)

Explore all 50 active DEV experiments with linked cohort, matrix, cell-effect,
representation, PCA, behavior–neural, missingness, and decoder visualizations.
It downloads the complete analysis-ready feature cache (about 24.3 MiB
compressed), not the roughly 16.8 GB raw neural package.

Both notebooks are **DEV-only educational explorers**. They never request
CONFIRM data, and exploratory decoder settings do not replace the authoritative
release analyses.

## Model workflow

The v3.3 analysis is centered on regularized logistic models: the Q1 outcome
decoder, the calibrated state-probability model, and the calibrated M0/M1
comparison for Q2. The diagram emphasizes model fitting, temporal isolation,
calibration, and evaluation rather than tree-based decision logic.

![v3.3 logistic-model workflow](docs/v33-model-workflow.svg)

The editable diagram source is
[`docs/v33-model-workflow.d2`](docs/v33-model-workflow.d2).

## v4 DEV workflow

v4 uses its own Python 3.11 environment (`requirements-v4.txt`) and does not
alter the v1–v3 analysis environments. Run the Actions in this order:

1. **build immutable neural DEV time-cache-v2** with one exact public
   `neural-dev-data-*` tag. It publishes the active-only, actual-frame cache as
   `neural-dev-time-v2-<source-run>`.
2. Run **build immutable neural DEV v4 HMM checkpoints** with the exact public
   time-cache tag. Fixed-shape K=2 primary and K=1/K=3 sensitivity fits publish
   `neural-dev-v4-hmm-<cache-source>-r2`.
3. Run **neural DEV v4 predictive-state hazard analysis** in `dev` mode with
   both exact public tags. It consumes, but never refits, the HMM checkpoints
   and dispatches one resumable shard per active target session (50 targets,
   at most ten concurrent jobs). It publishes
   `neural-dev-v4-<cache-source>-r3`. The `acceptance` mode remains an
   implementation-only dry run, performs no recovery simulation, and makes no
   analysis Release.

The r2 hazard draft is retained as an implementation/method failure and is not
overwritten. r3 separately records the immutable r2 HMM prereg hash and the r3
hazard-amendment hash. Interrupted target jobs resume only checksum- and
provenance-matched candidate groups.

A statistically nonestimable DEV result is still a valid v4 release. Pipeline
or integrity failures prevent publication. Every v4 result keeps
`numeric_sesoi=null` and `confirm_ready=false`; v4.1 is the separate freezing
step.

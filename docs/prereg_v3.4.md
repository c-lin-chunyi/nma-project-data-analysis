# Preregistration v3.4 — frozen Q1 CONFIRM analysis

**Status:** frozen before any CONFIRM behavioral or neural data access

**Scope:** the 29 mice assigned to CONFIRM by the immutable `split-lock`
Release.  CONFIRM identifiers are already public; downloading or reading their
behavior, neural measurements, or derived features constitutes CONFIRM access.

**Normative DEV base:** the public `neural-dev-v3.3-*` result freezes the Q1
estimator and the descriptive Q2 algorithm.  V3.4 changes their roles and the
opening rule: Q1 is the sole confirmatory analysis, Q2 is descriptive secondary,
and the v3.3 state anchor is retired rather than repaired or replaced.

Before CONFIRM access, the freeze Release records the exact split Release and
manifest SHA-256, v3.3 analysis Release and manifest SHA-256, this document's
SHA-256, code commit, and `requirements-pipeline.txt` SHA-256.  A mismatch stops
the workflow before data access.  The workflow also replays the v3.4 Q1 kernel
on the public DEV feature cache and requires exact session-, mouse-, and
fold-table equality with the frozen v3.3 Release before creating the access
receipt.

## 1. Population and data boundary

- The required input is exactly 29 mice, 29 ophys containers, 130 unique active
  behavior sessions, and their 130 unique active ophys experiments from
  `split-lock/confirm_mice.csv`.
- No DEV mouse, session, or experiment may appear.  Missing, duplicate, extra,
  passive, or cross-tier rows are integrity failures, not statistical outcomes.
- Only the 130 active experiments are downloaded.  Passive experiments, Q3,
  state-anchor data products, HMMs, and hazard models are outside v3.4.
- Raw Allen data are reduced to the frozen trial labels and fold-independent
  registered-window features.  All joins and file checksums must be exact.
- Feature shards use the dedicated `neural-confirm-feature-cache-v1` schema;
  a DEV-schema shard is an integrity failure and cannot be mixed into CONFIRM.

## 2. Frozen Piet-B labels and session eligibility

The behavioral construction is unchanged from v3.3.  Rolling rates use a
half-window of 25 trials and leave the current trial out.  A new lick bout begins
after a 0.7-second gap.  A trial is B-disengaged only when both:

- reward rate is below 0.5 rewards/minute (one reward per 120 seconds); and
- lick-bout rate is below 6 bouts/minute (one bout per 10 seconds).

B-engaged is the complement of that conjunction.  Ten raw trials on each side of
every B label transition are removed by the frozen guard.  A late hit is a hit
with finite response latency strictly greater than 0.30 seconds.  The primary
trial universe contains only guarded B-engaged late hits and misses.

A session is behaviorally eligible only when it contains at least 20 primary
late hits and at least 20 primary misses.  CONFIRM performs no construct,
threshold, latency, window, cell-count, or regularization sweep.

## 3. Confirmatory Q1

Q1 asks whether baseline-subtracted post-change event activity discriminates
late hits from misses among the frozen eligible trials.

- Signal/window: events mean in `[0,0.30)` seconds.
- Population: 50 Allen-QC cells selected deterministically for each of the ten
  registered cell seeds.
- Model: class-balanced L2 logistic regression, `C=1e-4`, with training-only
  standardization.
- Isolation: five contiguous outer test blocks with a ten-raw-trial purge around
  each test block.
- Session estimator: one pooled out-of-fold AUC per seed, averaged across exactly
  ten completed seeds.  A partial-seed result is nonestimable.
- Mouse estimator: eligible sessions are weighted by guarded B-engaged misses
  within mouse; mice are then equal-weighted.

Conditional within-test-fold AUC, random stratified cross-validation, and dF/F
replication are transportability diagnostics only.  They cannot replace the
pooled OOF estimator or alter the conclusion.  The DEV miss-threshold sweep is
not repeated.

## 4. Coverage, interval, and decision

All 130 active inputs must pass integrity checks.  A population conclusion also
requires at least 24 of the 29 CONFIRM mice to contain at least one estimable Q1
session.  Lower coverage produces `nonestimable_coverage`, not evidence against
the hypothesis.

The sole external SESOI is a mouse-level mean AUC of 0.55.  It is not computed
from the v3.3 state anchor or from any CONFIRM observation.  The mouse mean and a
two-sided 95% BCa interval use 2,000 mouse-level bootstrap resamples and frozen
seed 3305.

- `confirmatory_supported`: integrity and coverage pass and the interval lower
  bound is strictly greater than 0.55.
- `confirmatory_not_supported`: integrity and coverage pass but the lower bound
  is less than or equal to 0.55.
- `nonestimable_interval`: coverage passes but a finite BCa interval cannot be
  formed.
- `nonestimable_coverage`: fewer than 24 mice are estimable.
- `pipeline_failure`: required input, provenance, alignment, checksum, or
  execution integrity fails; no statistical conclusion is formed.

## 5. Descriptive secondary Q2

Q2 is executed regardless of the observed Q1 value and has no confirmatory
decision authority.  It retains the v3.3 trial universe, M0 behavioral
covariates, cross-fitted neural score, five outer folds, ten-trial purge, and ten
cell seeds.  M0 and M1 separately select `C` from
`10**{-4,-3,-2,-1,0,1,2}` using the nested four-fold one-SE rule.  Each model's
training-only inner-OOF logits fit its frozen sigmoid calibrator.

Required descriptive outputs are calibrated delta log loss, delta Brier, delta
AUC, raw uncalibrated metrics, the fixed-`C=1` comparator, calibration
intercepts/slopes, selected C values, and orthogonality diagnostics.  Sessions
are weighted by engaged misses within mouse and mice are equal-weighted.  A
descriptive two-sided 95% BCa interval uses 2,000 resamples and seed 3304.

If fewer than 24 mice are Q2-estimable, session and mouse tables remain required
but the group summary is `secondary_nonestimable_coverage`.  Q2 has no SESOI,
significance decision, multiplicity-adjusted claim, or route to changing Q1.

## 6. Frozen GitHub execution order and publication

The machine-readable execution contract is
`docs/confirm_v3.4_gh_order.json`.  The only authorized GitHub Actions order is:

1. Run **freeze CONFIRM v3.4 preregistration** once.  It reads only the already
   public `split-lock`, `neural-dev-features-v1-29482249873`, and
   `neural-dev-v3.3-29551296569` Releases and publishes
   `confirm-v3.4-freeze-<code-commit>`.
2. From that same code commit, run **neural CONFIRM v3.4 one-shot** with the
   exact tag emitted in step 1.  Its fixed job chain is `open → behavior-pull →
   labels → features → analyze → publish`.

The upstream DEV workflows are immutable prerequisites, not steps to rerun.
The freeze workflow refuses a second `confirm-v3.4-freeze-*` Release and also
refuses to run after the access receipt exists.  The order contract itself is
an asset and a hashed input of the freeze Release.

The formal workflow has a constant concurrency group, manual approval through
the `confirm-v3.4` GitHub Environment, and no model-setting inputs.  Before data
access it creates the immutable `confirm-v3.4-access` receipt binding the run ID,
commit, freeze hash, and split hash.  A new dispatch is refused after that
receipt exists.  A GitHub rerun of the same run ID may restore only checksum- and
provenance-matched completed shards; it may not change code, inputs, or methods.

The required freeze asset is `freeze-manifest.json` with schema
`confirm-v3.4-freeze-v1`.  It records `code_commit`, `prereg_sha256`,
`requirements_sha256`, `workflow_order_sha256`, `split_release`,
`split_manifest_sha256`, `v33_release`, and `v33_manifest_sha256`; the freeze
Release also contains this document, the order contract, and a `SHA256SUMS`
file.  The formal workflow refuses a missing or unequal field.
The `v34.py freeze-manifest` command builds this manifest without accessing
CONFIRM data.

The workflow performs frozen behavioral labeling, active-only neural extraction,
feature materialization, exact cache verification, Q1, and Q2 without publishing
an intermediate result.  Its final immutable Release is published whether Q1 is
supported, not supported, or statistically nonestimable.  Pipeline failures
retain diagnostic artifacts but do not publish a statistical conclusion.

Required final assets are the analysis manifest, Q1/Q2 session and mouse tables,
fold and selection diagnostics, coverage/failure table, this frozen document,
source manifests and hashes, environment/provenance records, and `SHA256SUMS`.
The final manifest uses schema `neural-confirm-v3.4`, records
`confirm_data_accessed=true`, and never derives a state SESOI or runs v4.

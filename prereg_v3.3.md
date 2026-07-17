# Preregistration v3.3 — conditional state anchor and calibrated Q2

This document is frozen before the v3.3 DEV analysis.  It changes analysis and
data access, not the DEV/CONFIRM split.  The immutable 70-experiment
`neural-dev-data-*` Release remains the source of truth.  A one-time, active-only
`neural-dev-features-v1-*` Release materializes registered-window matrices so
subsequent DEV model runs do not repeatedly stream the approximately 16 GB neural
package.  No workflow described here downloads Allen NWBs or reads CONFIRM traces,
outcomes, IDs, or derived features.

## §14 — deviations from v3.2

1. The v3.2 state anchor pooled decision scores from five separately fitted
   contiguous-fold models.  Because AUC is not decomposable across fold-specific
   score scales and engagement is temporally clustered, that pooled AUC is an
   invalid state-anchor estimator.  It remains an immutable diagnostic.
2. The authoritative v3.3 state AUC is the pair-weighted mean of within-test-fold
   AUCs under the same five contiguous outer blocks and ten-raw-trial purge.
3. The v3.2 training-prevalence state null is replaced by an exactly balanced
   fixed-0.5 null.  Absolute state-model and both null log-losses are reported.
4. Q2 replaces fixed `C=1.0` nuisance models with a frozen nested one-SE
   regularization rule and strictly training-only sigmoid calibration.  The raw
   v3.2 `C=1.0` estimates remain mandatory external comparators.
5. SESOI construction now has typed eligibility states.  An unusable or
   inconclusive anchor produces no numeric SESOI boundary.
6. Q1 K, C, signal, window, labels, folds, gap, seeds, session rule and pooled OOF
   primary estimator remain frozen.  A within-fold conditional Q1 AUC is added
   only as an aggregation audit and cannot replace the primary result.

## Immutable feature cache

- Input tags are exact `neural-dev-data-*` and matching `behavioral-v3.1-*`
  public Releases.  Their manifests and SHA-256 files are verified first.
- Expected source set is exactly 50 active plus 20 passive experiments in the ten
  frozen DEV containers.  The cache contains exactly the 50 active experiments.
- Each source container is streamed once, verified, materialized, uploaded as an
  independently checksummed cache shard, and deleted from the runner.  A draft
  cache Release may resume completed shards; a public cache Release is immutable.
- For every active experiment, the cache stores all valid Allen-QC cell IDs,
  aligned trial IDs, full behavioral labels and Q2 covariates, plus float32:
  - baseline-subtracted events mean in `[0,0.30)`;
  - unbaselined events mean in `[-1,0)`;
  - unbaselined events mean in `[0,0.30)`;
  - baseline-subtracted events mean over every stored `rel_time<0` sample;
  - baseline-subtracted dF/F mean in `[0,0.30)`.
- The cache is fold-independent and contains no fitted weights, selected C,
  imputation, scaling, calibration, OOF score, or outcome-adaptive cell subset.
- Deterministic K=50 cell subsets are regenerated from experiment ID, K and the
  ten frozen seeds.  Passive and continuous traces remain available only in the
  immutable neural data Release for later Q3 work.
- The feature-cache manifest records source tags and hashes, code commit, Python,
  runner image, environment hash, per-file and per-shard SHA-256, and explicitly
  states `allen_nwb_download=false`.

## Frozen Q1

- Target: late hit versus miss among B-engaged, guarded trials.
- Events, baseline-subtracted `[0,0.30)`, K=50, `C=1e-4`, ten deterministic cell
  seeds, five contiguous test blocks and a ten-raw-trial purge remain primary.
- Behavioral eligibility remains `n_late_hit_B>=20` and `n_miss_B>=20`.
- Primary session AUC remains one pooled OOF AUC per seed, averaged over seeds.
- The registered miss thresholds `{10,15,20,25,30}`, random-CV sensitivity,
  dF/F replication and temporal-support failures remain mandatory.
- Aggregation audit: calculate AUC within each estimable outer test fold and form
  the pair-weighted conditional AUC defined below.  It is diagnostic only; no
  Q1 conclusion or gate switches estimator after seeing it.

## Authoritative state anchor

- Target: B engaged versus disengaged using unbaselined events mean in `[-1,0)`.
- Outcome balance, transition guard, K=50, `C=1e-4`, ten cell seeds, five
  contiguous blocks and the ten-raw-trial purge remain unchanged.
- A test fold is AUC-estimable only when it contains both state classes.  A seed
  is anchor-estimable only when at least three of five test folds are estimable.
- For estimable folds `k`, the session-seed estimator is

  `AUC_cond = sum_k(n_pos_k*n_neg_k*AUC_k) / sum_k(n_pos_k*n_neg_k)`.

- Session estimates average the ten deterministic seed estimates.  Sessions are
  weighted by limiting outcome-balanced state count within mouse; mice are
  equal-weighted.  At least 8/10 DEV mice remain required.
- Mandatory fold output includes train/test state prevalence, both class counts,
  score mean/SD, within-fold AUC, comparable-pair count and estimability reason.
- The v3.2 pooled OOF AUC, an unweighted mean of estimable fold AUCs, random
  stratified CV, and a run-preserving circular-shift label null are diagnostics.
  None may replace the conditional anchor.  If conditional support is inadequate,
  the anchor is nonestimable; there is no adaptive fallback to random CV or a new
  block construction.
- The guarded conditional estimate is the only authoritative AUC anchor.
  Unguarded and other representation/window estimates remain diagnostic.

## State probability score

- State probabilities are produced with natural class prevalence; no class
  weighting is used in probability models.
- Within each outer training partition, four purged contiguous inner folds
  generate predictions unseen by their fitted state decoder.  A sigmoid
  calibrator is fit only to those inner OOF logits and labels.  The state decoder
  is refit on the full outer training partition; the frozen calibrator is then
  applied to the outer test logits.
- Report absolute calibrated state log-loss, raw uncalibrated log-loss,
  fixed-0.5 null log-loss and the v3.2 training-prevalence null log-loss.
- The authoritative state log-loss gain is
  `log_loss(y, 0.5) - log_loss(y, calibrated_probability)` on pooled outer-test
  predictions.  The fixed 0.5 null follows from exact outcome-within-state balance.

## Q2 incremental prediction

- The Q2 trial universe, M0 covariates, M1 neural score, outer folds, raw-trial
  purge, preprocessing isolation and nested neural-score cross-fitting remain as
  v3.2.  Natural outcome prevalence is preserved.
- For M0 and M1 separately, candidate nuisance-model C values are
  `10**{-4,-3,-2,-1,0,1,2}`.  Within each outer training partition, four purged
  contiguous inner folds evaluate held-out log-loss.  Choose the smallest C
  (strongest regularization) whose mean inner log-loss is within one standard
  error of that model's minimum.  No outer-test observation enters C selection.
- At the selected C, regenerate inner OOF probabilities and fit a sigmoid
  calibrator on their logits and labels.  Refit the base model on all outer
  training trials and apply that frozen calibrator to the outer test predictions.
- A missing class, incomplete inner score, non-finite calibrator or non-finite
  probability produces a typed `q2_nonestimable` reason.  Folds, C or calibration
  are never adaptively simplified.
- Primary: outer-held-out calibrated
  `delta_log_loss = log_loss(M0) - log_loss(M1)`.
- Secondary: calibrated delta Brier and delta AUC; raw uncalibrated metrics;
  v3.2 fixed-`C=1.0` outputs; M0/M1 calibration intercept and slope.
- Orthogonality diagnostics report within-outer-test correlation of neural score
  with M0 logit and M0 residual.  The Gaussian AUC-to-d quadrature calculation is
  descriptive only and cannot establish independence.
- Session estimates are weighted by engaged misses within mouse; mice are
  equal-weighted.  Uncertainty uses 2,000 mouse-level BCa bootstrap replicates.

## Typed SESOI and CONFIRM gates

- The AUC anchor status is:
  - `usable_positive` only if its mouse-level BCa 95% CI lower bound is above 0.5;
  - `invalid_direction` if its CI upper bound is below 0.5;
  - `inconclusive` if its CI includes 0.5;
  - `nonestimable` if coverage or fold support fails.
- The state-log-loss anchor uses analogous statuses around zero.
- Numeric SESOIs exist only for usable anchors:
  - `SESOI_Q1 = 0.5 + 0.2*(AUC_state_conditional - 0.5)`;
  - `SESOI_Q2 = 0.2*state_calibrated_logloss_gain`.
- Otherwise the manifest stores `null` for the corresponding SESOI and a typed
  reason.  No below-chance boundary participates in arithmetic comparisons.
- Coverage still requires exact feature-cache checksums and at least 8/10 DEV
  mice for Q1, anchor and Q2.
- With 29 frozen CONFIRM mice, projected precision gates remain:
  - `t(.975,28)*SD_Q1_DEV/sqrt(29) < SESOI_Q1-0.5`;
  - `t(.975,28)*SD_delta_logloss_DEV/sqrt(29) < SESOI_Q2`.
- Both anchors must be usable and both precision inequalities must pass for
  `confirm_ready=true`.  Observed DEV Q1 and Q2 means are not gate conditions.
- Failure stops.  K, C grid, folds, estimator, calibration or anchor cannot be
  changed without a new preregistration version.  This repository still contains
  no CONFIRM workflow.

## Required outputs and single-run discipline

- Cache manifest, source hashes, exact trial/cell flow and validation report.
- Q1 primary, selection sweep, conditional audit and fold diagnostics.
- State conditional anchor, all null/log-loss components, calibration and pooled
  v3.2 comparator.
- Q2 calibrated/raw session and mouse tables, selected C per outer fold,
  calibration diagnostics and orthogonality diagnostics.
- Typed SESOI states, projected half-widths and `confirm_ready` reasons.
- The v3.3 primary specification is executed once on DEV.  Diagnostics do not
  create alternate registered conclusions.  All results cite the immutable
  feature-cache tag and manifest SHA-256.

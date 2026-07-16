# Preregistration v3.2 — DEV K=50 anchor and Q2 correction

This document is frozen before the v3.2 DEV re-analysis. It changes analysis only:
the previously extracted 50 active and 20 passive DEV experiment bundles are reused
without another Allen NWB download. CONFIRM traces and outcomes remain unopened.
The immutable bundles are published once as a `neural-dev-data-*` Release; every
analysis Release records that tag and its data-manifest SHA-256 rather than
duplicating the large neural package.

## §14 — Deviations from v3.1

1. K=50 remains the only authoritative neural population size because varying cell
   yield is a session-level source of heterogeneity. The v3.1 all-cell and learning-
   curve results remain immutable external comparators and are not recomputed.
2. `C50=1e-4`, selected by the frozen v3.1 one-SE rule, is reused without a new grid
   search or outcome-dependent tuning.
3. The state anchor uses the unbaselined events mean in `[-1,0)` rather than the
   baseline-subtracted post-change response. This targets tonic engagement state.
4. Q2 M0/M1 are fit without class weighting and evaluated under the observed late-
   hit prevalence. Absolute metrics and calibration diagnostics are mandatory.
5. The Q2 SESOI is expressed in its primary unit: 20% of the held-out state
   log-loss gain over a training-fold intercept-only model.
6. Opening CONFIRM requires both Q1 and Q2 projected precision gates to pass.

## Frozen estimands

- Q1: late hit vs miss from baseline-subtracted events averaged over `[0,0.30)`,
  among B-engaged, guarded trials. K=50, `C=1e-4`, five contiguous blocks, a
  ten-raw-trial gap, ten deterministic cell seeds, and pooled OOF AUC are fixed.
- State anchor: B engaged vs disengaged from unbaselined events averaged over
  `[-1,0)`. Labels are balanced within late-hit/miss outcome. The guarded estimate
  is the only authoritative anchor; unguarded and the other baseline/time-window
  combinations are diagnostics only.
- Q2: held-out `delta_log_loss = log_loss(M0) - log_loss(M1)` under the observed
  outcome prevalence. `M1=M0+neural_score`; all preprocessing and neural scores are
  strictly nested cross-fitted. Sessions are weighted by engaged misses within
  mouse and mice are equal-weighted.

## SESOI and gates

- `SESOI_Q1 = 0.5 + 0.2 * (AUC_state_guarded - 0.5)`.
- `SESOI_Q2 = 0.2 * state_logloss_gain_guarded`.
- A non-finite or non-positive anchor margin fails its corresponding gate.
- Coverage requires complete 70/70 Appendix-A bundles and at least 8/10 DEV mice
  for Q1, the authoritative state anchor, and Q2.
- Q1 precision requires
  `t(.975,28) * SD_Q1_DEV / sqrt(29) < 0.2*(AUC_state_guarded-0.5)`.
- Q2 precision requires
  `t(.975,28) * SD_delta_logloss_DEV / sqrt(29) < SESOI_Q2`.
- `confirm_ready=true` only when coverage and both precision gates pass. Neither
  observed DEV Q1 nor observed DEV Q2 mean is a gate condition.

## Mandatory diagnostics

- Report guarded baseline/unbaseline by pre/post state-anchor estimates and the
  unguarded authoritative representation without generating another SESOI.
- Verify that the stored baseline-subtracted pre-change mean is numerically zero
  and its fixed-axis decision value is constant (AUC defined as 0.5).
- Report M0/M1 absolute log-loss, AUC and Brier score; neural-only metrics; Q1 AUC
  on the exact Q2 trial universe; prevalence; and descriptive calibration
  intercepts/slopes.
- Preserve the registered miss-threshold sweep. No other K or C sweep is run.

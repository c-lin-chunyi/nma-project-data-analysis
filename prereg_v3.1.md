# Preregistration v3.1 — Does residual lapse carry cortical structure?

Status: DEV amendment; CONFIRM remains closed until both machine-readable gates pass.

Amendment date: 2026-07-16 (Asia/Taipei).

## Provenance

- Allen manifest: `visual-behavior-ophys_project_manifest_v1.1.0.json`
- AllenSDK: `2.16.2`
- Python: `3.11`
- Extraction and analysis commits: the immutable `behavioral-manifest.json` and
  `analysis-manifest.json` in the corresponding Releases are authoritative.
- Runner image, resolved environment and their SHA-256 values are recorded by each
  workflow run and Release. No container is used.
- Split checksum: inherited unchanged from the one-time split Release.

This document amends v3 only where listed below. All other v3 definitions remain in
force. The frozen v3 Q1 comparator is always emitted alongside v3.1.

## §14 — Registered deviations

1. The executed split uses the metadata-only stratified Hamilton rule rather than
   the earlier unstratified first-quartile hash rule. It is not redrawn.
2. Primary engagement is Piet B: `NOT(low bout rate AND low reward rate)`. The v3
   reward-rate-only A result is the frozen comparator.
3. Primary population size is K=50. The frozen comparator uses all Allen-QC cells.
4. Sessions require at least 20 B-guarded late hits and 20 B-guarded misses. The
   miss threshold is swept over 10/15/20/25/30; no contamination session cutoff is
   added.
5. Novelty is exploratory because only 7/50 DEV sessions are novel. The frozen
   secondary model containing novelty is still reported.
6. The SESOI state anchor balances engagement within late-hit/miss outcome. Only the
   guarded v3.1 anchor is authoritative; unguarded is diagnostic.
7. Logistic C is chosen separately for each K on DEV by the mouse-level one-SE rule.
   CONFIRM uses the frozen K=50 value and never tunes C.
8. §11's square-root-neuron extrapolation is withdrawn. The DEV learning curve and
   projected CONFIRM CI half-width form the second opening gate.

## Primary estimands

- Q1: held-out decodability of **late hit vs miss** from the baseline-subtracted
  0–0.30 s events response, among B-engaged change trials retained by the ±10-trial
  transition guard.
- Q2: held-out incremental log loss of `M1=M0+neural_score` over M0, where M0 contains
  transition, frozen trial history, session position and pre-change pupil/running.
- Error unit: mouse. Sessions are weighted by engaged misses; mice are equal-weighted.

## Frozen pipeline

- Window saved: −1.25 to +1.5 s; events primary; dF/F retained.
- Early hits (`response_latency <= 0.30`) are removed. The contrast is never named
  hit vs miss.
- Five contiguous temporal blocks; ten-raw-trial training gap; pooled OOF AUC; ten
  seeds; class-balanced L2 logistic.
- K grid: 10/25/50/100/158/all. Each K receives one global DEV-frozen C selected
  from `10**[-4..4]` by the one-SE rule.
- Missing temporal class support is an estimability result, not a behavioral
  exclusion. Random stratified CV is reported beside blocked CV.

## §11 gates

Gate 1 requires complete 50-active + 20-passive Appendix-A extraction, an estimable
guarded state anchor, and at least 8/10 DEV mice estimable for both anchor and Q1.

Gate 2 uses the K=50 between-mouse Q1 SD:

`half_width = t(.975, 28) * SD_DEV / sqrt(29)`

and requires

`half_width < 0.2 * (AUC_state_guarded - 0.5)`.

The observed Q1 mean is not part of the gate. Failure stops the project; K is not
changed without a new preregistration.

## §12 single anchor

`AUC_state_guarded` is measured once on DEV with the v3.1 K=50/C50 pipeline and
outcome-balanced B engagement labels. The identical unguarded analysis is reported
only as a guard diagnostic.

- `SESOI_Q1 = 0.5 + 0.2 * (AUC_state_guarded - 0.5)`
- `SESOI_Q2 = 0.2 * (AUC_state_guarded - 0.5)`

No frozen-v3 or alternative anchor produces a second registered bound.

## Appendix A extraction contract

The exact DEV set is 50 active experiments plus both OPHYS_2/OPHYS_5 passive
experiments in each of the ten DEV containers: 70 total. Bundles retain events,
dF/F, valid cell IDs, complete task/stimulus tables, absolute licks/rewards, pupil,
running and provenance. Active sessions include trial-locked tensors. Passive
sessions retain continuous traces plus the compact stimulus table; deferred Q3
alignment does not require another NWB download.

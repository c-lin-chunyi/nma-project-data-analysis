# Preregistration v4 (DRAFT) — predictive behavioral state and causal neural history

**Status:** DRAFT, decision-complete DEV methods amendment

**Scope:** the immutable DEV split only; CONFIRM remains closed

**Freeze rule:** this file is not frozen until the acceptance conditions in
§13 pass and a later explicit change marks it frozen

This document supersedes v3.x only for the future v4 DEV analysis. It does not
alter, rerun, or reinterpret any v1–v3 result. The v3.3 fixed-window result is an
immutable external comparator and cannot select a v4 model, parameter, or
conclusion. V4 replaces the behavioral label “engagement” with a behavior-only
predictive latent state, models lick timing directly, and asks whether causal
neural history improves out-of-sample lick-hazard prediction beyond behavior.

V4 is intentionally not a CONFIRM-opening preregistration. It will produce no
numeric SESOI and keeps `confirm_ready` false. After exactly one registered v4
DEV analysis, a separate v4.1 amendment may freeze a numeric SESOI and an
opening gate without using the sign of the observed v4 effect as an eligibility
criterion.

## §1 — registered questions, estimands, and nonclaims

The primary question is whether causal event-history features improve
prequential prediction of the first protocol-window lick on valid go trials,
conditional on protocol variables, behavioral history, pupil, running, and the
one-step predictive latent-state distribution.

For each estimable session, the primary estimand is

\[
\Delta LL_s =
  LL_{\mathrm{trial},s}(M1)-LL_{\mathrm{trial},s}(M0),
\]

where each term is the sum of held-out discrete-time survival log likelihood
over evaluated trials divided by the number of those trials. Positive values
favor causal neural history.

The registered secondary model `M2` asks whether the neural contribution varies
with predictive latent state. The mandatory dF/F analysis replicates `M1`; it
does not replace the event-based primary. V4 makes no claim that:

- a latent state is “engaged,” inattentive, aroused, or otherwise
  psychologically identified;
- a state inferred after observing trial-\(t\) behavior is available for
  prediction of that behavior;
- timestamp-causal use of released deconvolved events means their computation
  was itself real-time or free of future information;
- a positive predictive increment identifies a causal effect of neural activity
  on licking.

The phrase **predictive latent behavioral state** refers only to the registered
behavior-only GLM-HMM and its one-step predictive distribution.

## §2 — data boundary, compatibility, and cache contract

- The source population is exactly the existing 50 active experiments from the
  ten mice in the immutable `neural-dev-data-*` release. All 50 active
  experiments are required at input validation. The 20 passive experiments do
  not enter the v4 estimand, fitting, tuning, or coverage count.
- Matching HDF5/Parquet release manifests and SHA-256 files are verified before
  any analysis. V4 does not download or read Allen NWB files and does not require
  AllenSDK.
- A future `time-resolved-cache-v2` may be constructed once from the existing
  HDF5/Parquet release. It is a lossless, fold-independent materialization of
  trial rows, actual ophys timestamps, behavior timestamps, cell identifiers,
  events, dF/F, pupil, and running required below. It contains no fitted
  parameters, imputation values, selected hyperparameters, state posteriors, or
  outcome-adaptive cell selection.
- The cache manifest records source tags and SHA-256 hashes, code commit, Python
  version, `requirements-v4.txt` SHA-256, installed-environment hash, JAX device
  and 64-bit status, per-file hashes, trial counts, cell counts, and
  `allen_nwb_download=false`.
- Trial identifiers, session identifiers, timestamps, and neural rows must join
  one-to-one. Duplicate, reordered, unmatched, nonmonotonic, or checksum-invalid
  records stop the run with a typed integrity failure; they are never silently
  dropped.
- V3.3 outputs and its fixed-window feature cache are read only as external
  comparators after all v4 registered outputs exist. They do not enter any v4
  model-selection calculation.

All training, tuning, standardization, imputation, state filtering, and history
construction below reset at session boundaries. No information is carried
between sessions except fitted parameters learned from explicitly permitted
training sessions.

## §3 — behavior-only multinomial GLM-HMM

### 3.1 Emission and covariates

Each scheduled change or sham/catch trial has one of three observed emissions:

1. `abort`: any lick in the half-open interval from the recorded trial start
   through, but not including, the scheduled change/sham timestamp,
   \([\text{start_time},\text{change_time})\);
2. `task_response`: if not aborted, the first lick lies in the official
   post-stimulus response window \([0.15,0.75]\) seconds;
3. `withhold`: if not aborted and no lick occurs in that window.

Auto-rewarded trials and trials lacking the fields needed to assign exactly one
of these categories have a missing emission. They remain in temporal order and
advance the latent transition process but contribute no emission likelihood.
The v4 wrapper supplies an explicit observation mask: a missing row receives
zero emission log likelihood while both adjacent transition likelihoods and
transition sufficient statistics are retained. The package's end-padding
sentinel is not used for internal missing trials. An invented outcome category
or deletion and concatenation across the gap is prohibited.

The emission GLM has exactly these inputs:

- intercept;
- scheduled change versus catch/sham;
- Allen `is_image_novel` for the scheduled stimulus, with an explicit missing
  level and no inference from OPHYS session names;
- previous observed emission, one-hot encoded with a separate
  `no_previous_observed_emission` level;
- previous reward indicator, with a separate missing indicator;
- session position, defined as zero-based raw-trial index divided by
  `max(n_raw_trials-1, 1)`.

Previous emission means the most recent earlier nonmissing emission within the
same session. Previous reward refers to the immediately preceding raw trial;
session starts receive the documented missing value. No neural, lick-latency,
pupil, running, current outcome, or future variable is an HMM input. Continuous
covariates are centered and scaled using only the relevant training sessions; a
training zero-variance variable is set to zero in both training and test data.
The transition matrix is stationary and receives no exogenous input.

### 3.2 Target-session isolation and state-number selection

For target session \(s\), HMM parameters are learned only from the other active
sessions of the same mouse. The target session contributes neither emissions nor
covariate standardization to its HMM fit. Candidate state counts are
\(K\in\{1,2,3,4\}\).

Within those training sessions, nested leave-one-session-out evaluation is used:
each training session is held out in turn, candidate parameters are fit to all
remaining training sessions, and held-out marginal log likelihood is divided by
the held-out raw-trial count, including rows with missing emissions. The score
for a candidate is the unweighted mean of these session scores; its standard
error is the sample standard deviation across held-out sessions divided by their
square root.
Let \(K^\*\) maximize the mean. The eligible set contains candidates whose mean
is at least `mean(K*) - SE(K*)`; the selected value is its smallest \(K\).
Thus \(K=1\) is an allowed substantive null result.

At least two estimable inner held-out sessions are required. A missing candidate
score, nonfinite likelihood, or inability to fit any inner fold makes the target
session HMM nonestimable; no inner fold, candidate \(K\), or state is removed.
After selecting \(K\), parameters are refit on every permitted same-mouse
training session.

### 3.3 Optimization and predictive posterior

The implementation imports `SoftmaxGLMHMM` from `glmhmmt` at the exact Git
commit pinned in `requirements-v4.txt`, with `num_classes=3`,
`transition_input_dim=0`, `baseline_class_idx=0` (`abort`), initial and
transition Dirichlet concentrations `1.1`, transition stickiness `0`,
initial-weight scale `1.0`, emission M-step `optax.adam(1e-2)`, 100 gradient
steps per EM M-step, and the commit's fixed emission L2 value `1e-4`. No
emission feature is frozen.

Each candidate and final refit uses the 20 integer JAX seeds `4100, ..., 4119`,
at most 500 EM iterations, and convergence tolerance `1e-4`, defined as absolute
change in unpenalized data marginal log likelihood below tolerance for two
successive EM iterations. JAX 64-bit mode must be enabled before model
construction. Among converged finite fits, the fit with the highest unpenalized
training data marginal likelihood is retained. No converged fit produces typed
reason `hmm_no_converged_initialization`.

State labels are ordered deterministically after fitting by lexicographic order
of the three emission probabilities at the all-reference covariate vector
(continuous inputs zero; categorical inputs at their documented reference
levels). This ordering affects columns and reporting only, never fit quality or
state interpretation.

For trial \(t\) in a target session, the sole registered state feature is

\[
q_t = p(z_t\mid y_{1:t-1}^{\mathrm{obs}},x_{1:t}),
\]

computed with `predict_state_probs` from the fitted initial distribution at each
session start. Filtering consumes an emission only after producing that trial's
prediction; missing emissions perform the transition update without an emission
update. Current and future emissions are prohibited. Smoothed
\(p(z_t\mid y_{1:T},x_{1:T})\) values are prohibited.

Hazard models use the first \(K-1\) components of the complete \(q_t\); the last
component is the reference. No state is selected as task responsive, no
probability is thresholded, and no state is named “engaged.” For \(K=1\), the
state feature has zero columns and the registered `M2` interaction is
`not_applicable_k1`.

## §4 — primary lick-hazard universe and time grid

The primary trial universe consists of trials that:

- contain an actually occurring scheduled image change (go trials);
- are not aborted and not auto-rewarded;
- have valid, monotonically increasing ophys timestamps and the source columns
  needed to apply the registered neural and behavioral extraction. Missing
  pupil/running values themselves are retained under §6.

A trial with any lick in \([0,0.15)\) seconds after the actual change is a
**pre-window competing event**. It is removed from the primary risk set and
reported by session and mouse; it is not recoded as censoring or as an event at
0.15 seconds.

All remaining trials enter the risk set at 0.15 seconds. The first lick in
\([0.15,0.75]\) is the event. A trial without such a lick is administratively
censored at 0.75 seconds. These boundaries are anchored to the Allen task's
[official response window](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/visual_behavior_neuropixels_analyzing_behavior_only_data.html);
the v3.x 0.30-second neural window is not a v4 behavioral cutoff.

For each trial, bin boundaries are 0.15 seconds, every actual ophys frame
timestamp strictly inside \((0.15,0.75)\), and 0.75 seconds. Bins are
left-closed/right-open, except that the final bin includes 0.75. An event belongs
to the unique bin containing its first-lick timestamp. Rows after an event do
not exist. Let \(\Delta_j\) be bin width in seconds; `log(Delta_j)` is an offset
so unequal frame intervals are respected. Duplicate, nonincreasing, or
nonfinite boundaries are an integrity failure, not repaired.

For bin \(j\), the complementary-log-log model is

\[
\log[-\log(1-h_{tj})] =
  \log(\Delta_j)+\alpha_j+w^\top X_{tj}.
\]

`alpha_j` is an unpenalized ordinal-bin baseline hazard shared across trials.
The trial log likelihood sums survival contributions through the event/censor
bin. Per-trial held-out log likelihood is the sum across trials divided by the
number of evaluated trials, not by the number of hazard rows.

## §5 — registered hazard models

### 5.1 Behavioral and protocol model M0

`M0` includes:

- the frame-specific baseline hazard and complete \(K-1\) predictive-state
  vector;
- flashes before the actual change;
- time since the preceding change, lick, and reward;
- session position as defined in §3.1;
- image-transition identity;
- preceding-omission indicator;
- previous outcome;
- pre-change pupil diameter and running speed.

Times since prior events are evaluated at change time, use only earlier
timestamps in the same session, are transformed as `log1p(seconds)`, and have a
separate `no_prior_event` indicator. Pre-change pupil and running are the median
of finite samples in \([-1.25,0)\) relative to actual change. Previous outcome
is the preceding raw trial's registered three-level emission plus explicit
missing/session-start level. Image transition is a training-defined categorical
variable; unseen test levels map to a fixed `unseen` level created in training.

### 5.2 Causal neural-history model M1

`M1` adds event-history features to every `M0` term. For each experiment and
each of the ten existing deterministic cell-subset seeds `0, ..., 9`, exactly
50 Allen-QC-valid cells are selected from the canonical stored HDF5 cell order,
preserving the existing v3.x subset rule. The PRNG seed is the unsigned
big-endian integer represented by the first eight bytes of
`SHA256(f"{experiment_id}:50:{cell_seed}")`; NumPy `default_rng` samples 50
indices without replacement and the selected indices are sorted. A session with
fewer than 50 valid cells is typed `neural_fewer_than_50_cells`; it is not
padded and no smaller \(K_{\rm cell}\) is tried.

For each selected cell:

- the cell's pre-change mean over \([-1.25,0)\) is subtracted from its event
  trace for that trial;
- at hazard-bin left edge \(u\), only source frames with actual timestamp
  \(v<u\) are eligible;
- causal lags \(u-v\) outside \([0,0.75]\) contribute zero;
- eligible values are convolved with \(B\) equally spaced raised-cosine basis
  functions on \([0,0.75]\), using an unweighted sum over source frames. For
  \(B>1\), centers are \(c_b=0.75b/(B-1)\), width
  \(d=0.75/(B-1)\), and
  \(\phi_b(l)=[1+\cos(\pi(l-c_b)/d)]/2\) when
  \(|l-c_b|\le d\), zero otherwise. Each basis is normalized to unit discrete
  \(L_2\) norm on the training frame grid. For \(B=1\), the registered basis is
  constant on the support. Candidate values are \(B\in\{1,2,3,4\}\).

The resulting 50-by-\(B\) features remain cell-specific; they are not averaged
across cells. Frames at or after a bin's left edge, at or after the first lick,
or in any future trial/block never contribute. Empty causal history yields
zeros after baseline subtraction rather than a missing value.

### 5.3 State-dependent model M2 and dF/F replication

`M2` adds every product between the \(K-1\) predictive-state components and the
50-by-\(B\) neural-basis features to `M1`. It is a preregistered secondary model
evaluated with the same likelihood and aggregation rules. When \(K=1\), `M2` is
not applicable, not equal to `M1`, and does not affect v4 estimability.

Events are primary. The mandatory dF/F replication replaces only the selected
cells' event traces with their dF/F traces and repeats baseline subtraction,
causal feature construction, `M0`/`M1` prediction, and reporting. For a target
session it reuses the event-primary selected \(B\) and penalties; it performs no
new hyperparameter search and cannot replace the event result.

## §6 — preprocessing, missingness, and penalties

All continuous non-neural covariates are standardized with mean and sample
standard deviation computed on the current training prefix only. Neural-history
columns are likewise standardized within the current training prefix after
construction. A zero-variance training column is set to zero in training and
test data and remains in the registered design matrix.

For every continuous covariate, finite training values are median-imputed and a
missingness indicator is included. Test values use that frozen median. If a
training column is entirely missing, its imputed value is zero and its
missingness indicator is one; this is reported but does not trigger a
pupil/running coverage cutoff. Categorical missing and unseen levels follow the
explicit levels in §5. No test-prefix statistic is used for preprocessing.

Ridge penalties are

\[
\lambda\in\{10^{-4},10^{-3},10^{-2},10^{-1},10^0,10^1,10^2\}.
\]

The offset and ordinal-bin baseline coefficients are unpenalized. All other
coefficients, including missingness indicators and interactions, receive the
same model-specific ridge penalty. There is no lasso, elastic net, feature
screening, post-selection refit, or optimizer fallback.

Every hazard fit maximizes the registered penalized complementary-log-log
likelihood in float64 with JAX analytic gradients and SciPy
`optimize.minimize(method="L-BFGS-B")`, zero initialization, `maxiter=2000`,
`ftol=1e-9`, and `gtol=1e-7`. Success requires the optimizer success flag plus
finite coefficients, objective, gradient, and test probabilities. A failure is
typed `hazard_nonconvergence`; neither initialization nor optimizer is changed.

## §7 — prequential blocks, tuning, and information boundary

### 7.1 Outer evaluation within each target session

Before applying any eligibility criterion, every session's chronologically
ordered raw trials are split with `numpy.array_split` into five contiguous
blocks whose sizes differ by at most one. Block 1 is warm-up only. For
\(k=2,3,4,5\), models are fit on raw blocks \(1,\ldots,k-1\) and predict block
\(k\). Only eligible hazard trials in test blocks 2–5 contribute to evaluation.

Every training index must be strictly less than every test-block index. State
filtering and behavioral/neural history for a test block may use earlier raw
trials in the same session, but no later trial or block. The fitted HMM
parameters remain target-session-external as specified in §3; only its forward
filter is run through the target session.

Each training prefix must contain at least one primary-window hazard event. A
zero-event prefix, nonconvergent hazard fit, nonfinite/incomplete test
probability, or absent eligible trial in any test block makes the whole
session-seed-model typed nonestimable. All four test blocks are mandatory.
There is no switch to blocked, random, leave-one-trial-out, or fewer-fold
cross-validation.

### 7.2 Target-session-external hyperparameter selection

For target session \(s\), hyperparameters are selected only from the same
mouse's other sessions. Each such tuning session is scored by its own complete
five-block prequential procedure. Candidate score is the unweighted mean of
session-level held-out per-trial log likelihood; its standard error is the
sample standard deviation across tuning sessions divided by their square root.
At least two fully estimable tuning sessions are required.

The predictive state covariate for a tuning session \(u\) is generated by a
nested HMM at the target-selected \(K\): that HMM is fit only to same-mouse
sessions excluding both target \(s\) and tuning session \(u\), with training-only
standardization. Its one-step forward posterior is then evaluated on \(u\).
Thus tuning-session emissions never estimate the HMM parameters used for that
session, and a target-session fit or smoothed posterior is never substituted.

For each model separately, let the best candidate maximize mean likelihood.
Candidates within one standard error of that best mean are eligible. `M0`
selects the largest eligible \(\lambda\). `M1` and `M2` first select the smallest
eligible \(B\), then the largest eligible \(\lambda\) at that \(B\). The selected
values are frozen for all four target-session training prefixes and ten cell
seeds. In neural-model tuning, each tuning-session score is first computed
within seed and then averaged equally across all ten seeds; all ten must be
estimable. Any missing candidate score makes selection nonestimable; candidate
models or tuning sessions are not discarded after inspection.

## §8 — session, mouse, and group estimators

For each cell seed, held-out log likelihood is summed across all eligible trials
in test blocks 2–5 and divided by their total count. `Delta_LL` is formed from
paired `M1` and `M0` log likelihoods on the identical trials, then averaged
equally across the ten seeds to obtain one session estimate. The session output
also reports each block and seed separately.

Within mouse \(m\), session estimates are weighted by the number of paired,
eligible, evaluated trials contributing to that session:

\[
\Delta LL_m =
\frac{\sum_s n_s\Delta LL_s}{\sum_s n_s}.
\]

Mice are equal-weighted. If \(n\) mice are estimable, the group estimate is the
arithmetic mean of their mouse estimates, with primary 95% interval

\[
\bar{\Delta LL}\ \pm\
t_{0.975,n-1}\,s_{\mathrm{mouse}}/\sqrt n.
\]

Population interpretation requires at least 8 of the ten immutable DEV mice. If
fewer than eight are estimable, v4 is globally `nonestimable_mouse_coverage`;
session and mouse outputs and the missingness mechanism are still reported, but
no group direction is interpreted.

Diagnostics are a 10,000-replicate mouse-level BCa 95% interval using seed
`4201`, and all leave-one-mouse-out group estimates. Bootstrap replicates sample
only the estimable mouse estimates with replacement and never resample sessions
or trials. These diagnostics cannot replace the t interval.

`M2-M1`, event/dF/F comparisons, state-number summaries, component coefficients,
hazard curves, and v3.3 comparisons are secondary or descriptive. Multiplicity
is not used to select a primary conclusion.

## §9 — typed nonestimability and no-fallback rule

At minimum the implementation distinguishes:

- `source_integrity_failure`;
- `trial_alignment_failure`;
- `invalid_timestamp_grid`;
- `hmm_insufficient_training_sessions`;
- `hmm_inner_candidate_failure`;
- `hmm_no_converged_initialization`;
- `hmm_nonfinite_predictive_posterior`;
- `neural_fewer_than_50_cells`;
- `hazard_no_training_event`;
- `hazard_tuning_insufficient_sessions`;
- `hazard_candidate_failure`;
- `hazard_nonconvergence`;
- `hazard_incomplete_prediction`;
- `hazard_empty_test_block`;
- `nonestimable_mouse_coverage`;
- `not_applicable_k1` for `M2` only.

The manifest records the first failure and all independently detectable
co-failures with session, block, seed, model, and hyperparameter identifiers.
No failure permits changing the universe, cutoff, \(K\), cell count, block
count, seed count, basis grid, penalty grid, imputation, optimizer, or
aggregation. Diagnostic analyses are never adaptive fallbacks.

## §10 — mandatory sensitivity and diagnostic outputs

The following are run and labeled nonprimary:

- the complete held-out \(K=1,2,3,4\) GLM-HMM likelihood table and selection
  stability across target sessions;
- results for every registered \(B,\lambda\) pair on permitted tuning sessions;
- all four outer test blocks and all ten cell seeds before aggregation;
- the pre-window competing-event fraction and administrative-censoring fraction
  by session and mouse;
- `M2-M1` where \(K>1\);
- the dF/F replication using event-selected hyperparameters;
- the 10,000-replicate BCa interval and leave-one-mouse-out estimates;
- estimates stratified descriptively by novelty and image transition;
- comparison with the immutable v3.3 fixed-window result after registered v4
  outputs are finalized.

Sensitivity results cannot alter the v4 primary universe, estimate, uncertainty,
or v4.1 eligibility.

## §11 — parameter-rationale register

| Choice | Role and source | Why adjacent alternatives are not primary | Mandatory stress test | Failure consequence |
|---|---|---|---|---|
| Response window `[0.15,0.75] s` | Allen protocol-defined response window | Avoids inventing a premotor cutoff such as `0.30 s`; neither boundary is claimed to perfectly separate preparation | Report `[0,0.15)` competing events and censoring fractions | Boundary/alignment failure is nonestimable |
| `K=1..4` | Allows a no-state model and a small interpretable state family | `K=1` must remain possible; larger searches are weakly identified with few same-mouse sessions | Full candidate likelihood and selection table | Any candidate failure invalidates selection |
| 20 HMM starts | Controls local optima at fixed computational cost | Fewer starts raise local-optimum risk; more starts add compute after a fixed reproducibility check | Report converged count and likelihood range | Zero convergence is nonestimable |
| 50 cells | Preserves the immutable v3.x neural sampling scale and equalizes feature dimension | Smaller/larger dimensions would change capacity and comparability | Ten seed-specific results and valid-cell counts | Fewer than 50 cells is nonestimable |
| 5 contiguous blocks | Gives four temporal predictions while preserving a warm-up prefix | Fewer blocks reduce temporal checks; more blocks increase zero-event prefixes | Blockwise likelihood/event counts | Any required block failure makes session nonestimable |
| 10 cell seeds | Carries forward deterministic cell-sampling uncertainty | One subset is unstable; more seeds change preregistered compute and weighting | All seed estimates and dispersion | Missing seed makes paired session result nonestimable |
| `B=1..4` | Represents increasingly flexible causal histories over the protocol window | A fixed basis count would hide smoothness choice; larger bases inflate 50-cell dimension | Full target-external tuning surface | Any candidate failure invalidates selection |
| Ridge `1e-4..1e2` | Fixed seven-decade shrinkage range | Narrower grids may force a boundary; adaptive extension would inspect outcomes | Selected value and boundary-hit flag | No adaptive extension; selection failure is nonestimable |
| At least 8/10 mice | Retains the established DEV population-coverage floor | Lower coverage is too sensitive to missing mice; 10/10 would turn any technical loss into a silent target change | Missingness table and leave-one-mouse-out | Whole v4 has no population interpretation |

The `0.30 s` fixed window, `B-engaged` label, ten-trial transition guard, 20/20
outcome eligibility, and 3/5-fold rule from earlier preregistrations have no role
in the v4 primary estimand. They may appear only when accurately labeling the
immutable v3.3 comparator.

## §12 — required release contents and single-run discipline

The future v4 DEV release contains:

- source, cache, feature, and environment manifests with SHA-256 hashes;
- complete trial-flow tables from raw trial through competing event,
  eligibility, event, censoring, block, and paired-model evaluation;
- HMM candidate/fold/start diagnostics, selected \(K\), fitted initial
  distribution, and predictive—not smoothed—posterior checks;
- hazard candidate surfaces, selected \(B,\lambda\), convergence, coefficient,
  calibration, event, censoring, block, seed, session, mouse, and group tables;
- primary event `M1-M0`, secondary `M2-M1`, mandatory dF/F replication, t
  interval, BCa interval, and leave-one-mouse-out results;
- all typed failures, mouse coverage, and a machine-readable v4.1 eligibility
  record.

The registered v4 primary specification is executed exactly once on immutable
DEV data after acceptance. Re-execution for a software defect must preserve the
failed release, document the defect before viewing corrected results, increment
the release identifier, and make no method change. A scientific method change
requires a new preregistration version.

## §13 — implementation acceptance, DRAFT freeze, and v4.1

Before the first v4 DEV analysis, future implementation must pass:

1. a clean Python 3.11 install from `requirements-v4.txt`, import
   `SoftmaxGLMHMM`, call `predict_state_probs`, and confirm JAX 64-bit
   likelihood arrays;
2. a future-behavior invariance test: changing emissions after trial \(t\)
   leaves all predictive posteriors through \(t\) unchanged;
3. a future-neural invariance test: changing neural frames at or after a bin
   left edge leaves that bin's causal features unchanged;
4. proof that internal missing emissions preserve transition propagation and
   that session padding remains distinct from internal missingness;
5. proof that frames at or after the first lick never enter risk covariates;
6. session-boundary reset, `[0,0.15)` competing-event, exact event-bin, and
   0.75-second administrative-censor tests;
7. expanding-window assertions that every training index and fitted
   preprocessing observation precedes the test block;
8. source/cache/release checksum, one-to-one trial alignment, monotonic
   timestamp, and feature-shape validation.

Parameter-recovery, latent-state-recovery, and synthetic \(K\)-selection
simulations are not acceptance gates and do not determine preregistration
freeze, DEV execution, v4.1 eligibility, or interpretation. They are not used
to claim that a fitted state has a particular psychological meaning. The
registered 20-start, 500-iteration configuration is evaluated on the immutable
DEV analysis itself under the typed failure and coverage rules above.

The two legacy requirements files must retain their pre-v4 byte contents.
Acceptance records their SHA-256 values before and after v4 setup. The v4
environment uses Python 3.11, the isolated fully pinned `requirements-v4.txt`,
JAX 64-bit mode, and no AllenSDK; existing workflows continue in their original
environments.

This document remains **DRAFT** until dependency installation and all
implementation-invariance, alignment, and causal-history acceptance checks
pass. Only a separate, explicit preregistration change may then mark v4 frozen.

After the one registered v4 DEV run, v4.1 eligibility requires only:

- validated source/cache integrity;
- at least 8/10 estimable mice;
- completion of every registered diagnostic.

Eligibility does not depend on the sign or magnitude of observed
\(\Delta LL\). V4 itself emits `numeric_sesoi=null`,
`confirm_ready=false`, and reason `v4_dev_methods_revision_confirm_closed`.

# Preregistration v4 r3 (DRAFT) — hazard-layer amendment

**Status:** DRAFT, decision-complete DEV methods amendment

**Scope:** immutable DEV only; CONFIRM remains closed

**Normative base:** [`prereg_v4_r2.md`](prereg_v4_r2.md), the byte-identical
snapshot from commit `7741509`, SHA-256
`015e0feec8ec9330ae72121a79c35578fbe82f374e161bda3b2ffceb083bf358`.
Every r2 rule remains registered unless this amendment explicitly replaces it.

**Failure motivating this amendment:** run `29665254652` yielded 0/50 estimable
hazard targets. Its r2 draft release and artifacts are retained without
overwrite and are not an interpretable neural estimand. The run also exposed
overflow/divide-by-zero in the cloglog implementation, loss of candidate-level
failure detail, serial mouse-scale recomputation, and an aggregate packaging
reference to the nonexistent `run/input/environment-input.sha256`.

**Freeze rule:** r3 remains DRAFT. It does not set a numeric SESOI, open
CONFIRM, or make `confirm_ready=true`.

## §A — immutable HMM input and separate provenance

r3 does not refit behavior. It consumes the public release
`neural-dev-v4-hmm-29482249873-r2` and requires schema
`neural-dev-v4-hmm-release-v1`, 263 registered fits, zero nonestimable fits,
the cache/environment hashes recorded by that release, and the r2 prereg hash
above. The r3 result records `hmm_prereg_sha256` and
`hazard_prereg_sha256` separately; they are not required to match.

The primary predictive posterior remains the behavior-only, target-external
\(p(z_t\mid y_{1:t-1},x_{1:t})\) from the \(K=2\) model. K=1 no-state hazard
and K=1/K=3 behavioral adequacy remain sensitivities. No smoothed posterior is
permitted.

## §B — r3 hazard likelihood

The r2 trial universe, actual-frame risk bins, causal neural history, fixed
raised-cosine basis grid \(B\in\{1,2,3,4\}\), ten deterministic 50-cell seeds,
ridge grid \(10^{-4},\ldots,10^2\), and offset
\(\log(\mathrm{bin\ width})\) remain unchanged.

The ordinal-bin baseline coefficients \(\alpha_j\) remain one coefficient per
observed ordinal frame bin. This amendment replaces the r2 unpenalized-baseline
rule: every fitted coefficient, including every \(\alpha_j\), receives the same
candidate ridge penalty \(\lambda\). The offset is fixed and is neither a
parameter nor penalized. No intercept, spline, additional penalty, warm start,
or fallback optimizer is introduced. Every \(\lambda\) starts from zero and
uses the registered L-BFGS-B settings.

Training and scoring use one float64 cloglog kernel. For an event row the
log-likelihood is evaluated as stable
`log1mexp(-exp(eta))`; its derivative uses
\(\exp(\eta)/\operatorname{expm1}(\exp(\eta))\) with the correct low- and
high-tail limits. Survival rows retain \(-\exp(\eta)\) wherever representable.
Optimizer probes above the finite guard use a continuous, equal-derivative
extension. A final fit or held-out non-event prediction that still requires
that extension is `hazard_nonrepresentable_prediction`, never an estimable
approximation.

Every fit records objective, gradient norm, optimizer status/message, runtime,
minimum/maximum \(\eta\), low/high event-tail counts, and protection-branch
count. Risk rows, training-only transforms, and design matrices are built once
per session/model/basis/seed group and reused across the seven penalties.

## §C — common tuning-session preflight

For each target, a common tuning-session eligibility set is frozen before any
candidate result is inspected. A tuning session is eligible only when:

1. its nested predictive posterior is finite and trial-aligned;
2. neural support exists and at least 50 cells are available;
3. each test block 2–5 has at least one primary-risk trial; and
4. each corresponding expanding training prefix has at least one hazard event.

Every excluded session is retained in `hazard_preflight.parquet` with
`hazard_tuning_session_ineligible` and the concrete detail. At least two
eligible sessions are required; otherwise the target is
`hazard_tuning_insufficient_sessions`. M0, M1, M2, and K=1 use exactly this
same frozen set.

## §D — complete candidates and one-SE selection

An M0 candidate is one \(\lambda\) and must succeed in every eligible session.
An M1 or M2 candidate is one \((B,\lambda)\) and must succeed in every eligible
session and all ten cell seeds. All attempted failure rows, exact exception
types, and block/seed status remain machine-readable. An incomplete candidate
does not enter one-SE selection and cannot drop a session or seed selectively;
it does not invalidate another complete candidate.

One-SE is evaluated only over complete candidates. Within one SE of the best
mean held-out per-trial likelihood it selects the smallest \(B\), then the
largest \(\lambda\). If no candidate is complete, the affected model is
`hazard_no_complete_candidate`.

## §E — outer evaluation and failure isolation

Outer evaluation still requires all four test blocks. M1, M2, dF/F, and K=1
each require all ten registered seeds when applicable. Primary target
estimability depends only on K2-events M0 and K2-events M1. An M2, dF/F, or K1
failure is reported independently and cannot alter primary session, mouse, or
group coverage. A primary M0 or M1 failure makes that target nonestimable.

The registered primary remains events K2 `M1-M0`. M2 `M2-M1`, dF/F using the
event-selected M1 hyperparameters, and K1 no-state `M1-M0` remain secondary or
sensitivity results.

## §F — target shards, checkpoints, and release

Formal execution uses an exact matrix of the 50 immutable active target
sessions, with at most ten concurrent standard runners. Each target checkpoint
binds target/mouse, cache and HMM release hashes, both prereg hashes,
environment, code commit, and all registered grids.

Atomic checkpoint groups are:

- preflight;
- each `(analysis, model, basis, cell_seed)` tuning group, containing all
  penalties and all eligible sessions;
- each selected-model `(analysis, model, signal, cell_seed)` evaluation; and
- the final target manifest.

A group is restored only if its manifest, file checksums, and complete
provenance match exactly. Registered statistical failures are complete
checkpoints. Backend/native failures fail the job, while the workflow uploads
already completed groups with `always()`.

Aggregation accepts exactly 50 unique, provenance-matched target shards.
Missing, duplicate, corrupt, or mismatched shards are pipeline failures.
Statistical nonestimability is publishable. The new tag suffix is `r3`; a
public tag is never overwritten. Packaging uses
`run/acceptance/environment-input.sha256`.

Every formal manifest fixes:

- `numeric_sesoi=null`;
- `confirm_ready=false`;
- `confirm_data_accessed=false`; and
- `allen_nwb_download=false`.

The r2 draft and its single asset are not edited. Running acceptance creates no
analysis release. Starting the expensive DEV workflow requires an explicit
`workflow_dispatch` with `mode=dev`.

## §G — registered acceptance requirements

Acceptance does not run recovery simulation. It must pass stable-cloglog
value/gradient references (including extreme tails), finite-difference/JAX
agreement, separation fixtures without warnings, baseline ridge-mask and
zero-event-bin tests, preflight and two-session rules, complete-candidate
selection, failure isolation, checkpoint resume/corruption checks, exact-50
planning and aggregation, packaging-path checks, future-information
invariance, post-lick exclusion, alignment, 8/10 coverage, closed-CONFIRM
assertions, and unchanged legacy requirement hashes.

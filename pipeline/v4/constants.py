"""Frozen constants from ``docs/prereg_v4.md``."""

from __future__ import annotations

CACHE_SCHEMA = "neural-dev-time-cache-v2"
HMM_CHECKPOINT_SCHEMA = "neural-dev-v4-hmm-checkpoint-v1"
HMM_RELEASE_SCHEMA = "neural-dev-v4-hmm-release-v1"
HMM_METHOD_REVISION = "r2"
MOUSE_SCHEMA = "neural-dev-v4-mouse-v3"
TARGET_SCHEMA = "neural-dev-v4-target-v1"
HAZARD_PLAN_SCHEMA = "neural-dev-v4-hazard-plan-v1"
HAZARD_CHECKPOINT_SCHEMA = "neural-dev-v4-hazard-checkpoint-v1"
RESULT_SCHEMA = "neural-dev-v4"
METHOD_REVISION = "r3"

WINDOW_START = -1.25
WINDOW_END = 0.75
RISK_START = 0.15
RISK_END = 0.75

PRIMARY_HMM_K = 2
HMM_SENSITIVITY_K = (1, 3)
HMM_K_GRID = (1, 2, 3)
HMM_SEEDS = tuple(range(4100, 4120))
HMM_MAX_ITER = 500
HMM_TOL = 1e-4

CELL_COUNT = 50
CELL_SEEDS = tuple(range(10))
BASIS_GRID = (1, 2, 3, 4)
RIDGE_GRID = tuple(10.0**power for power in range(-4, 3))
N_BLOCKS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 4201
REQUIRED_MICE = 8

EMISSION_ABORT = 0
EMISSION_TASK_RESPONSE = 1
EMISSION_WITHHOLD = 2
EMISSION_MISSING = 3

CACHE_SUFFIXES = (
    "time.h5",
    "trials.parquet",
    "stim.parquet",
    "licks.parquet",
    "rewards.parquet",
    "eye.parquet",
    "running.parquet",
    "meta.json",
)

TYPED_REASONS = {
    "source_integrity_failure",
    "trial_alignment_failure",
    "invalid_timestamp_grid",
    "hmm_insufficient_training_sessions",
    "hmm_inner_candidate_failure",
    "hmm_no_converged_initialization",
    "hmm_nonfinite_predictive_posterior",
    "hmm_backend_failure",
    "runtime_resource_exhaustion",
    "checkpoint_integrity_failure",
    "neural_fewer_than_50_cells",
    "hazard_no_training_event",
    "hazard_tuning_insufficient_sessions",
    "hazard_tuning_session_ineligible",
    "hazard_no_complete_candidate",
    "hazard_nonrepresentable_prediction",
    "hazard_nonconvergence",
    "hazard_incomplete_prediction",
    "hazard_empty_test_block",
    "nonestimable_mouse_coverage",
    "not_applicable_k1",
}

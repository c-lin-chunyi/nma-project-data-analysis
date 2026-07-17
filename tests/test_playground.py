from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from nma_play.decoder import (
    DecoderConfig,
    contiguous_purged_folds,
    deterministic_cell_indices,
    run_q1_decoder,
)
from nma_play.release import (
    FEATURE_NAMES,
    ReleaseDataError,
    _safe_extract_tar,
    load_behavioral_scan,
    load_feature_cache,
    sha256_file,
)


def _write_sums(directory: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def make_behavior_assets(root: Path) -> Path:
    assets = root / "behavior-assets"
    payload = root / "behavior-payload"
    assets.mkdir()
    payload.mkdir()
    n = 80
    trial = np.arange(n)
    late = trial % 4 == 0
    early = trial % 4 == 1
    miss = trial % 4 == 2
    aborted = trial % 4 == 3
    labels = pd.DataFrame({
        "behavior_session_id": 1001,
        "mouse_id": 11,
        "project_code": "VisualBehavior",
        "equipment_name": "CAM2P.4",
        "session_type": "OPHYS_1_images_A",
        "trial_id": trial,
        "trial_index": trial,
        "start_time": trial * 7.0,
        "stop_time": trial * 7.0 + 6,
        "change_time": trial * 7.0 + 3,
        "hit": late | early,
        "late_hit": late,
        "early_hit": early,
        "miss": miss,
        "aborted": aborted,
        "go": late | early | miss,
        "response_latency": np.where(late | early, 0.5, np.nan),
        "latency_status": "fixture",
        "reward_rate": 2 + np.sin(trial / 8),
        "bout_rate": 5 + np.cos(trial / 9),
        "rate_span_minutes": 2.0,
        "engaged_A": trial >= 10,
        "keep_A": trial % 9 != 0,
        "engaged_B": trial >= 5,
        "keep_B": trial % 11 != 0,
        "engaged_A_hysteretic": trial >= 15,
        "keep_A_hysteretic": trial % 13 != 0,
        "impulsive_regime": trial % 10 == 0,
        "first_ten": trial < 10,
        "is_image_novel": False,
    })
    session = pd.DataFrame([{
        "behavior_session_id": 1001,
        "mouse_id": 11,
        "project_code": "VisualBehavior",
        "equipment_name": "CAM2P.4",
        "session_type": "OPHYS_1_images_A",
        "n_trials": n,
        "n_hit": int((late | early).sum()),
        "n_late_hit": int(late.sum()),
        "n_early_hit": int(early.sum()),
        "n_miss": int(miss.sum()),
        "abort_frac": float(aborted.mean()),
        "survived_frac": float(1 - aborted.mean()),
        "contam": 0.1,
        "contam_n": 20,
        "contam_status": "eligible",
        "eng_A": 0.5,
        "eng_B": 0.7,
        "eng_A_hysteretic": 0.4,
        "late_hit_A": 15,
        "miss_A": 15,
        "late_hit_B": 20,
        "miss_B": 20,
        "late_hit_A_hysteretic": 12,
        "miss_A_hysteretic": 12,
        "impulsive_frac": 0.1,
        "impulsive_go": 4,
        "total_go": 60,
        "impulsive_abort_rate": 0.2,
        "nonimpulsive_abort_rate": 0.25,
        "behavioral_eligible": True,
        "eligibility_reasons": "",
    }])
    eligibility = session[[
        "behavior_session_id", "mouse_id", "behavioral_eligible",
        "eligibility_reasons", "late_hit_B", "miss_B", "contam", "contam_n",
        "contam_status",
    ]]
    persistence = pd.DataFrame([{
        "behavior_session_id": 1001, "mouse_id": 11,
        "project_code": "VisualBehavior", "equipment_name": "CAM2P.4",
        "session_type": "OPHYS_1_images_A", "impulsive_frac": 0.1,
        "max_run": 5, "n_runs": 2, "null_mean_frac": 0.1,
        "null_p_max_run": 0.5,
    }])
    guard = pd.DataFrame([{
        "behavior_session_id": 1001, "mouse_id": 11,
        "project_code": "VisualBehavior", "equipment_name": "CAM2P.4",
        "session_type": "OPHYS_1_images_A", "raw_go_A": 60, "kept_go_A": 50,
    }])
    sweep = pd.DataFrame([
        {
            "construct": construct, "miss_threshold": threshold,
            "min_late_hit": 20, "n_sessions": 1, "n_mice": 1,
            "median_late_hit": 20, "median_miss": 20,
        }
        for construct in ("v3.1_B_K50", "v3_A_all_C0.1")
        for threshold in (10, 15, 20, 25, 30)
    ])
    tables = {
        "_trial_labels.parquet": labels,
        "_session_scan.parquet": session,
        "_eligibility.parquet": eligibility,
        "_persistence.parquet": persistence,
        "_guard_diagnostics.parquet": guard,
        "_yield_sweep.parquet": sweep,
    }
    for name, frame in tables.items():
        frame.to_parquet(payload / name, index=False)
    manifest = {
        "schema": "behavioral-v3.1",
        "n_dev_sessions": 1,
        "primary": "B-engaged late-hit-vs-miss",
    }
    (assets / "behavioral-manifest.json").write_text(json.dumps(manifest))
    archive = assets / "behavioral-v3.1-scan.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(payload.iterdir()):
            tar.add(path, arcname=path.name)
    _write_sums(assets, [archive.name, "behavioral-manifest.json"])
    return assets


def make_feature_assets(root: Path, *, corrupt_h5: bool = False) -> Path:
    assets = root / "feature-assets"
    payload = root / "feature-payload"
    assets.mkdir()
    payload.mkdir()
    rng = np.random.default_rng(7)
    oeid, container, n_trials, n_cells = 2001, 3001, 100, 80
    trial_ids = np.arange(n_trials)
    y = trial_ids % 2 == 0
    labels = pd.DataFrame({
        "trial_id": trial_ids,
        "behavior_session_id": 4001,
        "mouse_id": 11,
        "project_code": "VisualBehavior",
        "equipment_name": "CAM2P.4",
        "session_type": "OPHYS_1_images_A",
        "trial_index": trial_ids,
        "start_time": trial_ids * 7.0,
        "stop_time": trial_ids * 7.0 + 6,
        "change_time": trial_ids * 7.0 + 3,
        "hit": y,
        "late_hit": y,
        "early_hit": False,
        "miss": ~y,
        "aborted": False,
        "go": True,
        "response_latency": np.where(y, 0.5, np.nan),
        "latency_status": "fixture",
        "reward_rate": 2.0,
        "bout_rate": 5.0,
        "rate_span_minutes": 2.0,
        "engaged_A": True,
        "keep_A": True,
        "engaged_B": True,
        "keep_B": True,
        "engaged_A_hysteretic": True,
        "keep_A_hysteretic": True,
        "impulsive_regime": False,
        "first_ten": trial_ids < 10,
        "is_image_novel": False,
    })
    q2 = pd.DataFrame({
        "trial_id": trial_ids,
        "change_time": trial_ids * 7.0 + 3,
        "session_position": trial_ids / (n_trials - 1),
        "transition": "a->b",
        "flashes_before_change": 4.0,
        "preceding_omission": False,
        "time_since_previous_change": np.r_[np.nan, np.repeat(7.0, n_trials - 1)],
        "time_since_previous_lick": 2.0,
        "time_since_previous_reward": 4.0,
        "previous_outcome": np.where(y, "hit", "miss"),
        "pre_change_pupil": rng.normal(1, 0.1, n_trials),
        "pupil_missing_frac": 0.0,
        "pre_change_running": rng.normal(10, 2, n_trials),
        "running_missing_frac": 0.0,
        "q2_covariates_complete": True,
    })
    labels.to_parquet(payload / f"{oeid}.labels.parquet", index=False)
    q2.to_parquet(payload / f"{oeid}.q2.parquet", index=False)
    h5_path = payload / f"{oeid}.features.h5"
    if corrupt_h5:
        h5_path.write_bytes(b"not an hdf5 file")
    else:
        signal = rng.normal(size=(n_trials, n_cells)).astype("float32")
        signal[:, :5] += y[:, None] * 0.35
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("trial_id", data=trial_ids)
            h5.create_dataset("cell_specimen_id", data=np.arange(5000, 5000 + n_cells))
            for offset, name in enumerate(FEATURE_NAMES):
                h5.create_dataset(name, data=signal + offset * 0.02)
    meta = {
        "schema": "neural-dev-feature-cache-v1",
        "identity": {
            "ophys_experiment_id": oeid,
            "behavior_session_id": 4001,
            "ophys_session_id": 5001,
            "ophys_container_id": container,
            "mouse_id": 11,
            "project_code": "VisualBehavior",
            "session_type": "OPHYS_1_images_A",
            "equipment_name": "CAM2P.4",
            "imaging_depth": 175,
            "targeted_structure": "VISp",
            "file_id": 1,
            "role": "active",
        },
        "n_trials": n_trials,
        "n_cells": n_cells,
        "features": list(FEATURE_NAMES),
    }
    (payload / f"{oeid}.feature-meta.json").write_text(json.dumps(meta))
    (payload / f"_features-container-{container}.json").write_text(
        json.dumps({"container_id": container, "complete": True})
    )
    archive = assets / f"feature-container-{container}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(payload.iterdir()):
            tar.add(path, arcname=path.name)
    manifest = {
        "schema": "neural-dev-feature-cache-v1",
        "n_active_experiments": 1,
        "n_source_experiments": 1,
        "n_containers": 1,
        "parts": [{
            "name": archive.name,
            "sha256": sha256_file(archive),
            "size": archive.stat().st_size,
        }],
        "fold_independent": True,
        "contains_oof_predictions": False,
        "allen_nwb_download": False,
    }
    validation = {
        "schema": "neural-dev-feature-cache-v1",
        "n_experiments": 1,
        "n_containers": 1,
        "complete": True,
        "failures": [],
        "allen_nwb_download": False,
    }
    experiments = pd.DataFrame([meta["identity"]])
    (assets / "feature-cache-manifest.json").write_text(json.dumps(manifest))
    (assets / "feature-cache-validation.json").write_text(json.dumps(validation))
    experiments.to_csv(assets / "dev_experiments.csv", index=False)
    _write_sums(assets, [
        "feature-cache-manifest.json",
        "feature-cache-validation.json",
        "dev_experiments.csv",
        archive.name,
    ])
    return assets


def test_behavior_loader_validates_and_reuses_cache(tmp_path):
    source = make_behavior_assets(tmp_path)
    cache = tmp_path / "cache"
    first = load_behavioral_scan(cache, source_dir=source, show_progress=False)
    assert len(first.trial_labels) == 80
    assert first.session_scan.behavior_session_id.tolist() == [1001]
    (source / "behavioral-v3.1-scan.tar.gz").unlink()
    second = load_behavioral_scan(cache, source_dir=source, show_progress=False)
    assert len(second.trial_labels) == 80


def test_behavior_loader_rejects_checksum_mismatch(tmp_path):
    source = make_behavior_assets(tmp_path)
    with (source / "behavioral-v3.1-scan.tar.gz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ReleaseDataError, match="SHA-256 mismatch"):
        load_behavioral_scan(tmp_path / "cache", source_dir=source, show_progress=False)


def test_safe_tar_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        content = b"escape"
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    with pytest.raises(ReleaseDataError, match="Unsafe path"):
        _safe_extract_tar(archive, tmp_path / "out")


def test_feature_cache_schema_and_alignment(tmp_path):
    source = make_feature_assets(tmp_path)
    cache = load_feature_cache(
        tmp_path / "cache", source_dir=source, show_progress=False
    )
    assert cache.experiment_ids == [2001]
    assert tuple(cache.matrix(2001).values.shape) == (100, 80)
    assert cache.index[["n_trials", "n_cells"]].iloc[0].tolist() == [100, 80]
    np.testing.assert_array_equal(
        cache.labels(2001).trial_id, cache.matrix(2001).trial_ids
    )


def test_feature_cache_rejects_corrupt_hdf5(tmp_path):
    source = make_feature_assets(tmp_path, corrupt_h5=True)
    with pytest.raises(ReleaseDataError, match="Could not read feature-cache files"):
        load_feature_cache(
            tmp_path / "cache", source_dir=source, show_progress=False
        )


def _pipeline_module():
    path = Path(__file__).parents[1] / "pipeline" / "verify-neural" / "neural.py"
    spec = importlib.util.spec_from_file_location("pipeline_neural", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decoder_cell_selection_and_folds_match_pipeline():
    pipeline = _pipeline_module()
    X = np.tile(np.arange(80), (100, 1))
    ours = deterministic_cell_indices(80, 50, 3, 2001)
    theirs = pipeline._subset_cells(X, 50, 3, 2001)[0].astype(int)
    np.testing.assert_array_equal(ours, theirs)
    raw = np.arange(100)
    for (our_train, our_test), (their_train, their_test) in zip(
        contiguous_purged_folds(raw, 5, 10), pipeline._folds(raw, 5, 10)
    ):
        np.testing.assert_array_equal(our_train, their_train)
        np.testing.assert_array_equal(our_test, their_test)


def test_decoder_is_deterministic_and_typed(tmp_path):
    source = make_feature_assets(tmp_path)
    cache = load_feature_cache(
        tmp_path / "cache", source_dir=source, show_progress=False
    )
    matrix = cache.matrix(2001)
    labels = cache.labels(2001)
    config = DecoderConfig(k=50, C=1e-4, n_seeds=2)
    first = run_q1_decoder(matrix, labels, config)
    second = run_q1_decoder(matrix, labels, config)
    assert first.status == "estimable"
    pd.testing.assert_frame_equal(first.seed_metrics, second.seed_metrics)
    pd.testing.assert_frame_equal(first.oof, second.oof)
    low_cells = run_q1_decoder(
        matrix, labels, DecoderConfig(k=100, n_seeds=1)
    )
    assert low_cells.status == "nonestimable"
    assert low_cells.reason == "low_cells"

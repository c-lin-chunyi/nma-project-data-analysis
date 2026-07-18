"""Materialize and validate the fold-independent time-resolved v4 cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import CACHE_SCHEMA, CACHE_SUFFIXES, WINDOW_END, WINDOW_START


SOURCE_SUFFIXES = {
    "trials": "trials.parquet",
    "stim": "stim.parquet",
    "licks": "licks.parquet",
    "rewards": "rewards.parquet",
    "eye": "eye.parquet",
    "running": "running.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_paths(root: Path, experiment_id: int) -> list[Path]:
    return [root / f"{int(experiment_id)}.{suffix}" for suffix in CACHE_SUFFIXES]


def _trial_ids(trials: pd.DataFrame) -> np.ndarray:
    values = (
        trials["trials_id"].to_numpy()
        if "trials_id" in trials
        else trials.index.to_numpy()
    )
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0 or len(np.unique(values)) != len(values):
        raise ValueError("trials must have nonempty unique integer trial IDs")
    return values


def _change_times(trials: pd.DataFrame) -> np.ndarray:
    if "change_time" not in trials:
        return np.full(len(trials), np.nan)
    return pd.to_numeric(trials["change_time"], errors="coerce").to_numpy(float)


def _copy_table(source: Path, staging: Path, experiment_id: int, name: str) -> Path:
    src = source / f"{experiment_id}.{SOURCE_SUFFIXES[name]}"
    if not src.is_file():
        raise ValueError(f"source experiment {experiment_id} lacks {src.name}")
    dst = staging / f"{experiment_id}.{SOURCE_SUFFIXES[name]}"
    shutil.copy2(src, dst)
    return dst


def materialize_experiment(
    source: Path,
    out: Path,
    row,
    *,
    neural_release: str,
    data_manifest_sha256: str,
) -> dict:
    """Atomically write one active experiment in ragged actual-frame form."""

    import h5py

    experiment_id = int(row.ophys_experiment_id)
    staging = out / ".staging" / str(experiment_id)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        tables = {
            name: pd.read_parquet(
                _copy_table(source, staging, experiment_id, name)
            )
            for name in SOURCE_SUFFIXES
        }
        trials = tables["trials"].reset_index(drop=True)
        trial_ids = _trial_ids(trials)
        changes = _change_times(trials)

        source_h5 = source / f"{experiment_id}.neural.h5"
        if not source_h5.is_file():
            raise ValueError(f"source experiment {experiment_id} lacks neural HDF5")
        target_h5 = staging / f"{experiment_id}.time.h5"
        with h5py.File(source_h5, "r") as src:
            required = {
                "ophys_timestamps",
                "cell_specimen_id",
                "cell_roi_id",
                "events",
                "dff",
            }
            missing = sorted(required - set(src.keys()))
            if missing:
                raise ValueError(f"source experiment {experiment_id} missing {missing}")
            timestamps = np.asarray(src["ophys_timestamps"][:], dtype=np.float64)
            cells = np.asarray(src["cell_specimen_id"][:], dtype=np.int64)
            rois = np.asarray(src["cell_roi_id"][:], dtype=np.int64)
            if (
                len(timestamps) < 2
                or not np.all(np.isfinite(timestamps))
                or not np.all(np.diff(timestamps) > 0)
            ):
                raise ValueError("invalid_timestamp_grid")
            if len(cells) == 0 or len(np.unique(cells)) != len(cells):
                raise ValueError("empty or duplicate canonical cell IDs")
            expected = (len(cells), len(timestamps))
            if src["events"].shape != expected or src["dff"].shape != expected:
                raise ValueError("continuous neural trace shape mismatch")

            valid = (
                np.isfinite(changes)
                & ((changes + WINDOW_START) >= timestamps[0])
                & ((changes + WINDOW_END) <= timestamps[-1])
            )
            valid_ids = trial_ids[valid]
            valid_changes = changes[valid]
            offsets = [0]
            source_indices: list[np.ndarray] = []
            for change in valid_changes:
                left = int(np.searchsorted(timestamps, change + WINDOW_START, "left"))
                right = int(np.searchsorted(timestamps, change + WINDOW_END, "left"))
                indices = np.arange(left, right, dtype=np.int64)
                if len(indices) == 0:
                    raise ValueError(f"empty neural window around change {change}")
                source_indices.append(indices)
                offsets.append(offsets[-1] + len(indices))
            flat = (
                np.concatenate(source_indices)
                if source_indices
                else np.empty(0, dtype=np.int64)
            )
            repeated_change = np.repeat(
                valid_changes, np.diff(np.asarray(offsets, dtype=np.int64))
            )

            with h5py.File(target_h5, "w") as dst:
                dst.attrs.update(
                    schema=CACHE_SCHEMA,
                    ophys_experiment_id=experiment_id,
                    source_neural_release=neural_release,
                    source_data_manifest_sha256=data_manifest_sha256,
                    window_start=WINDOW_START,
                    window_end=WINDOW_END,
                    actual_timestamps=True,
                    resampled=False,
                    fold_independent=True,
                    contains_fitted_values=False,
                )
                dst.create_dataset("cell_specimen_id", data=cells)
                dst.create_dataset("cell_roi_id", data=rois)
                dst.create_dataset("trial_id", data=valid_ids)
                dst.create_dataset("frame_offsets", data=np.asarray(offsets, np.int64))
                dst.create_dataset("source_frame_index", data=flat)
                absolute = timestamps[flat] if len(flat) else np.empty(0, float)
                dst.create_dataset("frame_timestamp", data=absolute)
                dst.create_dataset("relative_time", data=absolute - repeated_change)
                chunk_frames = max(1, min(256, len(flat)))
                chunk_cells = max(1, min(64, len(cells)))
                for signal in ("events", "dff"):
                    dataset = dst.create_dataset(
                        signal,
                        shape=(len(flat), len(cells)),
                        dtype=np.float32,
                        chunks=(chunk_frames, chunk_cells),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    for trial_pos, indices in enumerate(source_indices):
                        lo, hi = offsets[trial_pos], offsets[trial_pos + 1]
                        dataset[lo:hi] = np.asarray(src[signal][:, indices], np.float32).T

        flow = trials.copy()
        flow.insert(0, "trial_id_v4", trial_ids)
        flow["neural_valid"] = valid
        flow.to_parquet(
            staging / f"{experiment_id}.trials.parquet", index=False
        )

        meta = {
            "schema": CACHE_SCHEMA,
            "ophys_experiment_id": experiment_id,
            "behavior_session_id": int(row.behavior_session_id),
            "ophys_container_id": int(row.ophys_container_id),
            "mouse_id": int(row.mouse_id),
            "role": "active",
            "source_neural_release": neural_release,
            "source_data_manifest_sha256": data_manifest_sha256,
            "n_raw_trials": int(len(trials)),
            "n_neural_valid_trials": int(valid.sum()),
            "n_cells": int(len(cells)),
            "n_ragged_frames": int(len(flat)),
            "actual_timestamps": True,
            "resampled": False,
            "allen_nwb_download": False,
        }
        (staging / f"{experiment_id}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        expected_names = {path.name for path in cache_paths(staging, experiment_id)}
        actual_names = {path.name for path in staging.iterdir() if path.is_file()}
        if expected_names != actual_names:
            raise ValueError(
                f"cache package mismatch missing={sorted(expected_names-actual_names)} "
                f"extra={sorted(actual_names-expected_names)}"
            )
        out.mkdir(parents=True, exist_ok=True)
        for source_path in cache_paths(staging, experiment_id):
            source_path.replace(out / source_path.name)
        return meta
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def validate_source_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "ophys_experiment_id",
        "behavior_session_id",
        "ophys_container_id",
        "mouse_id",
        "role",
    }
    if not required.issubset(manifest.columns):
        raise ValueError(f"source manifest missing {sorted(required-set(manifest.columns))}")
    if (
        len(manifest) != 70
        or manifest.ophys_experiment_id.astype(int).nunique() != 70
        or manifest.ophys_container_id.astype(int).nunique() != 10
        or int(manifest.role.eq("active").sum()) != 50
        or int(manifest.role.eq("passive").sum()) != 20
    ):
        raise ValueError("v4 source must be exactly 50 active + 20 passive / 10 containers")


def materialize_container(
    source: Path,
    manifest_path: Path,
    out: Path,
    container: int,
    *,
    neural_release: str,
    data_manifest_sha256: str,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    validate_source_manifest(manifest)
    selected = manifest[
        manifest.role.eq("active")
        & manifest.ophys_container_id.astype(int).eq(int(container))
    ]
    if len(selected) != 5:
        raise ValueError(f"container {container} must contain exactly five active experiments")
    rows, failures = [], []
    for row in selected.itertuples(index=False):
        try:
            rows.append(
                materialize_experiment(
                    source,
                    out,
                    row,
                    neural_release=neural_release,
                    data_manifest_sha256=data_manifest_sha256,
                )
            )
        except Exception as exc:  # preserve every experiment failure in the report
            failures.append(
                {
                    "ophys_experiment_id": int(row.ophys_experiment_id),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    report = {
        "schema": CACHE_SCHEMA,
        "container_id": int(container),
        "n_expected": 5,
        "n_complete": len(rows),
        "complete": not failures and len(rows) == 5,
        "experiments": rows,
        "failures": failures,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"_time-container-{int(container)}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["complete"]:
        raise ValueError(f"cache materialization failed: {failures}")
    return report


def _numeric_cache_files(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name[:1].isdigit()
    }


def verify_cache(
    root: Path, manifest_path: Path, report_path: Path | None = None
) -> dict:
    import h5py

    manifest = pd.read_csv(manifest_path)
    validate_source_manifest(manifest)
    active = manifest[manifest.role.eq("active")].copy()
    expected = {
        path.name
        for experiment_id in active.ophys_experiment_id.astype(int)
        for path in cache_paths(root, experiment_id)
    }
    failures: list[dict] = []
    actual = _numeric_cache_files(root)
    if actual != expected:
        failures.append(
            {
                "reason": "package_set_mismatch",
                "missing": sorted(expected - actual),
                "extra": sorted(actual - expected),
            }
        )
    source_hashes: set[str] = set()
    for row in active.itertuples(index=False):
        experiment_id = int(row.ophys_experiment_id)
        try:
            meta = json.loads((root / f"{experiment_id}.meta.json").read_text())
            if meta.get("schema") != CACHE_SCHEMA:
                raise ValueError("metadata schema mismatch")
            if (
                int(meta["ophys_experiment_id"]) != experiment_id
                or int(meta["behavior_session_id"]) != int(row.behavior_session_id)
                or int(meta["ophys_container_id"]) != int(row.ophys_container_id)
                or int(meta["mouse_id"]) != int(row.mouse_id)
                or meta.get("role") != "active"
            ):
                raise ValueError("mouse identity mismatch")
            source_hashes.add(str(meta["source_data_manifest_sha256"]))
            trials = pd.read_parquet(root / f"{experiment_id}.trials.parquet")
            ids = trials["trial_id_v4"].to_numpy(np.int64)
            if len(ids) != int(meta["n_raw_trials"]) or len(np.unique(ids)) != len(ids):
                raise ValueError("trial_alignment_failure")
            with h5py.File(root / f"{experiment_id}.time.h5", "r") as h5:
                if h5.attrs.get("schema") != CACHE_SCHEMA:
                    raise ValueError("HDF5 schema mismatch")
                if (
                    int(h5.attrs["ophys_experiment_id"]) != experiment_id
                    or h5.attrs.get("source_data_manifest_sha256")
                    != meta["source_data_manifest_sha256"]
                ):
                    raise ValueError("HDF5 provenance mismatch")
                offsets = np.asarray(h5["frame_offsets"][:], np.int64)
                valid_ids = np.asarray(h5["trial_id"][:], np.int64)
                source_indices = np.asarray(h5["source_frame_index"][:], np.int64)
                rel = np.asarray(h5["relative_time"][:], float)
                absolute = np.asarray(h5["frame_timestamp"][:], float)
                cells = np.asarray(h5["cell_specimen_id"][:], np.int64)
                rois = np.asarray(h5["cell_roi_id"][:], np.int64)
                if (
                    len(cells) != int(meta["n_cells"])
                    or len(rois) != len(cells)
                    or len(np.unique(cells)) != len(cells)
                ):
                    raise ValueError("canonical-cell alignment failure")
                if len(offsets) != len(valid_ids) + 1 or offsets[0] != 0:
                    raise ValueError("invalid ragged offsets")
                if np.any(np.diff(offsets) <= 0) or offsets[-1] != len(rel):
                    raise ValueError("invalid ragged frame counts")
                if h5["events"].shape != (len(rel), len(cells)):
                    raise ValueError("events shape mismatch")
                if h5["dff"].shape != h5["events"].shape:
                    raise ValueError("dff shape mismatch")
                if not np.all(np.isfinite(absolute)):
                    raise ValueError("nonfinite frame timestamps")
                if len(source_indices) != len(absolute):
                    raise ValueError("source-frame alignment failure")
                changes_by_id = trials.set_index("trial_id_v4").change_time
                for lo, hi in zip(offsets[:-1], offsets[1:]):
                    if not np.all(np.diff(absolute[lo:hi]) > 0):
                        raise ValueError("invalid_timestamp_grid")
                    if not np.all(np.diff(source_indices[lo:hi]) == 1):
                        raise ValueError("source-frame alignment failure")
                    if rel[lo] < WINDOW_START or rel[hi - 1] >= WINDOW_END:
                        raise ValueError("ragged window outside registered support")
                for trial_id, lo, hi in zip(valid_ids, offsets[:-1], offsets[1:]):
                    change = float(changes_by_id.loc[int(trial_id)])
                    if not np.array_equal(
                        rel[lo:hi], absolute[lo:hi] - change
                    ):
                        raise ValueError("trial_alignment_failure")
                expected_valid = set(
                    trials.loc[trials.neural_valid.astype(bool), "trial_id_v4"].astype(int)
                )
                if set(map(int, valid_ids)) != expected_valid:
                    raise ValueError("trial_alignment_failure")
        except Exception as exc:
            failures.append(
                {
                    "ophys_experiment_id": experiment_id,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    if len(source_hashes) != 1:
        failures.append({"reason": "inconsistent_source_manifest_hashes"})
    report = {
        "schema": CACHE_SCHEMA,
        "complete": not failures,
        "n_source_experiments": 70,
        "n_active_experiments": 50,
        "n_passive_experiments": 20,
        "n_containers": 10,
        "fold_independent": True,
        "contains_fitted_values": False,
        "actual_timestamps": True,
        "resampled": False,
        "allen_nwb_download": False,
        "source_data_manifest_sha256": (
            next(iter(source_hashes)) if len(source_hashes) == 1 else None
        ),
        "failures": failures,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise ValueError(f"time-cache-v2 verification failed: {failures}")
    return report


def write_file_checksums(root: Path, paths: Iterable[Path], output: Path) -> None:
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in sorted(paths)]
    output.write_text("\n".join(lines) + "\n")

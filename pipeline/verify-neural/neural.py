#!/usr/bin/env python3
"""DEV-only Appendix-A extraction and v3.2 K=50 Q1/Q2 analysis.

The public contract has three commands:

  manifest  derive the exact 50 active + 20 passive experiment set from DEV
  pull      download one container shard and publish atomic, lossless bundles
  scan      run frozen K=50 Q1/Q2, the single SESOI anchor, and precision gates

No command accepts confirm_mice.csv.  Analysis selection is never baked into a
bundle: the bundle contains Allen-QC cells, continuous traces, task tables and
active trial-locked tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_START, WINDOW_END = -1.25, 1.50
FIT_START, FIT_END = 0.0, 0.30
PUPIL_START, PUPIL_END = -1.0, 0.0
GAP_RAW_TRIALS = 10
N_BLOCKS = 5
N_SEEDS = 10
PRIMARY_K = 50
FROZEN_C50 = 1e-4
CONFIRM_MICE = 29

BUNDLE_SUFFIXES = (
    "neural.h5", "trials.parquet", "stim.parquet", "licks.parquet",
    "rewards.parquet", "eye.parquet", "running.parquet", "raw_running.parquet",
    "q2.parquet", "meta.json",
)


def parse_shard(value: str) -> tuple[int, int]:
    try:
        k, n = map(int, value.split("/"))
    except Exception as exc:
        raise ValueError(f"invalid shard {value!r}; expected k/N") from exc
    if n < 1 or not 1 <= k <= n:
        raise ValueError(f"invalid shard {value!r}; require 1 <= k <= N")
    return k, n


def bundle_paths(out: Path, oeid: int) -> list[Path]:
    return [out / f"{int(oeid)}.{suffix}" for suffix in BUNDLE_SUFFIXES]


def bundle_complete(out: Path, oeid: int) -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in bundle_paths(out, oeid))


def build_experiment_manifest(dev: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    required = {"ophys_experiment_id", "ophys_container_id", "mouse_id"}
    if not required.issubset(dev.columns):
        raise ValueError(f"DEV table missing {sorted(required - set(dev.columns))}")
    table = experiments.reset_index() if "ophys_experiment_id" not in experiments.columns else experiments.copy()
    if not required.union({"session_type"}).issubset(table.columns):
        raise ValueError("Allen experiment table lacks Appendix-A identity columns")
    dev_active = set(dev["ophys_experiment_id"].astype(int))
    containers = set(dev["ophys_container_id"].astype(int))
    selected = table[table["ophys_container_id"].astype("Int64").isin(containers)].copy()
    selected["ophys_experiment_id"] = selected["ophys_experiment_id"].astype(int)
    selected["ophys_container_id"] = selected["ophys_container_id"].astype(int)
    passive = selected["session_type"].astype(str).str.match(r"OPHYS_[25]_.*_passive$")
    selected["role"] = np.where(passive, "passive", "active")
    found_active = set(selected.loc[selected.role.eq("active"), "ophys_experiment_id"])
    if found_active != dev_active:
        raise ValueError(f"active DEV mismatch; missing={sorted(dev_active-found_active)}, "
                         f"extra={sorted(found_active-dev_active)}")
    if len(selected) != 70 or int(passive.sum()) != 20 or len(containers) != 10:
        raise ValueError(f"Appendix-A set must be 50 active + 20 passive in 10 containers; "
                         f"got active={int((~passive).sum())}, passive={int(passive.sum())}, "
                         f"containers={len(containers)}")
    keep = [c for c in ("ophys_experiment_id", "behavior_session_id", "ophys_session_id",
                        "ophys_container_id", "mouse_id", "project_code", "session_type",
                        "equipment_name", "imaging_depth", "targeted_structure", "file_id",
                        "role") if c in selected.columns]
    return selected[keep].sort_values(["ophys_container_id", "role", "ophys_experiment_id"])


def make_manifest(dev_path: Path, out: Path, cache_dir: Path) -> int:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    dev = pd.read_csv(dev_path)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    result = build_experiment_manifest(dev, cache.get_ophys_experiment_table())
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"wrote {len(result)} experiments / {result.ophys_container_id.nunique()} containers")
    return 0


def _safe_table(value, name: str) -> tuple[pd.DataFrame, str | None]:
    try:
        table = value() if callable(value) else value
        if table is None:
            return pd.DataFrame({"_missing": pd.Series(dtype=bool)}), f"{name}:missing"
        return pd.DataFrame(table).reset_index(), None
    except Exception as exc:
        return pd.DataFrame({"_missing": [True]}), f"{name}:{type(exc).__name__}:{exc}"


def _timestamp_column(df: pd.DataFrame) -> str | None:
    return next((c for c in ("timestamps", "timestamp", "time") if c in df.columns), None)


def _pupil_column(df: pd.DataFrame) -> str | None:
    return next((c for c in ("pupil_area", "pupil_area_raw", "pupil_width") if c in df.columns), None)


def _running_column(df: pd.DataFrame) -> str | None:
    return next((c for c in ("speed", "running_speed", "velocity") if c in df.columns), None)


def _prechange_feature(times: np.ndarray, values: np.ndarray, align: np.ndarray,
                       *, normalize_median=False) -> tuple[np.ndarray, np.ndarray]:
    times, values = np.asarray(times, float), np.asarray(values, float)
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    if normalize_median and len(values):
        median = np.nanmedian(values)
        if np.isfinite(median) and median != 0:
            values = values / median
    result, missing = np.full(len(align), np.nan), np.ones(len(align))
    for i, center in enumerate(align):
        if not np.isfinite(center) or not len(times):
            continue
        grid = np.arange(center + PUPIL_START, center + PUPIL_END, 0.05)
        if not len(grid):
            continue
        inside = (grid >= times.min()) & (grid <= times.max())
        interp = np.full(len(grid), np.nan)
        interp[inside] = np.interp(grid[inside], times, values)
        # Do not bridge source gaps >=0.5 seconds.
        pos = np.searchsorted(times, grid)
        left, right = np.clip(pos - 1, 0, len(times)-1), np.clip(pos, 0, len(times)-1)
        exact = np.isclose(grid, times[left], atol=1e-9) | np.isclose(
            grid, times[right], atol=1e-9)
        interp[((times[right] - times[left]) >= 0.5) & ~exact] = np.nan
        missing[i] = float(np.isnan(interp).mean())
        if missing[i] <= .20:
            result[i] = float(np.nanmean(interp))
    return result, missing


def _trial_locked_scalar(table: pd.DataFrame, value_column: str | None,
                         align: np.ndarray, rel: np.ndarray, *,
                         normalize_median=False) -> np.ndarray:
    """Interpolate a scalar trace without bridging source gaps >=0.5 seconds."""
    result = np.full((len(align), len(rel)), np.nan, dtype=np.float32)
    time_column = _timestamp_column(table)
    if not time_column or not value_column:
        return result
    times = table[time_column].to_numpy(float)
    values = table[value_column].to_numpy(float)
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    if not len(times):
        return result
    order = np.argsort(times); times, values = times[order], values[order]
    if normalize_median:
        median = np.nanmedian(values)
        if np.isfinite(median) and median != 0:
            values = values / median
    for i, center in enumerate(align):
        target = center + rel
        inside = np.isfinite(center) & (target >= times[0]) & (target <= times[-1])
        if not np.any(inside):
            continue
        result[i, inside] = np.interp(target[inside], times, values).astype(np.float32)
        pos = np.searchsorted(times, target)
        left = np.clip(pos - 1, 0, len(times)-1)
        right = np.clip(pos, 0, len(times)-1)
        exact = np.isclose(target, times[left], atol=1e-9) | np.isclose(
            target, times[right], atol=1e-9)
        result[i, ((times[right] - times[left]) >= .5) & ~exact] = np.nan
    return result


def _q2_features(tr: pd.DataFrame, sp: pd.DataFrame, lk: pd.DataFrame,
                 rw: pd.DataFrame, eye: pd.DataFrame, running: pd.DataFrame) -> pd.DataFrame:
    n = len(tr)
    align = (tr["change_time"].to_numpy(float) if "change_time" in tr.columns
             else np.full(n, np.nan))
    trial_ids = (tr["trials_id"].to_numpy() if "trials_id" in tr.columns else np.arange(n))
    out = pd.DataFrame({"trial_id": trial_ids, "change_time": align,
                        "session_position": np.arange(n, dtype=float) / max(1, n-1)})
    out["transition"] = (tr.get("initial_image_name", pd.Series("unknown", index=tr.index)).astype(str)
                         + "->" + tr.get("change_image_name", pd.Series("unknown", index=tr.index)).astype(str))
    if "trials_id" in sp.columns:
        pre = sp.copy()
        if "start_time" in pre.columns:
            change_by_trial = pd.Series(align, index=trial_ids)
            pre = pre[pre["start_time"].to_numpy(float) <
                      pre["trials_id"].map(change_by_trial).to_numpy(float)]
        flashes = pre.groupby("trials_id").size()
        out["flashes_before_change"] = pd.Series(trial_ids).map(flashes).to_numpy(float)
        omitted = pre.get("omitted", pd.Series(False, index=pre.index)).fillna(False).astype(bool)
        preceding_table = pre.assign(_omitted=omitted)
        if "start_time" in preceding_table.columns:
            preceding_table = preceding_table.sort_values("start_time")
        preceding = preceding_table.groupby("trials_id")["_omitted"].last()
        out["preceding_omission"] = pd.Series(trial_ids).map(preceding).fillna(False).to_numpy(bool)
    else:
        out["flashes_before_change"] = np.nan
        out["preceding_omission"] = False
    previous_change = np.r_[np.nan, align[:-1]]
    out["time_since_previous_change"] = align - previous_change
    for label, table in (("lick", lk), ("reward", rw)):
        tc = _timestamp_column(table)
        events = np.sort(table[tc].dropna().to_numpy(float)) if tc else np.array([])
        idx = np.searchsorted(events, align) - 1
        values = np.full(n, np.nan)
        valid = idx >= 0
        values[valid] = align[valid] - events[idx[valid]]
        out[f"time_since_previous_{label}"] = values
    outcomes = np.select([tr.get("hit", False), tr.get("miss", False),
                          tr.get("false_alarm", False), tr.get("correct_reject", False)],
                         ["hit", "miss", "false_alarm", "correct_reject"], default="other")
    out["previous_outcome"] = np.concatenate((["none"], outcomes[:-1]))
    etc, epc = _timestamp_column(eye), _pupil_column(eye)
    rtc, rpc = _timestamp_column(running), _running_column(running)
    if etc and epc:
        out["pre_change_pupil"], out["pupil_missing_frac"] = _prechange_feature(
            eye[etc].to_numpy(float), eye[epc].to_numpy(float), align, normalize_median=True)
    else:
        out["pre_change_pupil"], out["pupil_missing_frac"] = np.nan, 1.0
    if rtc and rpc:
        out["pre_change_running"], out["running_missing_frac"] = _prechange_feature(
            running[rtc].to_numpy(float), running[rpc].to_numpy(float), align)
    else:
        out["pre_change_running"], out["running_missing_frac"] = np.nan, 1.0
    out["q2_covariates_complete"] = ((out.pupil_missing_frac <= .20) &
                                      (out.running_missing_frac <= .20))
    return out


def _write_h5(path: Path, exp, role: str, tr: pd.DataFrame,
              eye: pd.DataFrame, running: pd.DataFrame) -> dict:
    import h5py
    timestamps = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    cells = pd.DataFrame(exp.cell_specimen_table)
    if "valid_roi" in cells.columns:
        cells = cells[cells["valid_roi"].fillna(False).astype(bool)].copy()
    if "cell_specimen_id" in cells.columns:
        cells = cells.set_index("cell_specimen_id", drop=False)
    valid_ids = pd.Index(cells.index).astype(np.int64)
    if not len(valid_ids) or valid_ids.has_duplicates:
        raise ValueError("canonical valid-cell table is empty or has duplicate specimen IDs")
    dff_table, events_table = pd.DataFrame(exp.dff_traces), pd.DataFrame(exp.events)
    missing_dff = valid_ids.difference(dff_table.index)
    missing_events = valid_ids.difference(events_table.index)
    if len(missing_dff) or len(missing_events):
        raise ValueError(f"valid cells missing traces; dff={missing_dff.tolist()}, "
                         f"events={missing_events.tolist()}")
    dff = np.vstack(dff_table.loc[valid_ids, "dff"].to_numpy()).astype(np.float32)
    events = np.vstack(events_table.loc[valid_ids, "events"].to_numpy()).astype(np.float32)
    if dff.shape != events.shape or dff.shape[1] != len(timestamps):
        raise ValueError(f"trace/timestamp shape mismatch dff={dff.shape}, "
                         f"events={events.shape}, timestamps={len(timestamps)}")
    roi = (cells.loc[valid_ids, "cell_roi_id"].to_numpy(np.int64)
           if "cell_roi_id" in cells.columns else np.full(len(valid_ids), -1, np.int64))
    chunks = (1, min(4096, len(timestamps)))
    with h5py.File(path, "w") as h5:
        h5.attrs["role"] = role
        h5.create_dataset("ophys_timestamps", data=timestamps, compression="gzip", shuffle=True)
        h5.create_dataset("cell_specimen_id", data=valid_ids.to_numpy(np.int64))
        h5.create_dataset("cell_roi_id", data=roi)
        h5.create_dataset("events", data=events, chunks=chunks, compression="gzip",
                          compression_opts=4, shuffle=True)
        h5.create_dataset("dff", data=dff, chunks=chunks, compression="gzip",
                          compression_opts=4, shuffle=True)
        if role == "active":
            align = (tr["change_time"].to_numpy(float) if "change_time" in tr.columns
                     else np.full(len(tr), np.nan))
            dt = float(np.nanmedian(np.diff(timestamps)))
            rel = np.arange(WINDOW_START, WINDOW_END + dt/2, dt, dtype=np.float32)
            idx = np.searchsorted(timestamps, align[:, None] + rel[None, :])
            idx = np.clip(idx, 0, len(timestamps)-1)
            valid = (np.isfinite(align) &
                     ((align + WINDOW_START) >= timestamps[0]) &
                     ((align + WINDOW_END) <= timestamps[-1]))
            trial_ids = (tr["trials_id"].to_numpy(np.int64) if "trials_id" in tr.columns
                         else np.arange(len(tr), dtype=np.int64))
            group = h5.create_group("trial_locked")
            group.create_dataset("trial_id", data=trial_ids[valid])
            group.create_dataset("rel_time", data=rel)
            group.create_dataset("source_frame_index", data=idx[valid].astype(np.int32),
                                 compression="gzip")
            pupil = _trial_locked_scalar(eye, _pupil_column(eye), align, rel,
                                         normalize_median=True)
            speed = _trial_locked_scalar(running, _running_column(running), align, rel)
            group.create_dataset("pupil", data=pupil[valid], compression="gzip",
                                 compression_opts=4, shuffle=True)
            group.create_dataset("running", data=speed[valid], compression="gzip",
                                 compression_opts=4, shuffle=True)
            for name, matrix in (("events", events), ("dff", dff)):
                tensor = matrix[:, idx[valid]].transpose(1, 0, 2).astype(np.float32)
                base = tensor[:, :, rel < 0].mean(axis=2, keepdims=True)
                group.create_dataset(f"{name}_unbaselined", data=tensor,
                                     chunks=(1, min(32, tensor.shape[1]), len(rel)),
                                     compression="gzip", compression_opts=4, shuffle=True)
                group.create_dataset(f"{name}_baselined", data=tensor - base,
                                     chunks=(1, min(32, tensor.shape[1]), len(rel)),
                                     compression="gzip", compression_opts=4, shuffle=True)
    return {"n_cells": int(len(valid_ids)), "n_frames": int(len(timestamps)),
            "extra_dff_rois": int(len(dff_table.index.difference(valid_ids))),
            "extra_event_rois": int(len(events_table.index.difference(valid_ids)))}


def pull(manifest: pd.DataFrame, containers: list[int], out: Path, cache_dir: Path,
         retries: int = 3, report_name: str = "_pull.json") -> int:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".staging"; staging.mkdir(exist_ok=True)
    selected = manifest[manifest.ophys_container_id.astype(int).isin(containers)]
    ok, skipped, failed = [], [], []
    for row in selected.itertuples(index=False):
        oeid = int(row.ophys_experiment_id)
        if bundle_complete(out, oeid):
            skipped.append(oeid); continue
        for p in bundle_paths(out, oeid): p.unlink(missing_ok=True)
        last = None
        for attempt in range(1, retries + 1):
            stage = staging / str(oeid); shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
            cd = cache_dir / f"{oeid}-{attempt}"
            try:
                print(f"{oeid} ({row.role}) attempt {attempt}/{retries}", flush=True)
                cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cd)
                exp = cache.get_behavior_ophys_experiment(ophys_experiment_id=oeid)
                tr, e1 = _safe_table(lambda: exp.trials, "trials")
                sp, e2 = _safe_table(lambda: exp.stimulus_presentations, "stimulus_presentations")
                lk, e3 = _safe_table(lambda: exp.licks, "licks")
                rw, e4 = _safe_table(lambda: exp.rewards, "rewards")
                eye, eye_error = _safe_table(lambda: exp.eye_tracking, "eye_tracking")
                running, run_error = _safe_table(lambda: exp.running_speed, "running_speed")
                raw_running, raw_error = _safe_table(lambda: exp.raw_running_speed, "raw_running_speed")
                if any(x for x in (e1, e2, e3, e4)):
                    raise ValueError(f"required task table unavailable: {[x for x in (e1,e2,e3,e4) if x]}")
                q2 = _q2_features(tr, sp, lk, rw, eye, running)
                for suffix, table in (("trials.parquet", tr), ("stim.parquet", sp),
                                      ("licks.parquet", lk), ("rewards.parquet", rw),
                                      ("eye.parquet", eye), ("running.parquet", running),
                                      ("raw_running.parquet", raw_running), ("q2.parquet", q2)):
                    table.to_parquet(stage / f"{oeid}.{suffix}", index=False)
                qc = _write_h5(stage / f"{oeid}.neural.h5", exp, str(row.role), tr,
                               eye, running)
                meta = {c: (None if pd.isna(getattr(row, c)) else getattr(row, c))
                        for c in manifest.columns}
                meta.update(qc, eye_error=eye_error, running_error=run_error,
                            raw_running_error=raw_error,
                            filtered_events={"stored": False, "derivable": True,
                                             "filter": "causal_half_gaussian",
                                             "scale_seconds": 2.0 / 31.0,
                                             "n_time_steps": 20,
                                             "source": "AllenSDK Events defaults"})
                (stage / f"{oeid}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
                if not bundle_complete(stage, oeid): raise RuntimeError("staged bundle incomplete")
                for src, dst in zip(bundle_paths(stage, oeid), bundle_paths(out, oeid)):
                    src.replace(dst)
                ok.append({"ophys_experiment_id": oeid, "attempts": attempt, **qc})
                last = None; break
            except Exception:
                last = traceback.format_exc(limit=8)
                print(last, flush=True)
                if attempt < retries: time.sleep(5 * 2 ** (attempt-1))
            finally:
                shutil.rmtree(stage, ignore_errors=True); shutil.rmtree(cd, ignore_errors=True)
        if last: failed.append({"ophys_experiment_id": oeid, "error": last})
    shutil.rmtree(staging, ignore_errors=True)
    (out / report_name).write_text(json.dumps({"ok": ok, "skipped": skipped,
                                               "failed": failed}, indent=2))
    return 1 if failed else 0


def validate_bundles(root: Path, manifest: pd.DataFrame) -> None:
    expected = set(manifest.ophys_experiment_id.astype(int))
    incomplete = [x for x in sorted(expected) if not bundle_complete(root, x)]
    if incomplete: raise ValueError(f"incomplete neural bundles: {incomplete}")


def appendix_a_failures(root: Path, manifest: pd.DataFrame) -> list[dict]:
    """Return machine-readable schema failures without hiding a diagnostic run."""
    import h5py
    failures = []
    for row in manifest.itertuples(index=False):
        oeid, problems = int(row.ophys_experiment_id), []
        try:
            meta = json.loads((root / f"{oeid}.meta.json").read_text())
            for field in ("eye_error", "running_error", "raw_running_error"):
                if meta.get(field): problems.append(f"{field}:{meta[field]}")
            stim = pd.read_parquet(root / f"{oeid}.stim.parquet")
            eye = pd.read_parquet(root / f"{oeid}.eye.parquet")
            running = pd.read_parquet(root / f"{oeid}.running.parquet")
            raw_running = pd.read_parquet(root / f"{oeid}.raw_running.parquet")
            if stim.empty: problems.append("empty_stimulus_table")
            if not _timestamp_column(eye) or not _pupil_column(eye):
                problems.append("pupil_trace_schema")
            if not _timestamp_column(running) or not _running_column(running):
                problems.append("running_trace_schema")
            if not _timestamp_column(raw_running) or not _running_column(raw_running):
                problems.append("raw_running_trace_schema")
            with h5py.File(root / f"{oeid}.neural.h5", "r") as h5:
                required = {"ophys_timestamps", "cell_specimen_id", "cell_roi_id",
                            "events", "dff"}
                problems.extend(f"missing_h5:{name}" for name in sorted(required-set(h5.keys())))
                if required.issubset(h5.keys()):
                    shape = h5["events"].shape
                    if h5["dff"].shape != shape or shape != (
                            len(h5["cell_specimen_id"]), len(h5["ophys_timestamps"])):
                        problems.append("continuous_trace_shape")
                if str(row.role) == "active":
                    expected = {"trial_id", "rel_time", "source_frame_index",
                                "events_unbaselined", "events_baselined",
                                "dff_unbaselined", "dff_baselined", "pupil", "running"}
                    if "trial_locked" not in h5:
                        problems.append("missing_h5:trial_locked")
                    else:
                        problems.extend(f"missing_trial_locked:{name}" for name in
                                        sorted(expected-set(h5["trial_locked"].keys())))
        except Exception as exc:
            problems.append(f"validation_error:{type(exc).__name__}:{exc}")
        if problems:
            failures.append({"ophys_experiment_id": oeid, "role": str(row.role),
                             "problems": problems})
    return failures


def _folds(raw_index: np.ndarray, n_blocks=N_BLOCKS, gap=GAP_RAW_TRIALS):
    order = np.argsort(raw_index)
    for test in np.array_split(order, n_blocks):
        low, high = int(raw_index[test].min()), int(raw_index[test].max())
        train = np.setdiff1d(order, test)
        adjacent = ((raw_index[train] >= low-gap) & (raw_index[train] < low)) | (
                    (raw_index[train] > high) & (raw_index[train] <= high+gap))
        yield train[~adjacent], test


def _oof_auc(X: np.ndarray, y: np.ndarray, raw_index: np.ndarray, C: float,
             seed: int, *, blocked=True) -> tuple[float, np.ndarray, str | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    scores = np.full(len(y), np.nan)
    splits = (_folds(raw_index) if blocked else
              StratifiedKFold(N_BLOCKS, shuffle=True, random_state=seed).split(X, y))
    for train, test in splits:
        if len(np.unique(y[train])) < 2:
            return np.nan, scores, "temporal_support_nonestimable"
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(C=C, penalty="l2", class_weight="balanced",
                                   solver="liblinear", random_state=seed, max_iter=2000)
        model.fit(scaler.transform(X[train]), y[train])
        scores[test] = model.decision_function(scaler.transform(X[test]))
    if not np.isfinite(scores).all() or len(np.unique(y)) < 2:
        return np.nan, scores, "score_nonestimable"
    return float(roc_auc_score(y, scores)), scores, None


def _session_data(root: Path, oeid: int, labels: pd.DataFrame, signal="events",
                  *, baselined: bool = True, start: float = FIT_START,
                  end: float = FIT_END):
    import h5py
    with h5py.File(root / f"{oeid}.neural.h5", "r") as h5:
        tl = h5["trial_locked"]
        ids = tl["trial_id"][:]
        rel = tl["rel_time"][:]
        suffix = "baselined" if baselined else "unbaselined"
        tensor = tl[f"{signal}_{suffix}"][:]
        cells = h5["cell_specimen_id"][:]
    window = (rel >= start) & (rel < end)
    if not window.any():
        raise ValueError(f"empty feature window [{start}, {end}) for {oeid}")
    feature = tensor[:, :, window].mean(axis=2)
    selected = labels.set_index("trial_id").reindex(ids)
    return feature, cells, selected.reset_index()


def _subset_cells(X, k, seed, oeid):
    if k == "all": return X
    k = int(k)
    if X.shape[1] < k: return None
    digest = hashlib.sha256(f"{oeid}:{k}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return X[:, np.sort(rng.choice(X.shape[1], k, replace=False))]


def _evaluate_session(X, lab, state, keep, positive, negative, k, C, oeid,
                      blocked=True):
    mask = lab[state].fillna(False).astype(bool) & lab[keep].fillna(False).astype(bool)
    mask &= lab[positive].fillna(False).astype(bool) | lab[negative].fillna(False).astype(bool)
    y = lab.loc[mask, positive].astype(int).to_numpy()
    raw = lab.loc[mask, "trial_index"].astype(int).to_numpy()
    aucs, errors = [], []
    for seed in range(N_SEEDS):
        subset = _subset_cells(X[mask.to_numpy()], k, seed, oeid)
        if subset is None: return np.nan, "low_cells"
        auc, _, err = _oof_auc(subset, y, raw, C, seed, blocked=blocked)
        if err: errors.append(err)
        else: aucs.append(auc)
    return (float(np.mean(aucs)), None) if aucs else (np.nan, ";".join(sorted(set(errors))))


def _mouse_summary(rows: pd.DataFrame, value="auc", weight="miss_B") -> tuple[pd.DataFrame, float, float]:
    valid = rows[np.isfinite(rows[value])].copy()
    mouse_rows = []
    for mouse, group in valid.groupby("mouse_id"):
        weights = np.maximum(group[weight].to_numpy(float), 1)
        mouse_rows.append({"mouse_id": mouse, value: float(np.average(group[value], weights=weights))})
    mice = pd.DataFrame(mouse_rows)
    return mice, float(mice[value].mean()) if len(mice) else np.nan, (
        float(mice[value].std(ddof=1) / np.sqrt(len(mice))) if len(mice) > 1 else np.nan)


def _state_oof_metrics(X: np.ndarray, y: np.ndarray, raw: np.ndarray,
                       C: float, seed: int) -> tuple[float, float, str | None]:
    """Return held-out state AUC and natural-probability log-loss gain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scores = np.full(len(y), np.nan)
    probs = np.full(len(y), np.nan)
    null_probs = np.full(len(y), np.nan)
    for train, test in _folds(raw):
        if len(np.unique(y[train])) < 2:
            return np.nan, np.nan, "temporal_support_nonestimable"
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(C=C, penalty="l2", class_weight=None,
                                   solver="liblinear", random_state=seed, max_iter=2000)
        model.fit(scaler.transform(X[train]), y[train])
        transformed = scaler.transform(X[test])
        scores[test] = model.decision_function(transformed)
        probs[test] = model.predict_proba(transformed)[:, 1]
        prevalence = float(np.clip(y[train].mean(), 1e-6, 1 - 1e-6))
        null_probs[test] = prevalence
    if not (np.isfinite(scores).all() and np.isfinite(probs).all() and
            np.isfinite(null_probs).all()):
        return np.nan, np.nan, "state_score_incomplete"
    return (float(roc_auc_score(y, scores)),
            float(log_loss(y, null_probs, labels=[0, 1]) -
                  log_loss(y, probs, labels=[0, 1])), None)


def _state_anchor(session_cache, C50):
    """Run the one authoritative v3.2 anchor plus prespecified diagnostics."""
    rows = []
    specifications = (
        ("unbaselined_pre", "anchor_pre", True, True),
        ("unbaselined_pre", "anchor_pre", False, True),
        ("baselined_pre", "baseline_pre", True, False),
        ("baselined_post", "X", True, False),
        ("unbaselined_post", "unbaselined_post", True, False),
    )
    for item in session_cache:
        lab, meta = item["labels"], item["meta"]
        for representation, feature_key, guarded, authoritative in specifications:
            X, aucs, gains = item[feature_key], [], []
            eligible = (lab.late_hit | lab.miss) & (~lab.first_ten)
            if guarded:
                eligible &= lab.keep_B
            candidates = lab.loc[eligible].copy()
            group_counts = {
                (outcome, state): int((candidates[outcome] &
                                       (candidates.engaged_B == state)).sum())
                for outcome in ("late_hit", "miss") for state in (False, True)
            }
            balanced_per_state = sum(min(group_counts[(outcome, False)],
                                         group_counts[(outcome, True)])
                                     for outcome in ("late_hit", "miss"))
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + int(meta["ophys_experiment_id"]))
                chosen = []
                for outcome in ("late_hit", "miss"):
                    groups = [candidates[candidates[outcome] &
                                         (candidates.engaged_B == state)].index.to_numpy()
                              for state in (False, True)]
                    n = min(map(len, groups))
                    if n:
                        chosen.extend(rng.choice(group, n, replace=False).tolist()
                                      for group in groups)
                flat = np.array([i for group in chosen for i in group], dtype=int)
                if not len(flat):
                    continue
                y = lab.loc[flat, "engaged_B"].astype(int).to_numpy()
                raw = lab.loc[flat, "trial_index"].astype(int).to_numpy()
                subset = _subset_cells(X[flat], PRIMARY_K, seed,
                                       meta["ophys_experiment_id"])
                if subset is None:
                    continue
                auc, gain, err = _state_oof_metrics(subset, y, raw, C50, seed)
                if not err:
                    aucs.append(auc); gains.append(gain)
            rows.append({**meta, "representation": representation,
                         "guarded": guarded, "authoritative": authoritative,
                         "auc_state": float(np.mean(aucs)) if aucs else np.nan,
                         "state_logloss_gain": (float(np.mean(gains)) if gains else np.nan),
                         "limiting_state_n": int(balanced_per_state),
                         "n_state_disengaged": int(balanced_per_state),
                         "n_state_engaged": int(balanced_per_state)})
    df = pd.DataFrame(rows)
    summaries = {}
    for name, selected in (
        ("authoritative_guarded", df[df.authoritative & df.guarded]),
        ("unguarded_diagnostic", df[df.authoritative & ~df.guarded]),
    ):
        auc_mice, auc_mean, _ = _mouse_summary(
            selected.rename(columns={"auc_state": "auc"}), weight="limiting_state_n")
        gain_mice, gain_mean, _ = _mouse_summary(
            selected.rename(columns={"state_logloss_gain": "gain"}),
            value="gain", weight="limiting_state_n")
        summaries[name] = {
            "auc": auc_mean, "state_logloss_gain": gain_mean,
            "n_mice": int(len(auc_mice)),
            "auc_mouse_bca": _bca_mean(auc_mice.auc.to_numpy()
                                         if len(auc_mice) else np.array([]), seed=3),
            "logloss_gain_mouse_bca": _bca_mean(gain_mice.gain.to_numpy()
                                                  if len(gain_mice) else np.array([]), seed=4),
            "session_weight": "limiting outcome-balanced state class"}
    diagnostics = {}
    for (representation, guarded), group in df.groupby(["representation", "guarded"]):
        mice, mean, _ = _mouse_summary(
            group.rename(columns={"auc_state": "auc"}), weight="limiting_state_n")
        diagnostics[f"{representation}_{'guarded' if guarded else 'unguarded'}"] = {
            "auc": mean, "n_mice": int(len(mice)),
            "authoritative": bool(group.authoritative.all() and guarded)}
    summaries["representation_diagnostics"] = diagnostics
    return df, summaries


def _calibration_summary(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Descriptive calibration-in-the-large and slope on pooled OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(probabilities, float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, penalty="l2", class_weight=None,
                                   solver="liblinear", max_iter=2000).fit(logits, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def _q2_session(root: Path, item: dict, C50: float) -> tuple[dict, str | None]:
    """Strict outer/inner cross-fit under the observed outcome prevalence."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    oeid = int(item["meta"]["ophys_experiment_id"])
    q2 = pd.read_parquet(root / f"{oeid}.q2.parquet").set_index("trial_id")
    lab = item["labels"].copy()
    joined = lab.join(q2.drop(columns=["change_time"], errors="ignore"), on="trial_id")
    mask = (joined.engaged_B.astype(bool) & joined.keep_B.astype(bool) &
            (joined.late_hit.astype(bool) | joined.miss.astype(bool)) &
            joined.q2_covariates_complete.fillna(False).astype(bool))
    use = joined.loc[mask].copy()
    if len(use) < 2 or use.late_hit.nunique() < 2:
        return {}, "q2_class_nonestimable"
    Xn_all = item["X"][mask.to_numpy()]
    y = use.late_hit.astype(int).to_numpy()
    raw = use.trial_index.astype(int).to_numpy()
    continuous = ["flashes_before_change", "time_since_previous_change",
                  "time_since_previous_lick", "time_since_previous_reward",
                  "session_position", "pre_change_pupil", "pre_change_running"]
    categorical = ["transition", "preceding_omission", "previous_outcome"]

    def nuisance_model(include_neural=False):
        cont = continuous + (["neural_score"] if include_neural else [])
        pre = ColumnTransformer([
            ("continuous", Pipeline([("impute", SimpleImputer(strategy="median",
                                                               add_indicator=True)),
                                     ("scale", StandardScaler())]), cont),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             categorical),
        ])
        return Pipeline([("pre", pre),
                         ("model", LogisticRegression(C=1.0, penalty="l2",
                                                      class_weight=None,
                                                      solver="liblinear", max_iter=2000))])

    seed_rows = []
    for seed in range(N_SEEDS):
        Xn = _subset_cells(Xn_all, PRIMARY_K, seed, oeid)
        if Xn is None: return {}, "low_cells"
        pred0, pred1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
        pred_neural, q1_score = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for outer_train, outer_test in _folds(raw):
            if len(np.unique(y[outer_train])) < 2:
                return {}, "q2_temporal_support_nonestimable"
            # Inner OOF scores for M1 training; the outer test block never enters.
            inner_scores = np.full(len(outer_train), np.nan)
            inner_raw = raw[outer_train]
            for inner_train_local, inner_test_local in _folds(inner_raw, n_blocks=4):
                inner_train = outer_train[inner_train_local]
                inner_test = outer_train[inner_test_local]
                if len(np.unique(y[inner_train])) < 2:
                    return {}, "q2_inner_temporal_support_nonestimable"
                scaler = StandardScaler().fit(Xn[inner_train])
                neural = LogisticRegression(C=C50, class_weight=None, solver="liblinear",
                                            random_state=seed, max_iter=2000)
                neural.fit(scaler.transform(Xn[inner_train]), y[inner_train])
                inner_scores[inner_test_local] = neural.decision_function(
                    scaler.transform(Xn[inner_test]))
            if not np.isfinite(inner_scores).all():
                return {}, "q2_inner_score_incomplete"
            scaler = StandardScaler().fit(Xn[outer_train])
            neural = LogisticRegression(C=C50, class_weight=None, solver="liblinear",
                                        random_state=seed, max_iter=2000)
            neural.fit(scaler.transform(Xn[outer_train]), y[outer_train])
            transformed_test = scaler.transform(Xn[outer_test])
            test_score = neural.decision_function(transformed_test)
            pred_neural[outer_test] = neural.predict_proba(transformed_test)[:, 1]
            q1 = LogisticRegression(C=C50, class_weight="balanced", solver="liblinear",
                                    random_state=seed, max_iter=2000)
            q1.fit(scaler.transform(Xn[outer_train]), y[outer_train])
            q1_score[outer_test] = q1.decision_function(transformed_test)
            center, scale = float(inner_scores.mean()), float(inner_scores.std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                return {}, "q2_neural_score_zero_variance"
            train_frame = use.iloc[outer_train][continuous + categorical].copy()
            test_frame = use.iloc[outer_test][continuous + categorical].copy()
            train_frame["neural_score"] = (inner_scores - center) / scale
            test_frame["neural_score"] = (test_score - center) / scale
            m0, m1 = nuisance_model(False), nuisance_model(True)
            m0.fit(train_frame, y[outer_train]); m1.fit(train_frame, y[outer_train])
            pred0[outer_test] = m0.predict_proba(test_frame)[:, 1]
            pred1[outer_test] = m1.predict_proba(test_frame)[:, 1]
        if not np.isfinite(pred0).all() or not np.isfinite(pred1).all():
            return {}, "q2_outer_score_incomplete"
        if not np.isfinite(pred_neural).all() or not np.isfinite(q1_score).all():
            return {}, "q2_neural_probability_incomplete"
        m0_loss = float(log_loss(y, pred0, labels=[0, 1]))
        m1_loss = float(log_loss(y, pred1, labels=[0, 1]))
        m0_auc, m1_auc = float(roc_auc_score(y, pred0)), float(roc_auc_score(y, pred1))
        m0_ci, m0_slope = _calibration_summary(y, pred0)
        m1_ci, m1_slope = _calibration_summary(y, pred1)
        seed_rows.append({
            "n_trials": int(len(y)), "prevalence": float(y.mean()),
            "m0_log_loss": m0_loss, "m1_log_loss": m1_loss,
            "delta_log_loss": m0_loss - m1_loss,
            "m0_auc": m0_auc, "m1_auc": m1_auc, "delta_auc": m1_auc - m0_auc,
            "m0_brier": float(brier_score_loss(y, pred0)),
            "m1_brier": float(brier_score_loss(y, pred1)),
            "neural_only_auc": float(roc_auc_score(y, pred_neural)),
            "neural_only_log_loss": float(log_loss(y, pred_neural, labels=[0, 1])),
            "q1_auc_same_trials": float(roc_auc_score(y, q1_score)),
            "m0_calibration_intercept": m0_ci, "m0_calibration_slope": m0_slope,
            "m1_calibration_intercept": m1_ci, "m1_calibration_slope": m1_slope,
        })
    frame = pd.DataFrame(seed_rows)
    metrics = {column: (int(frame[column].iloc[0]) if column == "n_trials" else
                        float(frame[column].mean())) for column in frame.columns}
    return metrics, None


def _baseline_integrity(item: dict, C50: float) -> dict:
    """Verify that trial-wise baseline subtraction removed pre-change level."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    lab, oeid = item["labels"], int(item["meta"]["ophys_experiment_id"])
    mask = (lab.engaged_B.astype(bool) & lab.keep_B.astype(bool) &
            (lab.late_hit.astype(bool) | lab.miss.astype(bool)))
    y = lab.loc[mask, "late_hit"].astype(int).to_numpy()
    raw = lab.loc[mask, "trial_index"].astype(int).to_numpy()
    post, pre = item["X"][mask.to_numpy()], item["baseline_pre"][mask.to_numpy()]
    maximum_feature_abs = float(np.max(np.abs(pre))) if pre.size else np.nan
    seed_ranges, raw_aucs = [], []
    for seed in range(N_SEEDS):
        post_k = _subset_cells(post, PRIMARY_K, seed, oeid)
        pre_k = _subset_cells(pre, PRIMARY_K, seed, oeid)
        if post_k is None or pre_k is None:
            return {"passed": False, "error": "low_cells"}
        score = np.full(len(y), np.nan)
        fold_ranges = []
        for train, test in _folds(raw):
            if len(np.unique(y[train])) < 2:
                return {"passed": False, "error": "temporal_support_nonestimable"}
            scaler = StandardScaler().fit(post_k[train])
            model = LogisticRegression(C=C50, class_weight="balanced", solver="liblinear",
                                       random_state=seed, max_iter=2000)
            model.fit(scaler.transform(post_k[train]), y[train])
            score[test] = model.decision_function(scaler.transform(pre_k[test]))
            # Cross-validation fits a different intercept in every fold. The
            # constructional invariant is therefore within-fold constancy, not
            # equality of pooled scores produced by five different models.
            fold_ranges.append(float(np.ptp(score[test])))
        seed_ranges.append(float(max(fold_ranges)))
        raw_aucs.append(float(roc_auc_score(y, score)))
    max_score_range = float(max(seed_ranges))
    numerically_constant = bool(maximum_feature_abs <= 1e-5 and max_score_range <= 1e-5)
    return {"passed": numerically_constant,
            "max_abs_prechange_baselined_feature": maximum_feature_abs,
            "max_prechange_mean_dv_range": max_score_range,
            "raw_pooled_cross_fold_auc_mean": float(np.mean(raw_aucs)),
            "constant_score_auc": 0.5 if numerically_constant else None,
            "feature_tolerance": 1e-5, "score_range_tolerance": 1e-5}


def _auc_time_session(root: Path, item: dict, C50: float) -> tuple[pd.DataFrame, str | None]:
    """Project the one cross-fitted 0-.3 s axis at every saved timepoint."""
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    oeid = int(item["meta"]["ophys_experiment_id"])
    with h5py.File(root / f"{oeid}.neural.h5", "r") as h5:
        tensor = h5["trial_locked/events_baselined"][:]
        rel = h5["trial_locked/rel_time"][:]
    lab = item["labels"]
    mask = (lab.engaged_B.astype(bool) & lab.keep_B.astype(bool) &
            (lab.late_hit.astype(bool) | lab.miss.astype(bool)))
    y = lab.loc[mask, "late_hit"].astype(int).to_numpy()
    raw = lab.loc[mask, "trial_index"].astype(int).to_numpy()
    tensor = tensor[mask.to_numpy()]
    curves = []
    for seed in range(N_SEEDS):
        digest = hashlib.sha256(f"{oeid}:{PRIMARY_K}:{seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        if tensor.shape[1] < PRIMARY_K: return pd.DataFrame(), "low_cells"
        cells = np.sort(rng.choice(tensor.shape[1], PRIMARY_K, replace=False))
        fit_X = tensor[:, cells][:, :, (rel >= FIT_START) & (rel < FIT_END)].mean(axis=2)
        score_t = np.full((len(y), len(rel)), np.nan)
        for train, test in _folds(raw):
            if len(np.unique(y[train])) < 2:
                return pd.DataFrame(), "temporal_support_nonestimable"
            scaler = StandardScaler().fit(fit_X[train])
            model = LogisticRegression(C=C50, class_weight="balanced", solver="liblinear",
                                       random_state=seed, max_iter=2000)
            model.fit(scaler.transform(fit_X[train]), y[train])
            # Apply the fixed coefficients to each instantaneous population vector.
            for ti in range(len(rel)):
                score_t[test, ti] = model.decision_function(
                    scaler.transform(tensor[test][:, cells, ti]))
        curves.append([roc_auc_score(y, score_t[:, ti]) for ti in range(len(rel))])
    return pd.DataFrame({"ophys_experiment_id": oeid, "rel_time": rel,
                         "auc": np.mean(curves, axis=0)}), None


def _bca_mean(values: np.ndarray, seed=0) -> dict:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if len(values) < 2: return {"mean": float(values.mean()) if len(values) else None,
                               "low": None, "high": None, "n_mice": int(len(values))}
    from scipy.stats import bootstrap
    result = bootstrap((values,), np.mean, method="BCa", confidence_level=.95,
                       n_resamples=2000, random_state=np.random.default_rng(seed))
    return {"mean": float(values.mean()), "low": float(result.confidence_interval.low),
            "high": float(result.confidence_interval.high), "n_mice": int(len(values))}


def _precision_gates(*, appendix_complete: bool, q1_mice: int, anchor_mice: int,
                     q2_mice: int, q1_sd: float, q2_sd: float,
                     q1_margin: float, q2_sesoi: float) -> dict:
    try:
        from scipy.stats import t
        multiplier = float(t.ppf(.975, df=CONFIRM_MICE - 1))
    except Exception:
        multiplier = 2.048
    q1_half_width = (float(multiplier * q1_sd / np.sqrt(CONFIRM_MICE))
                     if np.isfinite(q1_sd) else np.nan)
    q2_half_width = (float(multiplier * q2_sd / np.sqrt(CONFIRM_MICE))
                     if np.isfinite(q2_sd) else np.nan)
    coverage = bool(appendix_complete and q1_mice >= 8 and anchor_mice >= 8 and
                    q2_mice >= 8)
    q1_precision = bool(coverage and np.isfinite(q1_margin) and q1_margin > 0 and
                        np.isfinite(q1_half_width) and q1_half_width < q1_margin)
    q2_precision = bool(coverage and np.isfinite(q2_sesoi) and q2_sesoi > 0 and
                        np.isfinite(q2_half_width) and q2_half_width < q2_sesoi)
    return {"coverage": coverage, "q1_precision": q1_precision,
            "q2_precision": q2_precision,
            "confirm_ready": bool(coverage and q1_precision and q2_precision),
            "q1_projected_half_width_29": q1_half_width,
            "q2_projected_half_width_29": q2_half_width}


def _weighted_random_intercept(df: pd.DataFrame, *, include_novel: bool) -> dict:
    """Weighted Gaussian random-intercept fit with sum-to-zero binary contrasts."""
    from scipy.optimize import minimize
    use = df[np.isfinite(df.auc)].copy()
    if include_novel: use = use[use.novel.notna()].copy()
    if len(use) < 3 or use.mouse_id.nunique() < 2:
        return {"error": "insufficient data"}
    if include_novel and use.novel.astype(bool).nunique() < 2:
        return {"error": "nonestimable_no_novelty_variation",
                "n_sessions": int(len(use)), "n_mice": int(use.mouse_id.nunique()),
                "observed_novel_levels": sorted(
                    map(bool, use.novel.astype(bool).unique().tolist()))}
    project_levels = sorted(use.project_code.astype(str).unique())
    if len(project_levels) != 2: return {"error": "project_code is not binary"}
    columns = [np.ones(len(use)),
               np.where(use.project_code.astype(str).eq(project_levels[1]), .5, -.5)]
    names = ["marginal_mean", f"project:{project_levels[1]}-{project_levels[0]}"]
    if include_novel:
        columns.append(np.where(use.novel.astype(bool), .5, -.5)); names.append("novel:true-false")
    X, y = np.column_stack(columns), use.auc.to_numpy(float)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        return {"error": "nonestimable_rank_deficient_fixed_effects",
                "rank": rank, "n_fixed_effects": int(X.shape[1]),
                "fixed_effects": names, "n_sessions": int(len(use)),
                "n_mice": int(use.mouse_id.nunique())}
    weights = np.maximum(use.miss_B.to_numpy(float), 1.0)
    groups = use.mouse_id.to_numpy()

    def solve(theta):
        sigma2, tau2 = np.exp(theta)
        precision = np.zeros((X.shape[1], X.shape[1])); rhs = np.zeros(X.shape[1])
        pieces, logdet = [], 0.0
        for mouse in np.unique(groups):
            idx = np.flatnonzero(groups == mouse)
            V = np.diag(sigma2 / weights[idx]) + tau2 * np.ones((len(idx), len(idx)))
            inv = np.linalg.inv(V)
            precision += X[idx].T @ inv @ X[idx]; rhs += X[idx].T @ inv @ y[idx]
            logdet += np.linalg.slogdet(V)[1]; pieces.append((idx, inv))
        beta = np.linalg.solve(precision, rhs)
        quad = sum((y[idx]-X[idx]@beta).T @ inv @ (y[idx]-X[idx]@beta)
                   for idx, inv in pieces)
        return .5 * (logdet + quad), beta, np.linalg.inv(precision)
    fit = minimize(lambda z: solve(z)[0], np.log([.01, .01]), method="Nelder-Mead")
    if not fit.success or not np.isfinite(fit.fun):
        return {"error": "variance_optimizer_failed", "message": str(fit.message),
                "n_sessions": int(len(use)), "n_mice": int(use.mouse_id.nunique())}
    _, beta, cov = solve(fit.x)
    return {"converged": bool(fit.success), "params": dict(zip(names, map(float, beta))),
            "standard_errors": dict(zip(names, map(float, np.sqrt(np.diag(cov))))),
            "residual_variance": float(np.exp(fit.x[0])),
            "between_mouse_variance": float(np.exp(fit.x[1])), "n_sessions": int(len(use)),
            "n_mice": int(use.mouse_id.nunique()), "weights": "n_engaged_miss"}


def _safe_secondary_model(df: pd.DataFrame, *, include_novel: bool) -> dict:
    """A secondary diagnostic must never prevent primary results from publishing."""
    try:
        return _weighted_random_intercept(df, include_novel=include_novel)
    except Exception as exc:
        numerical = isinstance(exc, (np.linalg.LinAlgError, FloatingPointError, ValueError))
        return {"error": ("secondary_model_numerical_failure" if numerical else
                          "secondary_model_unexpected_failure"),
                "exception": type(exc).__name__, "message": str(exc),
                "n_sessions_input": int(len(df))}


def scan(root: Path, experiment_manifest: Path, behavior_dir: Path, out: Path,
         *, data_release: str | None = None,
         data_manifest_sha256: str | None = None) -> int:
    manifest = pd.read_csv(experiment_manifest)
    validate_bundles(root, manifest)
    appendix_failures = appendix_a_failures(root, manifest)
    active = manifest[manifest.role.eq("active")].copy()
    labels = pd.read_parquet(behavior_dir / "_trial_labels.parquet")
    sessions = pd.read_parquet(behavior_dir / "_session_scan.parquet")
    expected_behavior = set(active.behavior_session_id.astype(int))
    label_behavior = set(labels.behavior_session_id.astype(int))
    session_behavior = set(sessions.behavior_session_id.astype(int))
    if label_behavior != expected_behavior or session_behavior != expected_behavior:
        raise ValueError(
            "behavior/neural active session mismatch; "
            f"labels_missing={sorted(expected_behavior-label_behavior)}, "
            f"labels_extra={sorted(label_behavior-expected_behavior)}, "
            f"sessions_missing={sorted(expected_behavior-session_behavior)}, "
            f"sessions_extra={sorted(session_behavior-expected_behavior)}")
    cache = []
    for row in active.itertuples(index=False):
        bsid, oeid = int(row.behavior_session_id), int(row.ophys_experiment_id)
        session_labels = labels[labels.behavior_session_id.eq(bsid)]
        X, _, lab = _session_data(root, oeid, session_labels)
        baseline_pre, _, _ = _session_data(
            root, oeid, session_labels, baselined=True, start=-1.0, end=0.0)
        anchor_pre, _, _ = _session_data(
            root, oeid, session_labels, baselined=False, start=-1.0, end=0.0)
        unbaselined_post, _, _ = _session_data(
            root, oeid, session_labels, baselined=False, start=FIT_START, end=FIT_END)
        ses = sessions.loc[sessions.behavior_session_id.eq(bsid)].iloc[0]
        novelty = lab.is_image_novel.dropna().astype(bool).unique()
        cache.append({"X": X, "baseline_pre": baseline_pre,
                      "anchor_pre": anchor_pre,
                      "unbaselined_post": unbaselined_post,
                      "labels": lab, "meta": {
            "ophys_experiment_id": oeid, "behavior_session_id": bsid,
            "mouse_id": int(row.mouse_id), "project_code": row.project_code,
            "novel": (bool(novelty[0]) if len(novelty) == 1 else None),
            "miss_B": int(ses.miss_B), "late_hit_B": int(ses.late_hit_B),
            "miss_A": int(ses.miss_A),
            "behavioral_eligible": bool(ses.behavioral_eligible)}})

    out.mkdir(parents=True, exist_ok=True)
    primary_rows = []
    for item in cache:
        meta = item["meta"]
        if not meta["behavioral_eligible"]:
            continue
        auc, err = _evaluate_session(item["X"], item["labels"], "engaged_B", "keep_B",
                                     "late_hit", "miss", PRIMARY_K, FROZEN_C50,
                                     meta["ophys_experiment_id"])
        primary_rows.append({**meta, "K": PRIMARY_K, "C": FROZEN_C50, "auc": auc,
                             "decoder_estimability": err or "estimable"})

    integrity_rows = []
    for item in cache:
        if item["meta"]["behavioral_eligible"]:
            integrity_rows.append({**item["meta"],
                                   **_baseline_integrity(item, FROZEN_C50)})
    integrity_df = pd.DataFrame(integrity_rows)
    integrity_df.to_parquet(out / "baseline_integrity.parquet", index=False)
    if len(integrity_df) and not bool(integrity_df.passed.all()):
        failed = integrity_df.loc[~integrity_df.passed, "ophys_experiment_id"].tolist()
        raise ValueError(f"baseline-subtraction integrity failed for {failed}")

    anchor_df, anchor = _state_anchor(cache, FROZEN_C50)
    guarded = anchor.get("authoritative_guarded", {})
    auc_state = float(guarded.get("auc", np.nan))
    state_logloss_gain = float(guarded.get("state_logloss_gain", np.nan))
    q1_sesoi_margin = .2 * (auc_state - .5) if np.isfinite(auc_state) else np.nan
    q2_sesoi = .2 * state_logloss_gain if np.isfinite(state_logloss_gain) else np.nan
    primary_df = pd.DataFrame(primary_rows)
    mouse_q1, _, _ = _mouse_summary(primary_df)

    # The registered miss-threshold sweep changes only the session-selection
    # rule.  C_50, cells, folds, labels and the late-hit>=20 rule stay frozen.
    # Fit every potentially selected session once, then re-aggregate the exact
    # primary estimator at each threshold.
    threshold_base = []
    for item in cache:
        meta = item["meta"]
        if meta["late_hit_B"] < 20 or meta["miss_B"] < 10:
            continue
        auc, err = _evaluate_session(
            item["X"], item["labels"], "engaged_B", "keep_B", "late_hit", "miss",
            PRIMARY_K, FROZEN_C50, meta["ophys_experiment_id"])
        threshold_base.append({**meta, "auc": auc,
                               "decoder_estimability": err or "estimable"})
    threshold_base = pd.DataFrame(threshold_base)
    threshold_session_rows, threshold_summary = [], []
    for threshold in (10, 15, 20, 25, 30):
        selected = (threshold_base[threshold_base.miss_B.ge(threshold)].copy()
                    if len(threshold_base) else threshold_base.copy())
        selected["miss_threshold"] = threshold
        threshold_session_rows.append(selected)
        mice, mean, _ = _mouse_summary(selected)
        interval = _bca_mean(mice.auc.to_numpy() if len(mice) else np.array([]),
                             seed=100 + threshold)
        threshold_summary.append({"miss_threshold": threshold,
                                  "n_behavioral_sessions": int(len(selected)),
                                  "n_estimable_sessions": int(np.isfinite(selected.auc).sum()),
                                  "n_mice": int(len(mice)), "mouse_mean_auc": mean,
                                  "ci_low": interval["low"], "ci_high": interval["high"]})
    sensitivity_rows, q2_rows, time_rows = [], [], []
    for item in cache:
        meta = item["meta"]
        if not meta["behavioral_eligible"]: continue
        random_auc, random_err = _evaluate_session(
            item["X"], item["labels"], "engaged_B", "keep_B", "late_hit", "miss",
            PRIMARY_K, FROZEN_C50, meta["ophys_experiment_id"], blocked=False)
        dff_X, _, dff_lab = _session_data(
            root, meta["ophys_experiment_id"],
            labels[labels.behavior_session_id.eq(meta["behavior_session_id"])], signal="dff")
        dff_auc, dff_err = _evaluate_session(
            dff_X, dff_lab, "engaged_B", "keep_B", "late_hit", "miss", PRIMARY_K,
            FROZEN_C50, meta["ophys_experiment_id"], blocked=True)
        sensitivity_rows.extend([
            {**meta, "analysis": "events_random_cv", "auc": random_auc,
             "decoder_estimability": random_err or "estimable"},
            {**meta, "analysis": "dff_blocked_cv", "auc": dff_auc,
             "decoder_estimability": dff_err or "estimable"},
        ])
        q2_metrics, q2err = _q2_session(root, item, FROZEN_C50)
        q2_rows.append({**meta, **q2_metrics,
                        "q2_estimability": q2err or "estimable"})
        time_df, time_err = _auc_time_session(root, item, FROZEN_C50)
        if time_err:
            sensitivity_rows.append({**meta, "analysis": "fixed_axis_auc_time",
                                     "auc": np.nan, "decoder_estimability": time_err})
        else:
            time_df["behavior_session_id"] = meta["behavior_session_id"]
            time_df["mouse_id"] = meta["mouse_id"]
            time_rows.append(time_df)

    q2_df = pd.DataFrame(q2_rows)
    q2_mouse_rows = []
    q2_valid = q2_df[q2_df.get("delta_log_loss", pd.Series(index=q2_df.index,
                                                            dtype=float)).notna()]
    q2_metric_columns = [column for column in q2_df.columns
                         if column in {"prevalence", "m0_log_loss", "m1_log_loss",
                                       "delta_log_loss", "m0_auc", "m1_auc", "delta_auc",
                                       "m0_brier", "m1_brier", "neural_only_auc",
                                       "neural_only_log_loss", "q1_auc_same_trials",
                                       "m0_calibration_intercept", "m0_calibration_slope",
                                       "m1_calibration_intercept", "m1_calibration_slope"}]
    for mouse, group in q2_valid.groupby("mouse_id"):
        weights = np.maximum(group.miss_B.to_numpy(float), 1)
        row = {"mouse_id": mouse}
        for column in q2_metric_columns:
            finite = np.isfinite(group[column].to_numpy(float))
            row[column] = (float(np.average(group.loc[finite, column], weights=weights[finite]))
                           if finite.any() else np.nan)
        q2_mouse_rows.append(row)
    q2_mice = pd.DataFrame(q2_mouse_rows)

    q1_sd = float(mouse_q1.auc.std(ddof=1)) if len(mouse_q1) > 1 else np.nan
    q2_sd = (float(q2_mice["delta_log_loss"].std(ddof=1))
             if len(q2_mice) > 1 and "delta_log_loss" in q2_mice else np.nan)
    gates = _precision_gates(
        appendix_complete=not appendix_failures, q1_mice=len(mouse_q1),
        anchor_mice=int(guarded.get("n_mice", 0)), q2_mice=len(q2_mice),
        q1_sd=q1_sd, q2_sd=q2_sd, q1_margin=q1_sesoi_margin, q2_sesoi=q2_sesoi)

    primary_df.to_parquet(out / "q1_sessions.parquet", index=False)
    anchor_df.to_parquet(out / "auc_state.parquet", index=False)
    mouse_q1.to_parquet(out / "q1_mice.parquet", index=False)
    pd.DataFrame(sensitivity_rows).to_parquet(out / "q1_sensitivity.parquet", index=False)
    (pd.concat(threshold_session_rows, ignore_index=True) if threshold_session_rows else
     pd.DataFrame()).to_parquet(out / "threshold_sweep_sessions.parquet", index=False)
    pd.DataFrame(threshold_summary).to_parquet(out / "threshold_sweep.parquet", index=False)
    (pd.concat(time_rows, ignore_index=True) if time_rows else
     pd.DataFrame(columns=["ophys_experiment_id", "behavior_session_id", "mouse_id",
                           "rel_time", "auc"])).to_parquet(out / "auc_time.parquet", index=False)
    q2_df.to_parquet(out / "q2_sessions.parquet", index=False)
    q2_mice.to_parquet(out / "q2_mice.parquet", index=False)
    result = {
        "schema": "neural-dev-v3.2", "n_expected_experiments": 70,
        "n_active": 50, "n_passive": 20,
        "data_source": {"analysis_only": True, "reused_extracted_bundles": True,
                        "allen_nwb_download": False,
                        "neural_data_release": data_release,
                        "data_manifest_sha256": data_manifest_sha256},
        "appendix_a": {"complete": not appendix_failures,
                       "failures": appendix_failures},
        "primary": {"K": PRIMARY_K, "C": FROZEN_C50,
                    "signal": "events", "target": "late-hit-vs-miss"},
        "k_policy": {"authoritative_K": 50, "other_K_computed": False,
                     "reason": "cell-count heterogeneity"},
        "c_policy": {"C50": FROZEN_C50, "source": "frozen v3.1 one-SE result",
                     "retuned": False},
        "v3_1_comparator": {"recomputed": False, "status": "immutable external reference"},
        "selection_sweep": {"miss_thresholds": [10, 15, 20, 25, 30],
                            "late_hit_min": 20,
                            "C_and_decoder_frozen": True},
        "anchor": {"authoritative": "unbaselined_pre_guarded",
                   "authoritative_guarded": guarded,
                   "unguarded_diagnostic": anchor.get("unguarded_diagnostic"),
                   "representation_diagnostics": anchor.get("representation_diagnostics"),
                   "other_representations_diagnostic_only": True},
        "sesoi": {"q1_auc_boundary": (.5 + q1_sesoi_margin
                                          if np.isfinite(q1_sesoi_margin) else None),
                  "q1_margin": (q1_sesoi_margin if np.isfinite(q1_sesoi_margin) else None),
                  "q2_delta_logloss": (q2_sesoi if np.isfinite(q2_sesoi) else None)},
        "gates": {**gates,
                  "q1_between_mouse_sd": q1_sd,
                  "q2_between_mouse_sd": q2_sd,
                  "required_mice": 8},
        "q1_mouse_bca": _bca_mean(mouse_q1.auc.to_numpy() if len(mouse_q1) else np.array([])),
        "q2": {"delta_log_loss": _bca_mean(
                   q2_mice["delta_log_loss"].to_numpy()
                   if len(q2_mice) and "delta_log_loss" in q2_mice else np.array([]), seed=1),
               "delta_auc": _bca_mean(
                   q2_mice["delta_auc"].to_numpy()
                   if len(q2_mice) and "delta_auc" in q2_mice else np.array([]), seed=2),
               "mouse_equal_means": {
                   column: float(q2_mice[column].mean()) for column in q2_metric_columns
                   if len(q2_mice) and column in q2_mice}},
        "q2_nuisance_model": {"regularization": "L2", "C": 1.0,
                              "class_weight": None, "natural_prevalence": True,
                              "selection": "fixed; not tuned on outcomes",
                              "neural_score_class_weight": None},
        "integrity": {"baseline_subtraction_all_passed":
                      bool(integrity_df.passed.all()) if len(integrity_df) else False,
                      "n_sessions": int(len(integrity_df))},
        "secondary_models": {
            "v3.2_project_only": _safe_secondary_model(primary_df, include_novel=False),
        },
    }
    (out / "analysis-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--ids-from", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--cache", type=Path, default=Path("/tmp/allen-meta"))
    p = sub.add_parser("pull")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/allen-neural"))
    p.add_argument("--retries", type=int, default=3)
    s = sub.add_parser("scan")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--neural", type=Path, required=True)
    s.add_argument("--behavior", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--data-release")
    s.add_argument("--data-manifest-sha256")
    args = ap.parse_args()
    if args.cmd == "manifest": return make_manifest(args.ids_from, args.out, args.cache)
    if args.cmd == "scan":
        return scan(args.neural, args.manifest, args.behavior, args.out,
                    data_release=args.data_release,
                    data_manifest_sha256=args.data_manifest_sha256)
    manifest = pd.read_csv(args.manifest)
    k, n = parse_shard(args.shard)
    containers = sorted(manifest.ophys_container_id.astype(int).unique())
    mine = containers[k-1::n]
    if not mine: raise SystemExit(f"empty container shard {k}/{n}")
    return pull(manifest, mine, args.out, args.cache, args.retries,
                f"_pull_{k:02d}-of-{n:02d}.json")


if __name__ == "__main__":
    raise SystemExit(main())

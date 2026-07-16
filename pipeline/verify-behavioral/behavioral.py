#!/usr/bin/env python3
"""
Behaviour pipeline · two stages, because they have opposite cost profiles.

  stage 1   pull    sharded, dumb, ONCE.  26.4 GB of NWB -> 36 MB of bundles
  stage 2   scan    local, seconds.       sweep every construct parameter freely

    python pipeline/verify-behavioral/behavioral.py pull \
        --ids-from split/dev_mice.csv --shard 3/10 --out bundles/
    python pipeline/verify-behavioral/behavioral.py scan bundles/ \
        --ids-from split/dev_mice.csv --sweep

Why not one pass
----------------
extract_v2.py's `behavior_scan` downloaded and computed together, so changing
BOUT_GAP from 0.7 to 0.5 meant re-pulling 26 GB. The construct parameters are
precisely the thing that must be swept -- the pilot showed two defensible
constructs disagreeing 100% on a 96.5%-abort session. Baking them into the
download pass makes the one decision dev exists to make cost a day per iteration.

The reduction ratio is ~1500:1 because an NWB carries the whole session and we
need three tables. Everything behaviour-related goes in the bundle, on the
Appendix A principle: re-pulling costs 26 GB, over-storing costs 18 MB.

Bundles and scan outputs are published as Release assets. The manual workflow
records which frozen split Release and source commit produced them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── construct parameters. NOT frozen. This is the dev decision. Stage 2 sweeps. ──
BOUT_GAP = 0.7      # s between licks that starts a new bout
HALF_WIN = 25       # +/- trials in the rolling window
RR_ALLEN = 2.0      # rewards/min                      (Allen's engaged_trial_count)
RR_PIET = 0.5       # 1 reward / 120 s                 (Piet)
BR_PIET = 6.0       # 1 bout   /  10 s                 (Piet)
GUARD = 10          # trials dropped around a label flip
FIT_END = 0.30      # end of the decoding window
MIN_LATE_HIT = 20
MIN_MISS = 20
CONTAM_MIN_HITS = 20
MISS_SWEEP = (10, 15, 20, 25, 30)

BUNDLE_SUFFIXES = (
    "trials.parquet",
    "stim.parquet",
    "rewards.parquet",
    "licks.parquet",
    "meta.json",
)


def bundle_paths(out: Path, bsid: int) -> list[Path]:
    return [out / f"{int(bsid)}.{suffix}" for suffix in BUNDLE_SUFFIXES]


def bundle_complete(out: Path, bsid: int) -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in bundle_paths(out, bsid))


def parse_shard(value: str) -> tuple[int, int]:
    try:
        fields = value.split("/")
        if len(fields) != 2:
            raise ValueError
        k, n = map(int, fields)
    except ValueError as exc:
        raise ValueError(f"invalid shard {value!r}; expected k/N") from exc
    if n < 1 or k < 1 or k > n:
        raise ValueError(f"invalid shard {value!r}; require 1 <= k <= N")
    return k, n


def retryable_error(exc: Exception) -> bool:
    """Retry transport/checksum failures, not deterministic decode/data bugs."""
    module = type(exc).__module__
    if isinstance(exc, (TypeError, ValueError, KeyError, AssertionError)):
        return False
    return not module.startswith(("hdmf", "pynwb", "pyarrow", "pandas"))


# ═══════════════════════════════════════════════════════════════════ stage 1 ══
def pull(ids: list[int], out: Path, cache_dir: Path, *, retries: int = 3,
         report_name: str = "_pull.json") -> int:
    """Download, reduce, delete. Peak disk is one NWB, not the full cohort.

    allensdk's cache KEEPS every file it fetches; left alone it fills the runner
    at session ~90. Each session gets a private cache dir that is destroyed
    immediately, so disk is O(1) in the number of sessions.
    """
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)

    ids = [int(x) for x in ids]
    if not ids:
        raise ValueError("no behavior session IDs were assigned to this shard")
    if retries < 1:
        raise ValueError("retries must be at least 1")

    out.mkdir(parents=True, exist_ok=True)
    staging_root = out / ".staging"
    staging_root.mkdir(exist_ok=True)
    ok, skipped, failed = [], [], []
    for i, bsid in enumerate(ids, 1):
        if bundle_complete(out, bsid):
            skipped.append(bsid)
            print(f"[{i}/{len(ids)}] {bsid}  already complete; skipping", flush=True)
            continue

        # A terminated prior attempt must never be mistaken for a valid bundle.
        for path in bundle_paths(out, bsid):
            path.unlink(missing_ok=True)

        t0 = time.time()
        last_error = None
        attempts_used = 0
        for attempt in range(1, retries + 1):
            attempts_used = attempt
            cd = cache_dir / f"{bsid}-attempt-{attempt}"
            stage = staging_root / str(bsid)
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            try:
                print(f"[{i}/{len(ids)}] {bsid}  attempt {attempt}/{retries} starting",
                      flush=True)
                cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cd)
                bs = cache.get_behavior_session(bsid)

                tr = bs.trials.reset_index()
                tr["lick_times"] = tr["lick_times"].apply(
                    lambda a: np.asarray(a, float).tolist())        # parquet-safe
                tr.to_parquet(stage / f"{bsid}.trials.parquet")

                sp = bs.stimulus_presentations.reset_index()
                keep = [c for c in ("stimulus_presentations_id", "start_time", "stop_time",
                                    "image_name", "image_index", "omitted", "is_change",
                                    "is_image_novel", "flashes_since_change", "trials_id",
                                    "stimulus_block", "active") if c in sp.columns]
                sp[keep].to_parquet(stage / f"{bsid}.stim.parquet")

                rw = bs.rewards.reset_index()
                rw.to_parquet(stage / f"{bsid}.rewards.parquet")
                pd.DataFrame({"timestamps": bs.licks["timestamps"].to_numpy(float)}
                             ).to_parquet(stage / f"{bsid}.licks.parquet")
                (stage / f"{bsid}.meta.json").write_text(
                    json.dumps({key: str(value) for key, value in bs.metadata.items()}))

                if not bundle_complete(stage, bsid):
                    raise RuntimeError("staged bundle is incomplete")
                for source, target in zip(bundle_paths(stage, bsid), bundle_paths(out, bsid)):
                    source.replace(target)
                if not bundle_complete(out, bsid):
                    raise RuntimeError("published bundle is incomplete")

                mb = sum(path.stat().st_size for path in bundle_paths(out, bsid)) / 1e6
                ok.append(dict(behavior_session_id=int(bsid), mb=round(float(mb), 3),
                               n_trials=int(len(tr)), sec=round(time.time() - t0, 1),
                               attempts=attempt))
                print(f"[{i}/{len(ids)}] {bsid}  {len(tr):5d} trials  "
                      f"{mb:6.3f} MB  {time.time()-t0:5.1f}s", flush=True)
                last_error = None
                break
            except Exception as exc:
                last_error = traceback.format_exc(limit=5)
                for path in bundle_paths(out, bsid):
                    path.unlink(missing_ok=True)
                will_retry = retryable_error(exc) and attempt < retries
                action = "retrying" if will_retry else "not retrying"
                print(f"[{i}/{len(ids)}] {bsid}  attempt {attempt}/{retries} FAILED; "
                      f"{action}", flush=True)
                if not will_retry:
                    print(last_error, flush=True)
                    break
                time.sleep(5 * (2 ** (attempt - 1)))
            finally:
                shutil.rmtree(cd, ignore_errors=True)      # <- the whole disk story
                shutil.rmtree(stage, ignore_errors=True)

        if last_error is not None:
            failed.append(dict(behavior_session_id=int(bsid), err=last_error,
                               attempts=attempts_used))

    shutil.rmtree(staging_root, ignore_errors=True)
    report = dict(ok=ok, skipped=[int(x) for x in skipped], failed=failed,
                  requested=[int(x) for x in ids])
    (out / report_name).write_text(json.dumps(report, indent=2))
    if ok:
        d = pd.DataFrame(ok)
        print(f"\n{len(ok)} downloaded / {len(skipped)} resumed / {len(failed)} failed | "
              f"{d.mb.sum():.1f} MB | "
              f"{d.sec.sum()/60:.1f} min | {d.sec.mean():.1f} s/session")
    # A shard that half-worked must not look like success.
    return 1 if failed else 0


# ═══════════════════════════════════════════════════════════════════ stage 2 ══
def _bouts(t, gap):
    t = np.sort(t)
    return t[np.r_[True, np.diff(t) > gap]] if len(t) else t


def _rate(ev, ts, te, half, loo=True):
    n = len(ts); i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half, n - 1)
    ev = np.sort(ev)
    cnt = (np.searchsorted(ev, te[hi]) - np.searchsorted(ev, ts[lo])).astype(float)
    span = te[hi] - ts[lo]
    if loo:
        cnt -= np.searchsorted(ev, te) - np.searchsorted(ev, ts)
        span -= te - ts
    return 60.0 * cnt / np.maximum(span, 1e-9)


def _guard(e, k):
    keep = np.ones(len(e), bool)
    for j in np.where(np.r_[False, e[1:] != e[:-1]])[0]:
        keep[max(0, j - k):j + k + 1] = False
    keep[:10] = False                      # rate is NaN for the first 10 trials
    return keep


def _hysteresis(rate: np.ndarray, span_minutes: np.ndarray,
                enter: float = RR_ALLEN) -> np.ndarray:
    """Forward Schmitt state; its width is the Poisson SE at the entry rate."""
    state = np.zeros(len(rate), dtype=bool)
    engaged = False
    for i, (value, span) in enumerate(zip(rate, span_minutes)):
        if not np.isfinite(value) or not np.isfinite(span) or span <= 0:
            state[i] = engaged
            continue
        exit_at = max(0.0, enter - np.sqrt(enter / span))
        if not engaged and value > enter:
            engaged = True
        elif engaged and value < exit_at:
            engaged = False
        state[i] = engaged
    return state


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    if not len(mask) or not mask.any():
        return np.array([], dtype=int)
    starts = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1]
    ends = np.r_[starts[1:], len(mask)]
    return (ends - starts)[mask[starts]]


def _rate_with_span(ev, ts, te, half, loo=True):
    n = len(ts); i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half, n - 1)
    ev = np.sort(ev)
    cnt = (np.searchsorted(ev, te[hi]) - np.searchsorted(ev, ts[lo])).astype(float)
    span = te[hi] - ts[lo]
    if loo:
        cnt -= np.searchsorted(ev, te) - np.searchsorted(ev, ts)
        span -= te - ts
    span_minutes = np.maximum(span, 1e-9) / 60.0
    return cnt / span_minutes, span_minutes


def validate_bundle_set(b: Path, ids_from: Path | None = None) -> list[int]:
    """Require complete bundles and, when supplied, an exact expected ID set."""
    if not b.is_dir():
        raise ValueError(f"bundle directory does not exist: {b}")

    found = set()
    for path in b.iterdir():
        if not path.is_file():
            continue
        for suffix in BUNDLE_SUFFIXES:
            marker = f".{suffix}"
            if path.name.endswith(marker):
                prefix = path.name[:-len(marker)]
                if prefix.isdigit():
                    found.add(int(prefix))
                break
    if not found:
        raise ValueError(f"no behavior bundles found in {b}")

    incomplete = {
        bsid: [path.name for path in bundle_paths(b, bsid)
               if not path.is_file() or path.stat().st_size == 0]
        for bsid in sorted(found) if not bundle_complete(b, bsid)
    }
    if incomplete:
        raise ValueError(f"incomplete behavior bundles: {incomplete}")

    if ids_from is not None:
        expected_df = pd.read_csv(ids_from)
        if "behavior_session_id" not in expected_df.columns:
            raise ValueError(f"{ids_from} has no behavior_session_id column")
        expected = set(expected_df["behavior_session_id"].dropna().astype(int).tolist())
        if not expected:
            raise ValueError(f"{ids_from} contains no behavior session IDs")
        missing, extra = sorted(expected - found), sorted(found - expected)
        if missing or extra:
            raise ValueError(f"bundle ID mismatch; missing={missing}, extra={extra}")
    return sorted(found)


def load(b: Path, bsid: int):
    tr = pd.read_parquet(b / f"{bsid}.trials.parquet")
    sp = pd.read_parquet(b / f"{bsid}.stim.parquet")
    rw = pd.read_parquet(b / f"{bsid}.rewards.parquet")
    lk = pd.read_parquet(b / f"{bsid}.licks.parquet")["timestamps"].to_numpy(float)
    md = json.loads((b / f"{bsid}.meta.json").read_text())
    ar = next((c for c in ("auto_rewarded", "autorewarded") if c in rw.columns), None)
    earned = (rw.loc[~rw[ar].astype(bool), "timestamps"] if ar else rw["timestamps"]).to_numpy(float)
    return tr, sp, lk, earned, md


def _trial_novelty(tr: pd.DataFrame, sp: pd.DataFrame) -> np.ndarray:
    """Use Allen's field, never OPHYS_4/6 names; unknown stays missing."""
    out = np.full(len(tr), np.nan, dtype=object)
    if "is_image_novel" not in sp.columns:
        return out
    values = sp["is_image_novel"]
    valid = values.notna()
    if "trials_id" in sp.columns and valid.any():
        by_trial = sp.loc[valid].groupby("trials_id")["is_image_novel"].agg(
            lambda x: bool(pd.Series(x).astype(bool).max()))
        trial_ids = (tr["trials_id"] if "trials_id" in tr.columns
                     else pd.Series(tr.index, index=tr.index))
        return trial_ids.map(by_trial).to_numpy(dtype=object)
    unique = pd.Series(values.loc[valid]).astype(bool).unique()
    if len(unique) == 1:
        out[:] = bool(unique[0])
    return out


def label_session(tr: pd.DataFrame, sp: pd.DataFrame, lk: np.ndarray,
                  earned: np.ndarray, *, bout_gap=BOUT_GAP, half=HALF_WIN,
                  rr_allen=RR_ALLEN, rr_piet=RR_PIET, br_piet=BR_PIET,
                  guard=GUARD) -> tuple[pd.DataFrame, dict, dict]:
    """Return lossless trial labels plus session and guard diagnostics."""
    ts, te = tr["start_time"].to_numpy(float), tr["stop_time"].to_numpy(float)
    rr, span = _rate_with_span(earned, ts, te, half)
    br, _ = _rate_with_span(_bouts(lk, bout_gap), ts, te, half)
    A = rr > rr_allen
    B = ~((br < br_piet) & (rr < rr_piet))
    AH = _hysteresis(rr, span, rr_allen)
    keep_a, keep_b, keep_ah = (_guard(x, guard) for x in (A, B, AH))

    hit = tr["hit"].astype(bool).to_numpy()
    miss = tr["miss"].astype(bool).to_numpy()
    aborted = tr["aborted"].astype(bool).to_numpy()
    rl = tr["response_latency"].to_numpy(float)
    late_hit = hit & np.isfinite(rl) & (rl > FIT_END)
    early_hit = hit & np.isfinite(rl) & (rl <= FIT_END)
    latency_status = np.where(~hit, "not_hit",
                              np.where(np.isfinite(rl), "eligible", "ineligible_nonfinite"))
    go = hit | miss
    impulsive = (br >= br_piet) & (rr < rr_piet)
    first_ten = np.arange(len(tr)) < 10

    trial_id = (tr["trials_id"].to_numpy() if "trials_id" in tr.columns
                else np.arange(len(tr), dtype=int))
    labels = pd.DataFrame({
        "trial_id": trial_id,
        "trial_index": np.arange(len(tr), dtype=int),
        "start_time": ts,
        "stop_time": te,
        "change_time": (tr["change_time"].to_numpy(float)
                        if "change_time" in tr.columns else np.full(len(tr), np.nan)),
        "hit": hit, "late_hit": late_hit, "early_hit": early_hit,
        "miss": miss, "aborted": aborted, "go": go,
        "response_latency": rl,
        "latency_status": latency_status,
        "reward_rate": rr, "bout_rate": br, "rate_span_minutes": span,
        "engaged_A": A, "keep_A": keep_a,
        "engaged_B": B, "keep_B": keep_b,
        "engaged_A_hysteretic": AH, "keep_A_hysteretic": keep_ah,
        "impulsive_regime": impulsive,
        "first_ten": first_ten,
        "is_image_novel": _trial_novelty(tr, sp),
    })

    def counts(state, keep):
        raw_go = int((go & state).sum())
        kept_go = int((go & state & keep).sum())
        flips = int(np.count_nonzero(state[1:] != state[:-1]))
        return dict(raw_go=raw_go, kept_go=kept_go,
                    guard_loss=(float((raw_go - kept_go) / raw_go) if raw_go else np.nan),
                    transitions=flips,
                    median_run=(float(np.median(_run_lengths(state)))
                                if len(_run_lengths(state)) else 0.0),
                    late_hit=int((late_hit & state & keep).sum()),
                    hit=int((hit & state & keep).sum()),
                    miss=int((miss & state & keep).sum()))

    ca, cb, ch = counts(A, keep_a), counts(B, keep_b), counts(AH, keep_ah)
    finite_hit_rt = rl[hit & np.isfinite(rl) & (rl > 0)]
    contam = float((finite_hit_rt <= FIT_END).mean()) if len(finite_hit_rt) else np.nan
    contam_status = ("ineligible_nan" if not np.isfinite(contam) else
                     "ineligible_low_n" if len(finite_hit_rt) < CONTAM_MIN_HITS else
                     "eligible")
    session = dict(
        n_trials=int(len(tr)), n_hit=int(hit.sum()), n_late_hit=int(late_hit.sum()),
        n_early_hit=int(early_hit.sum()), n_miss=int(miss.sum()),
        abort_frac=float(aborted.mean()), survived_frac=float((~aborted).mean()),
        contam=contam, contam_n=int(len(finite_hit_rt)), contam_status=contam_status,
        eng_A=float(A.mean()), eng_B=float(B.mean()), eng_A_hysteretic=float(AH.mean()),
        late_hit_A=ca["late_hit"], miss_A=ca["miss"],
        late_hit_B=cb["late_hit"], miss_B=cb["miss"],
        late_hit_A_hysteretic=ch["late_hit"], miss_A_hysteretic=ch["miss"],
        impulsive_frac=float(impulsive.mean()),
        impulsive_go=int((impulsive & go).sum()), total_go=int(go.sum()),
        impulsive_abort_rate=(float(aborted[impulsive].mean()) if impulsive.any() else np.nan),
        nonimpulsive_abort_rate=(float(aborted[~impulsive].mean()) if (~impulsive).any() else np.nan),
    )
    guard_diag = {}
    for name, values in (("A", ca), ("B", cb), ("A_hysteretic", ch)):
        guard_diag.update({f"{key}_{name}": value for key, value in values.items()})
    return labels, session, guard_diag


def diagnose(tr, sp, lk, earned, md, *, bout_gap=BOUT_GAP, half=HALF_WIN,
             rr_allen=RR_ALLEN, rr_piet=RR_PIET, br_piet=BR_PIET, guard=GUARD) -> dict:
    _, d, _ = label_session(tr, sp, lk, earned, bout_gap=bout_gap, half=half,
                            rr_allen=rr_allen, rr_piet=rr_piet,
                            br_piet=br_piet, guard=guard)
    d.update(mouse_id=md.get("mouse_id"), project_code=md.get("project_code"),
             equipment_name=md.get("equipment_name"), session_type=md.get("session_type"))
    return d


def scan(b: Path, sweep: bool, ids_from: Path | None = None) -> int:
    ids = validate_bundle_set(b, ids_from)
    cached = [load(b, i) for i in ids]
    print(f"{len(ids)} sessions loaded from {sum(f.stat().st_size for f in b.glob('*')) / 1e6:.1f} MB\n")
    session_rows, guard_rows, label_rows, persistence_rows = [], [], [], []
    for bsid, data in zip(ids, cached):
        tr, sp, lk, earned, md = data
        labels, ses, gd = label_session(tr, sp, lk, earned)
        common = dict(behavior_session_id=int(bsid), mouse_id=md.get("mouse_id"),
                      project_code=md.get("project_code"),
                      equipment_name=md.get("equipment_name"),
                      session_type=md.get("session_type"))
        labels.insert(0, "behavior_session_id", int(bsid))
        for key, value in reversed(list(common.items())[1:]):
            labels.insert(1, key, value)
        label_rows.append(labels)
        session_rows.append({**common, **ses})
        guard_rows.append({**common, **gd})

        imp = labels["impulsive_regime"].to_numpy(bool)
        rr_low = labels["reward_rate"].to_numpy(float) < RR_PIET
        br_high = labels["bout_rate"].to_numpy(float) >= BR_PIET
        shifts = np.unique(np.linspace(1, max(1, len(imp) - 1),
                                       min(199, max(1, len(imp) - 1))).astype(int))
        null_frac, null_run = [], []
        for shift in shifts:
            surrogate = rr_low & np.roll(br_high, int(shift))
            runs = _run_lengths(surrogate)
            null_frac.append(float(surrogate.mean()))
            null_run.append(int(runs.max()) if len(runs) else 0)
        observed_runs = _run_lengths(imp)
        observed_max = int(observed_runs.max()) if len(observed_runs) else 0
        persistence_rows.append({**common, "impulsive_frac": float(imp.mean()),
            "max_run": observed_max, "n_runs": int(len(observed_runs)),
            "null_mean_frac": float(np.mean(null_frac)),
            "null_p_max_run": float((1 + sum(x >= observed_max for x in null_run)) /
                                    (1 + len(null_run)))})

    df = pd.DataFrame(session_rows)
    reasons = []
    for row in df.itertuples():
        why = []
        if int(row.late_hit_B) < MIN_LATE_HIT: why.append("low_late_hit")
        if int(row.miss_B) < MIN_MISS: why.append("low_miss")
        reasons.append(";".join(why))
    df["behavioral_eligible"] = [not x for x in reasons]
    df["eligibility_reasons"] = reasons
    eligibility = df[["behavior_session_id", "mouse_id", "behavioral_eligible",
                      "eligibility_reasons", "late_hit_B", "miss_B", "contam",
                      "contam_n", "contam_status"]].copy()

    sweep_rows = []
    for threshold in MISS_SWEEP:
        for construct, late_col, miss_col in (
                ("v3.1_B_K50", "late_hit_B", "miss_B"),
                ("v3_A_all_C0.1", "late_hit_A", "miss_A")):
            mask = df[miss_col].ge(threshold)
            if construct.startswith("v3.1"):
                mask &= df[late_col].ge(MIN_LATE_HIT)
            selected = df.loc[mask]
            sweep_rows.append(dict(construct=construct, miss_threshold=threshold,
                                   min_late_hit=(MIN_LATE_HIT if construct.startswith("v3.1") else None),
                                   n_sessions=int(len(selected)),
                                   n_mice=int(selected.mouse_id.nunique()),
                                   median_late_hit=float(selected[late_col].median()) if len(selected) else np.nan,
                                   median_miss=float(selected[miss_col].median()) if len(selected) else np.nan))

    trial_labels = pd.concat(label_rows, ignore_index=True)
    trial_labels.to_parquet(b / "_trial_labels.parquet", index=False)
    df.to_parquet(b / "_session_scan.parquet", index=False)
    df.to_parquet(b / "_scan.parquet", index=False)  # compatibility
    eligibility.to_parquet(b / "_eligibility.parquet", index=False)
    pd.DataFrame(guard_rows).to_parquet(b / "_guard_diagnostics.parquet", index=False)
    pd.DataFrame(persistence_rows).to_parquet(b / "_persistence.parquet", index=False)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_parquet(b / "_yield_sweep.parquet", index=False)
    sweep_df.to_parquet(b / "_sweep.parquet", index=False)
    survival = df[["behavior_session_id", "mouse_id", "project_code", "n_trials",
                   "abort_frac", "survived_frac"]].copy()
    survival.to_parquet(b / "_survival.parquet", index=False)
    survival_model = {"formula": "survived ~ C(project_code)", "cluster": "mouse_id"}
    try:
        import statsmodels.api as sm
        trial_survival = trial_labels[["mouse_id", "project_code", "aborted"]].copy()
        trial_survival["survived"] = (~trial_survival.aborted.astype(bool)).astype(int)
        gee = sm.GEE.from_formula("survived ~ C(project_code)", groups="mouse_id",
                                  data=trial_survival,
                                  family=sm.families.Binomial(),
                                  cov_struct=sm.cov_struct.Exchangeable()).fit()
        survival_model.update(params={str(k): float(v) for k, v in gee.params.items()},
                              conf_int={str(k): [float(x) for x in gee.conf_int().loc[k]]
                                        for k in gee.params.index}, converged=bool(gee.converged))
        from scipy.stats import bootstrap
        mouse_survival = trial_survival.groupby(["project_code", "mouse_id"], as_index=False).survived.mean()
        boot = {}
        for project, group in mouse_survival.groupby("project_code"):
            values = group.survived.to_numpy(float)
            if len(values) >= 2:
                ci = bootstrap((values,), np.mean, method="BCa", n_resamples=2000,
                               random_state=np.random.default_rng(31))
                boot[str(project)] = {"mean": float(values.mean()),
                                      "low": float(ci.confidence_interval.low),
                                      "high": float(ci.confidence_interval.high),
                                      "n_mice": int(len(values))}
        survival_model["mouse_cluster_bootstrap"] = boot
    except Exception as exc:
        survival_model["error"] = f"{type(exc).__name__}: {exc}"
    (b / "_survival_model.json").write_text(json.dumps(survival_model, indent=2) + "\n")

    manifest = {
        "schema": "behavioral-v3.1",
        "dev_ids_sha256": hashlib.sha256(
            "\n".join(map(str, sorted(ids))).encode()).hexdigest(),
        "n_dev_sessions": len(ids),
        "primary": "B-engaged late-hit-vs-miss",
        "frozen_comparator": "A-engaged all-cell C=0.1",
        "parameters": {"bout_gap": BOUT_GAP, "half_win": HALF_WIN,
                       "rr_allen": RR_ALLEN, "rr_piet": RR_PIET,
                       "br_piet": BR_PIET, "guard": GUARD,
                       "fit_end": FIT_END, "min_late_hit": MIN_LATE_HIT,
                       "min_miss": MIN_MISS, "contam_min_hits": CONTAM_MIN_HITS,
                       "miss_sweep": list(MISS_SWEEP)},
        "authoritative_sesoi": None,
        "files": ["_trial_labels.parquet", "_session_scan.parquet",
                  "_eligibility.parquet", "_guard_diagnostics.parquet",
                  "_persistence.parquet", "_yield_sweep.parquet", "_survival.parquet",
                  "_survival_model.json"],
    }
    (b / "behavioral-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("DEV v3.1: late hit vs miss (never hit vs miss)")
    print(f"eligible: {int(df.behavioral_eligible.sum())}/{len(df)} sessions / "
          f"{df.loc[df.behavioral_eligible, 'mouse_id'].nunique()} mice")
    print(sweep_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    p = s.add_parser("pull")
    p.add_argument("--ids-from", type=Path, required=True)
    p.add_argument("--shard", default="1/1", help="k/N, 1-indexed, strided")
    p.add_argument("--out", type=Path, default=Path("bundles"))
    p.add_argument("--cache", type=Path, default=Path("/tmp/allen"))
    p.add_argument("--retries", type=int, default=3)
    q = s.add_parser("scan")
    q.add_argument("dir", type=Path)
    q.add_argument("--ids-from", type=Path)
    q.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    if a.cmd == "scan":
        raise SystemExit(scan(a.dir, a.sweep, a.ids_from))
    ids = sorted(pd.read_csv(a.ids_from)["behavior_session_id"].astype(int).unique())
    if not ids:
        raise SystemExit(f"no behavior session IDs found in {a.ids_from}")
    try:
        k, n = parse_shard(a.shard)
    except ValueError as exc:
        ap.error(str(exc))
    mine = ids[k - 1::n]        # strided, so shards balance even if size tracks order
    if not len(mine):
        raise SystemExit(f"shard {k}/{n} is empty for {len(ids)} sessions")
    print(f"shard {k}/{n}: {len(mine)} of {len(ids)} sessions")
    report_name = f"_pull_{k:02d}-of-{n:02d}.json"
    raise SystemExit(pull(mine, a.out, a.cache, retries=a.retries,
                          report_name=report_name))

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
        for attempt in range(1, retries + 1):
            cd = cache_dir / f"{bsid}-attempt-{attempt}"
            stage = staging_root / str(bsid)
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            try:
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
            except Exception:
                last_error = traceback.format_exc(limit=5)
                for path in bundle_paths(out, bsid):
                    path.unlink(missing_ok=True)
                print(f"[{i}/{len(ids)}] {bsid}  attempt {attempt}/{retries} FAILED",
                      flush=True)
                if attempt < retries:
                    time.sleep(5 * (2 ** (attempt - 1)))
            finally:
                shutil.rmtree(cd, ignore_errors=True)      # <- the whole disk story
                shutil.rmtree(stage, ignore_errors=True)

        if last_error is not None:
            failed.append(dict(behavior_session_id=int(bsid), err=last_error,
                               attempts=retries))

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
    rw = pd.read_parquet(b / f"{bsid}.rewards.parquet")
    lk = pd.read_parquet(b / f"{bsid}.licks.parquet")["timestamps"].to_numpy(float)
    md = json.loads((b / f"{bsid}.meta.json").read_text())
    ar = next((c for c in ("auto_rewarded", "autorewarded") if c in rw.columns), None)
    earned = (rw.loc[~rw[ar].astype(bool), "timestamps"] if ar else rw["timestamps"]).to_numpy(float)
    return tr, lk, earned, md


def diagnose(tr, lk, earned, md, *, bout_gap=BOUT_GAP, half=HALF_WIN,
             rr_allen=RR_ALLEN, rr_piet=RR_PIET, br_piet=BR_PIET, guard=GUARD) -> dict:
    ts, te = tr["start_time"].to_numpy(float), tr["stop_time"].to_numpy(float)
    rr = _rate(earned, ts, te, half)
    br = _rate(_bouts(lk, bout_gap), ts, te, half)
    A = rr > rr_allen                              # reward rate only
    B = ~((br < br_piet) & (rr < rr_piet))         # NOT(low bouts AND low rewards)

    hit = tr["hit"].astype(bool).to_numpy()
    miss = tr["miss"].astype(bool).to_numpy()
    rl = tr["response_latency"].to_numpy(float)
    rt = rl[hit & np.isfinite(rl) & (rl > 0)]
    dur = (te.max() - ts.min()) / 60

    d = dict(n_trials=len(tr), abort_frac=float(tr["aborted"].astype(bool).mean()),
             n_hit=int(hit.sum()), n_miss=int(miss.sum()),
             session_reward_rate=float(hit.sum() / dur),
             median_hit_rt=float(np.median(rt)) if len(rt) else np.nan,
             # §6's "94% of hits lick after 0.30s" is an n=1 fact. This is the field.
             contam=float((rt <= FIT_END).mean()) if len(rt) else np.nan,
             # "miss" is not one thing: +inf = never licked, finite = licked late
             miss_no_lick=float(np.isposinf(rl[miss]).mean()) if miss.any() else np.nan,
             mouse_id=md.get("mouse_id"), project_code=md.get("project_code"),
             equipment_name=md.get("equipment_name"), session_type=md.get("session_type"))
    for tag, e in (("A", A), ("B", B)):
        k = _guard(e, guard)
        d[f"eng_{tag}"] = float(e.mean())
        d[f"hit_{tag}"] = int((hit & e & k).sum())
        d[f"miss_{tag}"] = int((miss & e & k).sum())
    d["disagree"] = float((A != B).mean())
    return d


def scan(b: Path, sweep: bool, ids_from: Path | None = None) -> int:
    ids = validate_bundle_set(b, ids_from)
    cached = [load(b, i) for i in ids]                      # 36 MB, fits in RAM
    print(f"{len(ids)} sessions loaded from {sum(f.stat().st_size for f in b.glob('*')) / 1e6:.1f} MB\n")

    df = pd.DataFrame([diagnose(*c) for c in cached])
    df.insert(0, "behavior_session_id", ids)
    df.to_parquet(b / "_scan.parquet")

    print("THE DEV DECISIONS, AS DISTRIBUTIONS")
    print(f"{'':22s}" + "".join(f"{p:>9s}" for p in ("p10", "p25", "p50", "p75", "p90")))
    for c in ["abort_frac", "median_hit_rt", "contam", "miss_no_lick",
              "eng_A", "eng_B", "disagree", "miss_A", "miss_B"]:
        q = df[c].quantile([.1, .25, .5, .75, .9])
        print(f"{c:22s}" + "".join(f"{v:9.3f}" for v in q))

    print(f"\nsessions unusable for Q1 (>20% of hits lick inside the fit window): "
          f"{int((df.contam > .20).sum())}/{len(df)}")
    print(f"sessions with abort_frac > 0.90: {int((df.abort_frac > .90).sum())}/{len(df)}")
    for tag in ("A", "B"):
        qualified = df[df[f"miss_{tag}"] >= 20]
        print(f"n_engaged_miss>=20 ({tag}): {len(qualified)} sessions / "
              f"{qualified.mouse_id.nunique()} mice")

    if sweep:
        # The entire reason for the two-stage split: this costs seconds, not 26 GB.
        print("\n\nPARAMETER SWEEP  (each row = a full re-analysis of all sessions)")
        print(f"{'bout_gap':>9s}{'half':>6s}{'rr_allen':>9s}{'br_piet':>8s}"
              f"{'eng_A':>7s}{'eng_B':>7s}{'disagr':>8s}{'missA':>7s}{'missB':>7s}"
              f"{'sesA':>6s}{'miceA':>7s}{'sesB':>6s}{'miceB':>7s}")
        sweep_rows = []
        for gap in (0.5, 0.7, 1.0):
            for half in (15, 25, 40):
                for rra in (1.0, 2.0):
                    for brp in (3.0, 6.0):
                        r = pd.DataFrame([diagnose(*c, bout_gap=gap, half=half,
                                                   rr_allen=rra, br_piet=brp) for c in cached])
                        qa, qb = r[r.miss_A >= 20], r[r.miss_B >= 20]
                        row = dict(
                            bout_gap=gap, half=half, rr_allen=rra, br_piet=brp,
                            median_eng_A=float(r.eng_A.median()),
                            median_eng_B=float(r.eng_B.median()),
                            median_disagree=float(r.disagree.median()),
                            median_miss_A=float(r.miss_A.median()),
                            median_miss_B=float(r.miss_B.median()),
                            qualifying_sessions_A=int(len(qa)),
                            qualifying_mice_A=int(qa.mouse_id.nunique()),
                            qualifying_sessions_B=int(len(qb)),
                            qualifying_mice_B=int(qb.mouse_id.nunique()),
                        )
                        sweep_rows.append(row)
                        print(f"{gap:9.1f}{half:6d}{rra:9.1f}{brp:8.1f}"
                              f"{r.eng_A.median():7.3f}{r.eng_B.median():7.3f}"
                              f"{r.disagree.median():8.3f}"
                              f"{r.miss_A.median():7.0f}{r.miss_B.median():7.0f}"
                              f"{len(qa):6d}{qa.mouse_id.nunique():7d}"
                              f"{len(qb):6d}{qb.mouse_id.nunique():7d}")
        pd.DataFrame(sweep_rows).to_parquet(b / "_sweep.parquet", index=False)
        print("\n>> If eng_A/eng_B move a lot across this grid, the construct is not a\n"
              ">> detail and §5.4 cannot be frozen by assertion. If abort_frac is bimodal,\n"
              ">> consider a THIRD state (impulsive) rather than forcing the dichotomy.")
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

#!/usr/bin/env python3
"""
dev/confirm mouse split  ·  prereg §0.2-0.3

    python split.py build  --out split/
    python split.py verify split/split_manifest.json

Runs on the manifest CSV only. No NWB, no traces, no behaviour. ~1.3 MB, seconds.

Why stratified rather than a single hash draw
---------------------------------------------
§0.3 v3 said: sort mice by sha256(mouse_id), take the first 25%. Two holes.

1. NO BALANCE GUARANTEE. 39 mice, 23 VisualBehavior / 16 Task1B, draw 10:
   hypergeometric SD = 1.36, so dev could hold 2 Task1B or 7. `project_code` is a
   fixed effect in §9 AND the SESOI anchor is measured on dev -- an unbalanced dev
   measures the anchor in a different population than the one it is applied to.
2. NOT AUDITABLE AS FAIR. A single draw is unfalsifiable: if it comes out skewed
   you either accept it or you are gaming it. Rejection sampling over seeds fixes
   balance but re-opens the hole through its fallback rule.

Stratified hash sort closes both: within each stratum, sort by
sha256(f"{mouse_id}") and take the stratum's quota. Deterministic, balanced by
construction, no search, no threshold, no fallback, and re-runnable by anyone.

What may and may not be a stratifier
------------------------------------
ONLY session metadata. NOT behaviour.

  eligible: cre_line, equipment_name, project_code, imaging_depth,
            targeted_structure, session_type
  BANNED:   abort rate, hit rate, engaged fraction, n_engaged_miss, RT

Stratifying on abort rate would make dev more representative and would improve
the SESOI's transportability -- and it would make the split a function of
behaviour, which is the thing the split exists to protect. §0.1 disclosed that
cohort behaviour was already inspected; that disclosure is a confession, not a
licence. Behaviour balance is REPORTED after the fact (§ report_balance) and, if
poor, is carried as a caveat on the anchor. The split is never re-drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

DEV_FRAC = 0.25

# ---- FROZEN cohort definition (metadata only) -------------------------------
CRE = "Slc17a7-IRES2-Cre"
RIG_PREFIX = "CAM2P"
PILOT_MICE = [403491]                 # 775614751's mouse (probe)
PILOT_CONTAINERS = [782536745]        # 775614751's container (probe)

# ---- FROZEN strata RULE, not a frozen strata LIST ---------------------------
# 40 mice will not support three stratifiers: project_code x depth x equipment is
# 12 cells of ~3, all of which collapse. Two slots is the hard ceiling. So which
# two is a decision -- and a decision made by looking at the balance table is the
# split being tuned on the data. The table is metadata-only so no OUTCOME can
# leak, but a convenient dev set can still be shopped for. Hence: a rule declared
# before the run, which the data resolves without a human in the loop.
#
#   slot 1  project_code   always. It is a fixed effect in §9 and the SESOI anchor
#                          is measured on dev, so an unbalanced dev measures the
#                          anchor in a different population than it is applied to.
#
#   slot 2  equipment_name IF it is between-mouse (every mouse maps to one rig),
#                          ELSE depth_bin.
#
# Why equipment wins slot 2 when it is available: rig -> SNR -> n_cells -> AUC is
# the shortest metadata path to the outcome MEASURE. Median neurons/session across
# CAM2P rigs is 151-190, a 26% spread, and d scales with sqrt(n_cells). depth is
# biology and matters, but its effect on the effect SIZE is speculative, and it is
# reported either way. If a mouse spans rigs, equipment is not a mouse-level
# variable at all and the question is moot -- hence the condition, not a judgement.
STRATA_SLOT1 = "project_code"
STRATA_SLOT2_PREF = "equipment_name"
STRATA_SLOT2_FALLBACK = "depth_bin"
DEPTH_CUT = 250
MIN_STRATUM = 4


def choose_strata(coh: pd.DataFrame) -> tuple[list[str], str]:
    """Resolve the slot-2 rule against the manifest. No human in the loop."""
    span = coh.groupby("mouse_id")[STRATA_SLOT2_PREF].nunique()
    between = bool((span == 1).all())
    pick = STRATA_SLOT2_PREF if between else STRATA_SLOT2_FALLBACK
    why = (f"{STRATA_SLOT2_PREF} is between-mouse (max rigs/mouse={span.max()}) -> slot 2"
           if between else
           f"{STRATA_SLOT2_PREF} is NOT between-mouse (max rigs/mouse={span.max()}); "
           f"falling back to {STRATA_SLOT2_FALLBACK}")
    return [STRATA_SLOT1, pick], why


def _h(mouse_id) -> str:
    return hashlib.sha256(str(int(mouse_id)).encode()).hexdigest()


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Hamilton apportionment: floor each quota, hand out the remainder by
    descending fractional part, ties broken by stratum key. Deterministic, and
    the quotas sum to `total` exactly."""
    raw = {k: v * DEV_FRAC for k, v in counts.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(q.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))
    for k in order[:max(rem, 0)]:
        q[k] += 1
    return q


def build_cohort(exp_table: pd.DataFrame) -> pd.DataFrame:
    t = exp_table.reset_index()
    active = ~t["session_type"].str.endswith("_passive")
    m = (t["cre_line"].eq(CRE) & t["equipment_name"].str.startswith(RIG_PREFIX) & active)
    coh = t[m].copy()
    coh = coh[~coh["mouse_id"].isin(PILOT_MICE)]
    coh = coh[~coh["ophys_container_id"].isin(PILOT_CONTAINERS)]
    coh["depth_bin"] = np.where(coh["imaging_depth"] <= DEPTH_CUT, "superficial", "deep")
    return coh


def mouse_table(coh: pd.DataFrame) -> pd.DataFrame:
    """One row per mouse. A mouse's stratum is its modal value; ties -> first
    sorted. In the CAM2P cohort a mouse has one container at one depth, so this
    is almost always degenerate -- the mode is a guard, not a policy."""
    g = coh.groupby("mouse_id")
    mt = pd.DataFrame({
        "n_sessions": g.size(),
        "n_containers": g["ophys_container_id"].nunique(),
        **{c: g[c].agg(lambda s: sorted(s.mode())[0]) for c in
           ["project_code", "depth_bin", "equipment_name", "targeted_structure"]},
        "depth_values": g["imaging_depth"].agg(lambda s: sorted(set(s))),
    }).reset_index()
    mt["hash"] = mt["mouse_id"].map(_h)
    return mt.sort_values("mouse_id").reset_index(drop=True)


def split(mt: pd.DataFrame, strata: list[str], why: str = "") -> tuple[pd.DataFrame, dict]:
    mt = mt.copy()
    mt["stratum"] = mt[strata].astype(str).agg(" | ".join, axis=1)

    # collapse thin strata onto the project_code marginal -- declared, not tuned
    small = mt["stratum"].value_counts()
    thin = set(small[small < MIN_STRATUM].index)
    mt["stratum"] = np.where(mt["stratum"].isin(thin),
                             mt["project_code"].astype(str) + " | <collapsed>",
                             mt["stratum"])

    counts = mt["stratum"].value_counts().to_dict()
    target = int(np.ceil(DEV_FRAC * len(mt)))
    quota = largest_remainder(counts, target)

    dev = []
    for s, k in sorted(quota.items()):
        pool = mt[mt["stratum"].eq(s)].sort_values("hash")   # hash sort WITHIN stratum
        dev += list(pool["mouse_id"].iloc[:k])
    mt["tier"] = np.where(mt["mouse_id"].isin(dev), "dev", "confirm")

    info = dict(dev_frac=DEV_FRAC, strata=strata, strata_rule=why, depth_cut=DEPTH_CUT,
                min_stratum=MIN_STRATUM, collapsed=sorted(thin),
                counts=counts, quota=quota, target=target,
                n_mice=len(mt), n_dev=int((mt.tier == "dev").sum()))
    return mt, info


def report_balance(mt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["project_code", "depth_bin", "equipment_name", "targeted_structure"]:
        for v in sorted(mt[c].astype(str).unique()):
            d = mt[mt.tier.eq("dev")][c].astype(str).eq(v).mean()
            f = mt[mt.tier.eq("confirm")][c].astype(str).eq(v).mean()
            rows.append(dict(var=c, level=v, dev=round(d, 3), confirm=round(f, 3),
                             diff=round(d - f, 3)))
    rows.append(dict(var="n_sessions", level="mean/mouse",
                     dev=round(mt[mt.tier.eq("dev")].n_sessions.mean(), 2),
                     confirm=round(mt[mt.tier.eq("confirm")].n_sessions.mean(), 2),
                     diff=None))
    return pd.DataFrame(rows)


def checksum(mt: pd.DataFrame) -> str:
    ids = sorted(mt.loc[mt.tier.eq("dev"), "mouse_id"].astype(int).tolist())
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def run_provenance() -> dict:
    """Record the environment that performed the draw without affecting it."""
    import allensdk

    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    run_url = (
        f"{server}/{repository}/actions/runs/{run_id}"
        if server and repository and run_id else None
    )
    return {
        "allensdk_version": allensdk.__version__,
        "python_version": platform.python_version(),
        "analysis_commit": os.environ.get("GITHUB_SHA"),
        "github_run_url": run_url,
    }


def build(out: Path) -> int:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=Path("~/allen_cache").expanduser())
    manifest = cache.current_manifest()
    coh = build_cohort(cache.get_ophys_experiment_table())
    frozen = out / "split_manifest.json"
    if frozen.exists():
        print(f"REFUSING: {frozen} exists. The split is drawn once. Use `verify`.")
        return 2
    strata, why = choose_strata(coh)
    mt, info = split(mouse_table(coh), strata, why)

    out.mkdir(parents=True, exist_ok=True)
    print(f"strata rule -> {strata}\n  because: {why}\n")
    print(f"manifest: {manifest}")
    print(f"cohort:   {len(coh)} experiments / {len(mt)} mice "
          f"(pilot mouse {PILOT_MICE} + container {PILOT_CONTAINERS} removed)\n")
    print("strata (mice -> dev quota):")
    for s in sorted(info["counts"]):
        print(f"  {s:44s} {info['counts'][s]:3d} -> {info['quota'][s]}")
    if info["collapsed"]:
        print(f"  collapsed (< {MIN_STRATUM} mice): {info['collapsed']}")

    print(f"\ndev = {info['n_dev']} mice, confirm = {len(mt)-info['n_dev']} mice")
    print("\nbalance:")
    bal = report_balance(mt)
    print(bal.to_string(index=False))

    ck = checksum(mt)
    print(f"\nDEV CHECKSUM  {ck}     <- goes in prereg §0.3; anyone can re-derive it")
    print("dev mice:", sorted(mt.loc[mt.tier.eq('dev'), 'mouse_id'].astype(int)))

    # behavior_session_id lists -- the input to `extract.py behavior-scan`
    for tier in ("dev", "confirm"):
        ids = set(mt.loc[mt.tier.eq(tier), "mouse_id"])
        coh[coh.mouse_id.isin(ids)][
            ["behavior_session_id", "ophys_experiment_id", "mouse_id", "project_code",
             "session_type", "imaging_depth", "targeted_structure", "ophys_container_id"]
        ].to_csv(out / f"{tier}_mice.csv", index=False)
    mt.to_csv(out / "mouse_table.csv", index=False)
    bal.to_csv(out / "balance.csv", index=False)
    (out / "split_manifest.json").write_text(json.dumps(
        dict(allen_manifest=manifest, cre=CRE, rig_prefix=RIG_PREFIX,
             pilot_mice=PILOT_MICE, pilot_containers=PILOT_CONTAINERS,
             provenance=run_provenance(),
             dev_checksum=ck,
             dev_mice=sorted(mt.loc[mt.tier.eq("dev"), "mouse_id"].astype(int).tolist()),
             confirm_mice=sorted(mt.loc[mt.tier.eq("confirm"), "mouse_id"].astype(int).tolist()),
             **info), indent=2))

    print(f"\nwrote {out}/  ->  paste split_manifest.json into prereg §0.3, then:")
    print(f"  python pipeline/verify-behavioral/behavioral.py pull "
          f"--ids-from {out}/dev_mice.csv --out bundles/")
    return 0


def verify(p: Path) -> int:
    """Strictly re-derive every frozen split decision and detect any drift."""
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    j = json.loads(p.read_text())
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=Path("~/allen_cache").expanduser())
    cur = cache.current_manifest()
    coh = build_cohort(cache.get_ophys_experiment_table())
    strata, why = choose_strata(coh)
    mt, info = split(mouse_table(coh), strata, why)
    ck = checksum(mt)
    dev = sorted(mt.loc[mt.tier.eq("dev"), "mouse_id"].astype(int).tolist())
    confirm = sorted(mt.loc[mt.tier.eq("confirm"), "mouse_id"].astype(int).tolist())

    drift = []

    def check(label, recorded, recomputed):
        if recorded != recomputed:
            drift.append(label)
            print(f"! {label} drift:\n  recorded:   {recorded}\n  recomputed: {recomputed}")

    check("Allen manifest", j.get("allen_manifest"), cur)
    check("strata", j.get("strata"), strata)
    check("strata rule", j.get("strata_rule"), why)
    for key in ("dev_frac", "depth_cut", "min_stratum", "collapsed", "counts",
                "quota", "target", "n_mice", "n_dev"):
        check(key, j.get(key), info[key])
    check("dev mice", j.get("dev_mice"), dev)
    check("confirm mice", j.get("confirm_mice"), confirm)
    check("dev checksum", j.get("dev_checksum"), ck)

    if drift:
        print(f"\nSTRICT VERIFY FAILED: {', '.join(drift)}")
        return 1
    print(f"STRICT VERIFY MATCH: {ck} | {len(dev)} dev / {len(confirm)} confirm mice")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("build"); b.add_argument("--out", type=Path, default=Path("split"))
    v = s.add_parser("verify"); v.add_argument("path", type=Path)
    a = ap.parse_args()
    raise SystemExit(build(a.out) if a.cmd == "build" else verify(a.path))

"""Closed-loop confirmation of the frozen S1 rules with REAL resets (no renewal lookup).

Rules are read from results/<fam>_calibration.json (frozen before any test number was seen).
Each policy computes its monitor online, from post-reset logs only, exactly as defined in
run_s1.monitor_rows, and resets the network when its rule fires. Fresh seeds 20.. (never used
before). Utility = mean online accuracy over the 40 tasks; number of resets.

usage: python cl_s1.py <worker> <n_workers> <fam> <seed> [seed ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

import pl_common as P
import run_s1 as R
import eval_s1 as E

OUT = R.HERE / "closed_loop"


def monitor_value(m, logs, tasks, r0, k):
    """Value of monitor m after task k-1 (global index), using only tasks >= r0."""
    j = k - 1
    rec = tasks[j]
    a, b = j * R.STEPS, (j + 1) * R.STEPS
    if m == "acc_mean":
        return rec["online_acc"]
    if m == "loss_mean":
        return float(logs["loss"][a:b].mean())
    if m == "gnorm_mean":
        return float(logs["gnorm"][a:b].mean())
    if m == "pnorm_end":
        return float(logs["pnorm"][b - 1])
    if m in E.DOMAIN:
        return rec[m]
    stat, L, wn = m.split("|")
    x = logs[L].astype(float)
    if wn == "within":
        seg = x[a:b]
    else:
        if b - R.WSPAN < r0 * R.STEPS:
            return np.nan
        seg = x[b - R.WSPAN:b]
    if stat in R.STATS:
        return float(R.STATS[stat](seg))
    return float(R.simple(seg)[stat])


def make_policy(m, rule, hist):
    def policy(logs, tasks, k):
        r0 = hist["r0"]
        hist["v"].append(monitor_value(m, logs, tasks, r0, k))
        v = np.array(hist["v"] + [np.nan] * (R.T - len(hist["v"])))
        fire = E.alarm_index(v, rule) == len(hist["v"]) - 1
        if fire:
            hist["r0"], hist["v"] = k, []
        return fire
    return policy


def fixed_policy(I, hist):
    def policy(logs, tasks, k):
        if k - hist["r0"] >= I:
            hist["r0"] = k
            return True
        return False
    return policy


if __name__ == "__main__":
    wid, nw, fam = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    seeds = [int(s) for s in sys.argv[4:]]
    OUT.mkdir(exist_ok=True)
    chosen = json.load(open(E.RES / f"{fam}_calibration.json"))
    pol = json.load(open(OUT / f"{fam}_policies.json"))      # {name: monitor or "fixed"/"never"}
    jobs = [(c, s, name) for c in R.CONDS for s in seeds for name in pol]
    for i, (c, s, name) in enumerate(jobs):
        if i % nw != wid:
            continue
        f = OUT / f"{fam}_{c}_s{s}_{name}.json"
        if f.exists():
            continue
        hist = {"r0": 0, "v": []}
        if pol[name] == "never":
            p = None
        elif pol[name] == "fixed":
            p = fixed_policy(chosen["fixed"]["I"], hist)
        else:
            p = make_policy(pol[name][0], tuple(pol[name][1]), hist)
        cfg = dict(depth=2, wd=0.0, steps=R.STEPS, batch=R.BATCH, noise_levels=R.FAMILIES[fam], **R.CONDS[c])
        t0 = time.perf_counter()
        r = P.run_stream(cfg, s, R.T, policy=p, intervention="reset")
        util = float(np.mean([t["online_acc"] for t in r["tasks"]]))
        json.dump({"fam": fam, "cond": c, "seed": s, "policy": name, "monitor": pol[name], "util": util,
                   "resets": r["n_interventions"], "seconds": time.perf_counter() - t0,
                   "acc": [t["online_acc"] for t in r["tasks"]]}, open(f, "w"))
        print(f"{fam} {c} s{s} {name}: util {util:.4f} resets {r['n_interventions']} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    print("worker done", flush=True)

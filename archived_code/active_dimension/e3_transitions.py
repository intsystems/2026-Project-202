"""E3 -- a controlled simplification, and whether MG sees it.

``r_high -> r_low -> r_high``, three equal segments, everything else held fixed: same data,
same backbone, same 10 available directions, same drive amplitude per active mode, same
learning rate.  Only the number of excited directions changes.

Two things are measured, and they are different questions:

**level error**   the median calibrated estimate over the windows lying *entirely* inside a
                  segment, against the active dimension measured on that same segment.
                  Windows straddling a switch are excluded, because a window that spans the
                  boundary has no single true answer.  (``exp10`` labelled every window by
                  the ground truth at its right edge, so straddling windows were scored
                  against a value that held for only part of their data.)

**detection lag** the first window whose right edge is after the switch, whose estimate has
                  crossed a threshold set from the **pre-switch segment alone** (its median
                  minus Z times its window-to-window scatter), and which stays across for Q
                  consecutive windows.  A window of length W cannot respond before its right
                  edge is W past the switch, so W is the irreducible floor and the lag
                  resolution is one stride.  The detection *rate* is reported beside the
                  median lag: a median over detections only would flatter the arm that
                  mostly fails to detect anything.

Run in the arm where MG works (``qp``) and in the arm where E2 says it does not (``noise``),
because a detector that fires identically in both is not detecting a change of dimension.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from dynamics import F_FAST, Spec, equalise_gains, simulate
from e2_rank_sweep import SWEEP_OBS as OBSERVERS
from e1_calibration import load_frozen
from runner import get_system, spec_columns
from system import rank_pr

OUT = Path(__file__).resolve().parent / "results" / "e3_transitions"
OUT.mkdir(parents=True, exist_ok=True)

SEG = 15_000
BURN = 4_000
LEVELS = ((6, 2), (8, 3), (4, 1))
SEEDS = (0, 1, 2, 3)
HOLD_Q = 2
Z = 3.0            # detection threshold, in pre-switch window-scatter units


def schedule(hi, lo, T, burn):
    s = np.full(T, hi, int)
    a, b = burn + SEG, burn + 2 * SEG
    s[a:b] = lo
    return s, a, b


def job(mode, hi, lo, seed, cfg):
    A, c_star = get_system(seed, 10)
    T = BURN + 3 * SEG
    sched, t1, t2 = schedule(hi, lo, T, BURN)
    kw = dict(seed=seed, k=10, r=hi, T=T, burn=BURN, mode=mode, precondition=True,
              r_schedule=sched, eta=0.15, f0=F_FAST)
    kw.update(drive_amp=0.8, noise_amp=0.0) if mode == "qp" else kw.update(
        drive_amp=0.0, noise_amp=0.08)
    spec = Spec(**kw)
    mix, cond = equalise_gains(A, spec, c_star)
    logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)

    # ground truth per segment, measured on the trajectory itself
    b = BURN
    segs = [(0, SEG, hi), (SEG, 2 * SEG, lo), (2 * SEG, 3 * SEG, hi)]
    truth = {i: rank_pr(C[s:e])[1] for i, (s, e, _) in enumerate(segs)}

    rows = []
    for o in OBSERVERS:
        x = logs[o]
        if x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()
        right, tr = MG.sliding(z, cfg)
        v = tr["MG"]
        ok = ~(tr["degenerate"] > 0.5) & np.isfinite(v)
        left = right - cfg.window + 1
        lev, lag = {}, {}
        for i, (s, e, _) in enumerate(segs):
            inside = ok & (left >= s) & (right < e)
            lev[i] = float(np.median(v[inside])) if inside.any() else np.nan
        # Detection uses ONLY the pre-switch segment: threshold = pre-switch median -/+ Z
        # times the pre-switch window-to-window scatter.  Taking the midpoint of the two
        # measured levels, as a first version did, hands the detector the answer -- and in
        # the `noise` arm, where the two levels coincide, it makes the crossing a coin flip,
        # so the arm that cannot detect anything reports the SHORTEST lag.
        for j, (t_sw, i_from, i_to) in enumerate([(SEG, 0, 1), (2 * SEG, 1, 2)]):
            pre = ok & (left >= segs[i_from][0]) & (right < segs[i_from][1])
            if pre.sum() < 3:
                lag[j] = np.nan; continue
            mu, sd = float(np.median(v[pre])), float(np.std(v[pre]))
            down = (i_to == 1)
            thr = mu - Z * sd if down else mu + Z * sd
            cross = (v < thr) if down else (v > thr)
            hit = np.nan
            for idx in np.where(ok & cross & (right >= t_sw))[0]:
                w = slice(idx, idx + HOLD_Q)
                if idx + HOLD_Q <= len(v) and ok[w].any() and bool(np.all(cross[w][ok[w]])):
                    hit = float(right[idx] - t_sw); break
            lag[j] = hit
        rows.append(dict(observer=o, mode=mode, hi=hi, lo=lo, seed=seed,
                         level0=lev[0], level1=lev[1], level2=lev[2],
                         truth0=truth[0], truth1=truth[1], truth2=truth[2],
                         lag_down=lag[0], lag_up=lag[1], window=cfg.window,
                         detected_down=bool(np.isfinite(lag[0])),
                         detected_up=bool(np.isfinite(lag[1])),
                         drive_cond=cond, n_windows=int(ok.sum()),
                         **{k: val for k, val in spec_columns(spec).items()
                            if k in ("eta", "drive_amp", "noise_amp", "precondition")}))
    return rows


def main():
    cfg, _ = load_frozen()
    t0 = time.time()
    jobs = [(m, hi, lo, s, cfg) for m in ("qp", "noise")
            for hi, lo in LEVELS for s in SEEDS]
    print(f"{len(jobs)} transition runs, config {cfg}", flush=True)
    res = Parallel(n_jobs=12, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "transitions_raw.csv", index=False)

    df["drop_seen"] = df.level0 - df.level1
    df["recover_seen"] = df.level2 - df.level1
    print(f"\ndone in {time.time()-t0:.0f}s")
    for m in ("qp", "noise"):
        s = df[df["mode"] == m]
        print(f"\n=== {m}: MG level per segment (median over seeds) ===")
        print(s.groupby(["hi", "lo", "observer"])[
            ["truth0", "truth1", "truth2", "level0", "level1", "level2",
             "drop_seen", "lag_down", "lag_up"]].median().round(2).to_string())
        print("--- detection RATE: a median lag over detections only flatters the arm that "
              "mostly fails, so the rate is reported beside it ---")
        print(s.groupby("observer")[["detected_down", "detected_up"]].mean()
              .round(2).to_string())


if __name__ == "__main__":
    main()

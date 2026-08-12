"""E10 -- is the matched-window fall of E9 the trajectory, or the shape of the observer?

E9 finds that at a window matched to the transition the estimate falls at the generalisation
step in every generalising run and in neither control, and that the fall does not survive
removing a linear trend from the window.  That second fact admits two readings: the fall is
carried by the observer's own shape (the parameter norm's decline steepens at the transition,
and ``sec:nuisance`` shows a smooth gain change alone moves the estimate), or detrending a
scalar removes so much of it that nothing could survive.  Detrending a 1024-dimensional
sketch per coordinate, which is what ``PR^det`` does, removes one direction of many; on a
one-dimensional log it removes most of the series.

The test that separates the two readings keeps the shape and destroys everything else.  For
each run and observer:

    S     = the log smoothed over ``SMOOTH`` samples (Savitzky-Golay, cubic), which keeps the
            large-scale shape, including whatever the norm does at the transition
    r     = x - S, the fluctuation
    x*    = S + IAAFT-surrogate(r), a series with the same shape, the same fluctuation power
            spectrum and the same marginal, and no nonlinear temporal structure

``x*`` is then put through E9's pipeline unchanged and the same depth statistic computed at
the same ``t_gen``.  If the surrogates fall as deeply, the fall is the shape and carries no
information about the trajectory; if they do not, the fall needs structure a smooth shape and
a matched spectrum cannot supply.  The smoothing length is swept, because it is the one free
parameter and a fall that only survives one choice of it has not survived.

Two defects in the first version of this file are fixed here, and both mattered.

*Determinism.*  The surrogate rng was seeded with ``abs(hash((run, col, smooth)))``.  Python
salts ``str.__hash__`` with ``PYTHONHASHSEED`` at interpreter start, so that expression takes
a different value in every process: the committed ``surrogates.csv`` was produced under a seed
that cannot be recovered, and re-running the file reproduced only eight of its eighteen
parameter-norm p-values.  Seeding now goes through :func:`cell_rng`, a ``zlib.crc32`` of the
cell's own strings folded together with an explicit ``--seed`` base, which is stable across
processes, platforms and Python versions.  The whole experiment can be re-seeded as a unit,
and it is: ``--seeds`` base seeds are run and the across-seed spread of every p-value is
reported, because a p-value resolved to 1/40 from a single draw is not a number to quote.

*Alignment.*  ``depth`` was called as ``depth(t, ms, tgen)`` -- the window grid ``t`` of the
observed trace against the values ``ms`` of a surrogate one.  ``trace`` drops any window whose
values are constant, so a surrogate that produced one constant window returned a shorter
array: usually a crash (three of four fresh invocations), and silently a mis-aligned statistic
whenever a different window happened to be dropped and the lengths still matched.  Each
surrogate is now evaluated on the grid it actually has, ``depth`` refuses a grid and a value
array of different lengths, and ``grid_match`` records per row whether the surrogate's grid
equalled the observed one so a dropped window is visible in the output rather than hidden.

    python e10_surrogate.py                 # ~12 min on 12 cores, 5 base seeds
    python e10_surrogate.py --seeds 1       # the single-seed run, for a quick check
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

import mg as MG
from e1_calibration import load_frozen

HERE = Path(__file__).resolve().parent
FINE = HERE.parent / "active_rank" / "results_fine"
OUT = HERE / "results" / "e9_matched_window"

WINDOW = 60
CELL = dict(max_E=10, tau=1, k_neighbors=20, theiler="autocorr")   # E9's headline rule
COLUMNS = ("weight_norm", "train_loss")
SMOOTH = (101, 201, 401)        # samples; 1010, 2010, 4010 optimiser steps
N_SURR = 39                     # 39 + the observed gives a 1/40 = 0.025 resolution
GEN = {"mod_wd1": 13700.0, "mod_wd1_s43": 6660.0, "mod_wd1_s44": 3680.0, "s5_wd1": 6735.0}
CONTROL_OF = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
PRE, POST = (-3000, -1000), (-1000, 2000)


def cell_rng(base, run, col, smooth):
    """A generator that depends only on its arguments, not on the process it runs in.

    ``zlib.crc32`` is a fixed function of the bytes; ``hash`` on a str is not.  The four
    components go in as separate entropy words so no two cells can collide by concatenation.
    """
    key = f"{run}|{col}|{smooth}".encode("utf-8")
    return np.random.default_rng([int(base), zlib.crc32(key),
                                  zlib.crc32(run.encode()), int(smooth)])


def iaaft(x, rng, n_iter=100):
    """Amplitude-adjusted Fourier transform surrogate: same spectrum, same marginal."""
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    srt = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        f = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(f)), n)
        y = srt[np.argsort(np.argsort(y))]
    return y


def trace(x, step, win, cfg):
    """The estimate on the direct measurement's own window midpoints.

    Returns the grid and the values together, always the same length: a caller that keeps only
    the values cannot then pair them with somebody else's grid.
    """
    dstep = int(np.median(np.diff(step)))
    half = WINDOW // 2
    t, m = [], []
    for _, w in win.iterrows():
        mid = (w.right_step + w.left_step) / 2.0
        c = int(round(mid / dstep))
        a, b = c - half, c + WINDOW - half
        if a < 0 or b > len(x):
            continue
        v = x[a:b]
        if not np.isfinite(v).all() or v.std() <= 1e-12:
            continue
        t.append(mid)
        m.append(MG.all_estimators((v - v.mean()) / v.std(), cfg)["MG"])
    return np.array(t, float), np.array(m, float)


def depth(t, y, c):
    """Pre-transition level over post-transition floor, on one trace's own grid."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.shape != y.shape:
        raise ValueError(f"grid {t.shape} and values {y.shape} disagree: a trace has been "
                         f"paired with another trace's window grid")
    pre = (t >= c + PRE[0]) & (t <= c + PRE[1])
    post = (t >= c + POST[0]) & (t <= c + POST[1])
    if pre.sum() < 3 or post.sum() < 3:
        return np.nan
    b, f = np.nanmedian(y[pre]), np.nanmin(y[post])
    return b / f if np.isfinite(b) and np.isfinite(f) and f > 0 else np.nan


def job(run, col, smooth, seed, base):
    df = pd.read_csv(FINE / f"{run}_train.csv")
    if col not in df.columns:
        return []
    step, x = df["step"].to_numpy(), df[col].to_numpy(float)
    win = pd.read_csv(FINE / "rank_windows.csv")
    win = win[win.run == run].sort_values("right_step")
    tgen = GEN.get(run, GEN.get(CONTROL_OF.get(run, ""), np.nan))
    cfg = dataclasses.replace(base, window=WINDOW, stride=1, **CELL)

    s = savgol_filter(x, min(smooth, len(x) - (1 - len(x) % 2)), 3)
    r = x - s
    t, m = trace(x, step, win, cfg)
    rng = cell_rng(seed, run, col, smooth)
    out = [dict(run=run, column=col, smooth=smooth, seed=int(seed), kind="observed", i=-1,
                depth=depth(t, m, tgen), n_windows=len(t), grid_match=True)]
    for i in range(N_SURR):
        ts, ms = trace(s + iaaft(r, rng), step, win, cfg)
        out.append(dict(run=run, column=col, smooth=smooth, seed=int(seed), kind="surrogate",
                        i=i, depth=depth(ts, ms, tgen), n_windows=len(ts),
                        grid_match=bool(len(ts) == len(t) and np.array_equal(ts, t))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0, help="base seed of the first replicate")
    ap.add_argument("--seeds", type=int, default=5,
                    help="how many base seeds to run; the spread across them is reported "
                         "because a single 1/40-resolution p-value is not quotable")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--columns", nargs="+", default=list(COLUMNS))
    ap.add_argument("--smooth", type=int, nargs="+", default=list(SMOOTH))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    base, _ = load_frozen()
    runs = sorted(p.stem.replace("_train", "") for p in FINE.glob("*_train.csv"))
    seeds = [a.seed + i for i in range(a.seeds)]
    jobs = [(r, c, sm, sd, base)
            for r in runs for c in a.columns for sm in a.smooth for sd in seeds]
    print(f"{len(jobs)} jobs x {N_SURR + 1} series  "
          f"({len(runs)} runs, {len(a.columns)} observers, {len(a.smooth)} smoothings, "
          f"{len(seeds)} seeds {seeds})", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(a.out / "surrogates.csv", index=False)
    bad = int((~df.grid_match).sum())
    print(f"\n{len(df)} rows in {time.time() - t0:.0f}s; "
          f"{bad} surrogate(s) on a grid other than the observed one")

    # ---- p per (run, observer, smoothing, seed) ----------------------------------
    rows = []
    for (run, col, sm, sd), g in df.groupby(["run", "column", "smooth", "seed"]):
        o = g[g.kind == "observed"].depth.iloc[0]
        sv = g[g.kind == "surrogate"].depth.to_numpy(float)
        sv = sv[np.isfinite(sv)]
        rows.append(dict(run=run, column=col, smooth=sm, seed=sd, observed=o,
                         surr_median=np.median(sv) if len(sv) else np.nan,
                         surr_max=np.max(sv) if len(sv) else np.nan,
                         p=(1 + (sv >= o).sum()) / (1 + len(sv))
                         if len(sv) and np.isfinite(o) else np.nan,
                         n=len(sv), generalises=run in GEN))
    t = pd.DataFrame(rows).sort_values(["column", "run", "smooth", "seed"])
    t.to_csv(a.out / "surrogate_summary.csv", index=False)

    # ---- the across-seed spread, which is what may be quoted ---------------------
    sp = (t.groupby(["run", "column", "smooth"])
           .agg(generalises=("generalises", "first"), observed=("observed", "first"),
                p_min=("p", "min"), p_median=("p", "median"), p_max=("p", "max"),
                p_sd=("p", "std"), n_seeds=("p", "size")).reset_index())
    sp.to_csv(a.out / "surrogate_seed_spread.csv", index=False)

    for col in a.columns:
        u = sp[sp.column == col]
        print(f"\n=== {col}: p across {len(seeds)} base seeds ===")
        print(u[["run", "smooth", "generalises", "observed", "p_min", "p_median",
                 "p_max", "p_sd"]].round(4).to_string(index=False))
        g, c = u[u.generalises], u[~u.generalises]
        # the two numbers the claim rests on
        gmax = t[(t.column == col) & t.generalises].p.max()
        cmin = t[(t.column == col) & ~t.generalises].p.min()
        print(f"  worst generalising cell over all seeds: p = {gmax:.3f}"
              f"   (cell maxima: max over cells of p_max = {g.p_max.max():.3f})")
        print(f"  best control cell over all seeds:       p = {cmin:.3f}"
              f"   (cell minima: min over cells of p_min = {c.p_min.min():.3f})")
        print(f"  separated at every seed: {bool(gmax < cmin)}")


if __name__ == "__main__":
    main()

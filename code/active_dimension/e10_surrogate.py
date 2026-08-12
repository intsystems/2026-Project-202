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

    python e10_surrogate.py        # ~6 min on 12 cores
"""

from __future__ import annotations

import dataclasses
import time
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
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 60
CELL = dict(max_E=10, tau=1, k_neighbors=20, theiler="autocorr")   # E9's headline rule
COLUMNS = ("weight_norm", "train_loss")
SMOOTH = (101, 201, 401)        # samples; 1010, 2010, 4010 optimiser steps
N_SURR = 39                     # 39 + the observed gives a 1/40 = 0.025 resolution
GEN = {"mod_wd1": 13700.0, "mod_wd1_s43": 6660.0, "mod_wd1_s44": 3680.0, "s5_wd1": 6735.0}
CONTROL_OF = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
PRE, POST = (-3000, -1000), (-1000, 2000)


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
    """The estimate on the direct measurement's own window midpoints."""
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
    pre = (t >= c + PRE[0]) & (t <= c + PRE[1])
    post = (t >= c + POST[0]) & (t <= c + POST[1])
    if pre.sum() < 3 or post.sum() < 3:
        return np.nan
    b, f = np.nanmedian(y[pre]), np.nanmin(y[post])
    return b / f if np.isfinite(b) and np.isfinite(f) and f > 0 else np.nan


def job(run, col, smooth, base):
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
    obs = depth(t, m, tgen)
    rng = np.random.default_rng(abs(hash((run, col, smooth))) % (2 ** 31))
    out = [dict(run=run, column=col, smooth=smooth, kind="observed", i=-1, depth=obs)]
    for i in range(N_SURR):
        _, ms = trace(s + iaaft(r, rng), step, win, cfg)
        out.append(dict(run=run, column=col, smooth=smooth, kind="surrogate", i=i,
                        depth=depth(t, ms, tgen)))
    return out


def main():
    base, _ = load_frozen()
    runs = sorted(p.stem.replace("_train", "") for p in FINE.glob("*_train.csv"))
    jobs = [(r, c, s, base) for r in runs for c in COLUMNS for s in SMOOTH]
    print(f"{len(jobs)} jobs x {N_SURR + 1} series", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=12, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "surrogates.csv", index=False)

    rows = []
    for (run, col, sm), g in df.groupby(["run", "column", "smooth"]):
        o = g[g.kind == "observed"].depth.iloc[0]
        s = g[g.kind == "surrogate"].depth.to_numpy(float)
        s = s[np.isfinite(s)]
        rows.append(dict(run=run, column=col, smooth=sm, observed=o,
                         surr_median=np.median(s) if len(s) else np.nan,
                         surr_max=np.max(s) if len(s) else np.nan,
                         p=(1 + (s >= o).sum()) / (1 + len(s)) if len(s) and np.isfinite(o)
                         else np.nan, n=len(s)))
    t = pd.DataFrame(rows).sort_values(["column", "smooth", "run"])
    t.to_csv(OUT / "surrogate_summary.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

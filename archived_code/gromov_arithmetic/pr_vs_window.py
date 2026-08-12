"""Is the full-batch participation ratio of 1.0 a fact about the run or about the window?

``../active_rank/analyze_rank.py`` measures every full-batch window of every run at a
participation ratio between 1.000 and 1.362, against 1.09-7.0 in the mini-batch setting.
The report reads that as resolution-limited rather than negative: a deterministic update
makes the trajectory a smooth curve, and over 600 optimiser steps a smooth curve is
straight to within the precision of the statistic.  If that reading is right the number
is reporting the window length, and lengthening the window has to move it.

This file is that test.  It computes the same participation ratio over windows spanning
600 to 120 000 steps and reports it against window length.

**Two ladders, because "longer window" is two different changes.**  A window can be
lengthened by taking more samples of the same trajectory or by spreading the same number
of samples further apart, and the participation ratio responds to both -- to the second
because the curve has more room to bend, but to the first also, because the statistic is
bounded above by the sample count and its noise floor falls as samples are added.  Only
the second isolates the question:

``fixed_n``   60 samples per window, sampled every ``stride`` logged rows.  The sample
              count, and so the bound and the floor, are exactly those of the published
              600-step measurement; only the span changes.  This is the ladder to read.
``fixed_dt``  every logged row, more of them per window.  Reported beside it as the
              control that says how much of any rise is sample count rather than span.

Both are computed in both recorded spaces, parameters and probe logits, and averaged
over the two independent sketches, exactly as ``analyze_rank.py`` does -- ``pr`` and
``detrend`` are imported from it rather than reimplemented, so a difference in the
numbers cannot come from a difference in the estimator.

    python pr_vs_window.py --indir ./results/rank_fb_long --outdir ./results/rank_fb_long
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_AR = HERE.parent / "active_rank"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))

from analyze_rank import detrend, milestones, pr as pr_svd  # noqa: E402


def pr(X, tol=1e-12):
    """``analyze_rank.pr`` without the singular value decomposition.

    The published windows are 60 rows wide and a full SVD of one is free.  These are up
    to 2 400 rows of a 1 024-dimensional sketch and there are thousands of them, which
    the SVD is too slow for.  It is also unnecessary: with ``lambda`` the eigenvalues of
    the window's covariance, ``sum lambda`` is the squared Frobenius norm of the centred
    window and ``sum lambda^2`` is the squared Frobenius norm of its Gram matrix, so the
    participation ratio is available from two matrix norms and no decomposition.  The
    Gram matrix is taken on whichever side is smaller.

    This is an identity, not an approximation, and ``--check`` verifies it against the
    imported estimator on real windows rather than asserting it.
    """
    X = np.asarray(X, float)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return np.nan
    total = float((X * X).sum())
    if not total > tol:
        return np.nan
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    return total * total / float((G * G).sum())

# 60 samples is what the published measurement used; the strides turn it into the
# window lengths below at log_every = 50.  A stride is only run if the record is long
# enough to hold at least MIN_WINDOWS of the resulting span.
STRIDES = (1, 2, 4, 10, 20, 40)
ROWS = (60, 120, 240, 600, 1200, 2400)
N_SAMPLES = 60
MIN_WINDOWS = 3


def window_pr(arr, a, n, stride):
    """The three position/increment participation ratios of one window, per sketch.

    ``arr`` is (T, n_sketch, dim); the window is ``n`` samples starting at row ``a``
    taken every ``stride`` rows, so it spans ``stride * (n - 1) + 1`` rows of the record.
    """
    out = {"pos": [], "pos_det": [], "step": []}
    for s in range(arr.shape[1]):
        W = arr[a:a + stride * n:stride, s, :]
        out["pos"].append(pr(W))
        out["pos_det"].append(pr(detrend(W)))
        out["step"].append(pr(np.diff(W, axis=0)))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def sweep(npz, log_every, ladder):
    """One row per (window length, window position) for one ladder of window lengths."""
    step, z, zf = npz["step"], npz["z"], npz["zf"]
    move = npz["param_step"]
    T = len(step)
    rows = []
    for stride, n in ladder:
        span = stride * (n - 1) + 1                    # rows the window occupies
        if T < span + MIN_WINDOWS:
            continue
        # Enough positions to see the spread of the statistic, without recomputing a
        # 2400-row Gram matrix hundreds of times.  A long window costs quadratically
        # more and there is less to see -- consecutive placements of a window covering
        # a fifth of the run overlap almost completely -- so it is sampled less densely.
        target = 40 if span <= 240 else 12
        adv = max(1, (T - span) // target)
        for a in range(0, T - span + 1, adv):
            b = a + span
            rec = {"n_samples": n, "row_stride": stride,
                   "window_steps": int(stride * n * log_every),
                   "left_step": int(step[a]), "right_step": int(step[b - 1]),
                   "centre_step": int(0.5 * (step[a] + step[b - 1]))}
            for name, arr in (("", z), ("fn_", zf)):
                for k, v in window_pr(arr, a, n, stride).items():
                    rec[f"{name}PR_{k}"] = v
            rec["move"] = float(np.nansum(move[a:b]))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise(df, t_gen):
    """Median / range over window positions, and the window straddling generalisation."""
    out = []
    for (n, stride, w), g in df.groupby(["n_samples", "row_stride", "window_steps"]):
        rec = {"n_samples": n, "row_stride": stride, "window_steps": w,
               "n_windows": len(g)}
        for c in ("PR_pos_det", "PR_step", "fn_PR_pos_det", "fn_PR_step"):
            rec[f"{c}_med"] = float(g[c].median())
            rec[f"{c}_min"] = float(g[c].min())
            rec[f"{c}_max"] = float(g[c].max())
        if t_gen is not None:
            k = (g.centre_step - t_gen).abs().idxmin()
            rec["at_gen_centre"] = int(g.centre_step[k])
            for c in ("PR_pos_det", "PR_step", "fn_PR_pos_det", "fn_PR_step"):
                rec[f"{c}_at_gen"] = float(g[c][k])
        out.append(rec)
    return pd.DataFrame(out).sort_values(["n_samples", "window_steps"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default=str(HERE / "results" / "rank_fb_long"))
    p.add_argument("--outdir", default=None)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--check", action="store_true",
                   help="verify the Gram-matrix estimator against analyze_rank's SVD one")
    a = p.parse_args(argv)

    indir = Path(a.indir)
    outdir = Path(a.outdir) if a.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    frames, summaries = [], []
    for npz_path in sorted(indir.glob("*_rank.npz")):
        tag = npz_path.stem.replace("_rank", "")
        csv = indir / f"{tag}_train.csv"
        if not csv.exists():
            print(f"[{tag}] no {csv.name}, skipping")
            continue
        npz, log = np.load(npz_path, allow_pickle=True), pd.read_csv(csv)
        t_mem, t_gen = milestones(log)
        step = npz["step"]
        log_every = int(step[1] - step[0])

        if a.check:
            for n, stride in ((60, 1), (60, 20), (600, 1)):
                W = npz["z"][0:stride * n:stride, 0, :]
                for label, M in (("pos", W), ("pos_det", detrend(W))):
                    fast, ref = pr(M), pr_svd(M)
                    print(f"  check n={n:>4} stride={stride:>2} {label:<8} "
                          f"gram={fast:.9f} svd={ref:.9f} "
                          f"rel={abs(fast - ref) / ref:.2e}")

        ladders = {"fixed_n": [(s, a.n_samples) for s in STRIDES],
                   "fixed_dt": [(1, n) for n in ROWS]}
        for name, ladder in ladders.items():
            df = sweep(npz, log_every, ladder)
            if df.empty:
                continue
            df.insert(0, "run", tag)
            df.insert(1, "ladder", name)
            frames.append(df)
            s = summarise(df, t_gen)
            s.insert(0, "run", tag)
            s.insert(1, "ladder", name)
            s["t_mem"], s["t_gen"] = t_mem, t_gen
            summaries.append(s)
        print(f"[{tag}] rows={len(step)} log_every={log_every} "
              f"t_mem={t_mem} t_gen={t_gen}", flush=True)

    if not frames:
        raise SystemExit(f"no usable *_rank.npz under {indir}")

    windows = pd.concat(frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    windows.to_csv(outdir / "pr_vs_window_windows.csv", index=False)
    summary.to_csv(outdir / "pr_vs_window.csv", index=False)

    cols = ["window_steps", "n_windows", "PR_pos_det_med", "PR_pos_det_max",
            "PR_step_med", "fn_PR_pos_det_med", "fn_PR_step_med"]
    if "PR_pos_det_at_gen" in summary:
        cols += ["PR_pos_det_at_gen", "fn_PR_pos_det_at_gen"]
    for (run, ladder), g in summary.groupby(["run", "ladder"]):
        print(f"\n=== {run} / {ladder}  (t_gen={g.t_gen.iloc[0]}) ===")
        print(g[cols].round(3).to_string(index=False))
    print(f"\n-> {outdir / 'pr_vs_window.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())

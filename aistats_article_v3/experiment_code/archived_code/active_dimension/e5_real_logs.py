"""E5 -- where on the atlas do this project's own training logs actually sit?

E0 measures what MG does for three classes of r-dimensional system.  E2-E4 measure what it
does on a real network whose active rank is known.  This file asks the only question that
connects them to the original motivation: **which regime is a real grokking log in?**

For each of the seven 120 000-step reruns in ``dimension_recovery/results/extended`` and each
1-D column, it reports the quantities the atlas is indexed by:

``ident_ratio``   E(2 max_E) / E(max_E).  Near 1 the estimate is a property of the data; near
                  2 it is a property of the embedding and no dimension is identifiable.
``roughness``     std(diff x)/std(x): the per-step innovation as a fraction of the window's
                  spread.  In the atlas this, not r, is what sets MG for every stochastic
                  family.
``acorr``         the lag at which the ACF falls to 1/e.
``oscillations``  mean-crossings of the linearly detrended window.  A delay embedding can
                  only see a torus the trajectory has gone round; zero crossings means the
                  window contains no recurrence at all.
``PRdelay``       the linear participation ratio, which the exp10-12 audit found recovers k
                  *better* than MG on identical synthetic data.

Nothing here is a new claim about grokking.  It is a placement of the existing logs on a
map built from systems whose answer is known.

    python e5_real_logs.py       # ~10 min on 12 cores
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from e1_calibration import load_frozen

ROOT = Path(__file__).resolve().parent.parent / "dimension_recovery" / "results" / "extended"
OUT = Path(__file__).resolve().parent / "results" / "e5_real_logs"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = ("weight_norm", "train_loss", "val_loss", "train_acc", "val_acc")


def oscillations(x):
    """Mean-crossings of the linearly detrended window: how many times it comes back."""
    t = np.arange(len(x), dtype=float)
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    return int(np.count_nonzero(np.diff(np.signbit(d))))


def job(path, col, cfg):
    df = pd.read_csv(path)
    if col not in df.columns:
        return []
    x = df[col].to_numpy(float)
    step = df["step"].to_numpy()
    dstep = int(np.median(np.diff(step)))
    # ---------------------------------------------------------------- FROZEN CONFIG OVERRIDE
    # This departs from results/e1_calibration/frozen_config.json in two fields.  Anything
    # computed below is NOT at the frozen configuration and must not be quoted as if it were.
    #
    #   window : frozen 8000  ->  min(8000, max(2000, len(x)//3))
    #   stride : frozen 2000  ->  1000   (unconditional)
    #
    # Why: the logs are 12 000 samples (120 000 steps at a logging stride of 10), so the
    # frozen 8 000-sample window leaves three positions.  Shortened to a third of the record
    # for a usable number of windows -- realised value here is 4000 samples = 39 990 steps,
    # which is the span icomp_v2/make_figures.py hard-codes as SPAN.  max_E, tau, k_neighbors,
    # theiler and dither are untouched, so the estimator itself is the frozen one; only the
    # window geometry moves.  cfg2 is derived AFTER the resize so the identifiability ratio
    # compares two embedding dimensions on the *same* window rather than on two different ones.
    cfg = dataclasses.replace(cfg, window=min(cfg.window, max(2000, len(x) // 3)),
                              stride=1000)
    cfg2 = dataclasses.replace(cfg, max_E=2 * cfg.max_E)
    out = []
    for s in MG.window_starts(len(x), cfg):
        w = x[s:s + cfg.window]
        if not np.isfinite(w).all() or w.std() <= 1e-12:
            continue
        z = (w - w.mean()) / w.std()
        e1, e2 = MG.all_estimators(z, cfg), MG.all_estimators(z, cfg2)
        out.append(dict(run=path.stem.replace("_train", ""), column=col,
                        right_step=int(step[s + cfg.window - 1]), log_stride=dstep,
                        MG=e1["MG"], MG_2E=e2["MG"],
                        ident_ratio=(e2["MG"] / e1["MG"]) if e1["MG"] else np.nan,
                        LB=e1["LB"], TwoNN=e1["TwoNN"], PRdelay=e1["PRdelay"],
                        roughness=e1["roughness"], acorr=e1["acorr"],
                        oscillations=oscillations(w), degenerate=bool(e1["degenerate"])))
    return out


def main():
    cfg, _ = load_frozen()
    paths = sorted(ROOT.glob("*_train.csv"))
    if not paths:
        raise SystemExit(f"no logs under {ROOT}")
    t0 = time.time()
    jobs = [(p, c, cfg) for p in paths for c in COLUMNS]
    print(f"{len(paths)} runs x {len(COLUMNS)} columns, config {cfg}", flush=True)
    res = Parallel(n_jobs=12, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "real_logs_windows.csv", index=False)

    g = (df.groupby(["run", "column"])
           .agg(MG=("MG", "median"), MG_2E=("MG_2E", "median"),
                ident=("ident_ratio", "median"), PRdelay=("PRdelay", "median"),
                roughness=("roughness", "median"), acorr=("acorr", "median"),
                osc=("oscillations", "median"), degen=("degenerate", "mean"),
                n=("MG", "size")).reset_index())
    g.to_csv(OUT / "real_logs_summary.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")
    print("\n=== per run x column (median over windows) ===")
    print(g.round(3).to_string(index=False))
    print("\n=== reference values from the E0 atlas ===")
    print("  identifiable (deterministic torus, r resolvable) : ident ~ 1.00-1.03")
    print("  torus, r above the resolvable range              : ident ~ 1.19")
    print("  band-limited stochastic (no r exists)            : ident ~ 1.20-1.34")
    print("  white-driven OU (no r exists)                    : ident ~ 1.44")


if __name__ == "__main__":
    main()

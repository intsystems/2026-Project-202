"""E9 -- the log estimate at a window matched to the transition, on the direct measurement's grid.

``app:window`` reports a resolution limit, not a result: the frozen configuration's window
spans 39 990 optimiser steps, the collapse of ``sec:direct`` lasts one to two thousand, and
nothing localised to the transition could appear.  The conclusion names the fix -- "re-running
the log estimate at a window matched to the transition" -- and this is that re-run.

Design, fixed before looking at any output:

* **The grid is the direct measurement's own.**  Every window here is centred on the midpoint
  of a window in ``active_rank/results_fine/rank_windows.csv``, so ``MG(t)`` and
  ``PR^det(t)`` are two statistics of the same run at the same instants and "do they fall
  together" is a question about paired samples rather than about two plots.
* **One window for every run**, ``WINDOW`` samples = 600 optimiser steps at the stride-10
  logging of these runs.  The direct measurement's own windows are 590 steps for the modular
  runs and 295 for S5; matching the midpoint and not the width keeps one configuration across
  the six runs, and the S5 mismatch is a factor of two and is disclosed.
* **The whole grid is reported.**  A window this short cannot carry the frozen ``max_E = 20``
  at ``tau = 4`` -- the delay span alone is 76 of the 60 samples -- so a configuration has to
  be chosen, and choosing one on the outcome is exactly the failure requirement 2 of the
  protocol exists to prevent.  Every cell of ``GRID`` is written out; the headline cell is
  named by a rule that cannot see the answer (largest ``max_E`` whose delay span is at most a
  quarter of the window, at the frozen ``k`` and Theiler rule), and the paper reports how many
  of the others agree with it.
* **The controls run through the identical pipeline**, aligned on the generalisation step of
  their matched run, and the observers include the two that define generalisation
  (``val_loss``, ``val_acc``) only so that their circularity is visible rather than hidden.
* **Every cell is run twice, on the window and on the linearly detrended window.**  The
  parameter norm has a feature of its own at the transition -- its decline steepens there --
  and a smooth series with a steepening decline can move the estimate with no change in the
  trajectory at all, which is the nuisance of ``sec:nuisance``.  ``PR^det`` removes a linear
  trend per coordinate before taking the participation ratio; the detrended arm removes the
  same thing from the log, so the two statistics are made comparable and the fall is asked to
  survive the removal of the shape that could counterfeit it.

    python e9_matched_window.py        # ~4 min on 12 cores
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from e1_calibration import load_frozen

HERE = Path(__file__).resolve().parent
FINE = HERE.parent / "active_rank" / "results_fine"
OUT = HERE / "results" / "e9_matched_window"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 60                       # samples; 600 optimiser steps at the stride-10 logging
COLUMNS = ("weight_norm", "train_loss", "val_loss", "train_acc", "val_acc")
CONTROL_OF = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}

# every axis the estimate depends on, crossed.  max_E * tau <= WINDOW - 20 is the estimator's
# own length gate; cells that fail it return nan and are kept in the output as nan.
GRID = [dict(max_E=E, tau=t, k_neighbors=k, theiler=th)
        for E, t, k, th in itertools.product((4, 6, 10), (1, 2, 4), (5, 20),
                                             ("autocorr", "embedding"))]
DETREND = (False, True)
# the rule, applied without reference to any output: the delay span (max_E - 1) * tau must be
# at most a quarter of the window, and at that constraint max_E is as large as it goes.
HEADLINE = dict(max_E=10, tau=1, k_neighbors=20, theiler="autocorr")


def oscillations(x):
    t = np.arange(len(x), dtype=float)
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    return int(np.count_nonzero(np.diff(np.signbit(d))))


def job(run, col, cell, det, base):
    """One (run, observer, configuration): the MG trace on the direct measurement's grid."""
    df = pd.read_csv(FINE / f"{run}_train.csv")
    if col not in df.columns:
        return []
    step = df["step"].to_numpy()
    x = df[col].to_numpy(float)
    dstep = int(np.median(np.diff(step)))
    win = pd.read_csv(FINE / "rank_windows.csv")
    win = win[win.run == run].sort_values("right_step")

    cfg = dataclasses.replace(base, window=WINDOW, stride=1, **cell)
    cfg2 = dataclasses.replace(cfg, max_E=2 * cfg.max_E)
    half = WINDOW // 2
    out = []
    for _, w in win.iterrows():
        mid = (w.right_step + w.left_step) / 2.0
        c = int(round(mid / dstep))                       # index of the midpoint sample
        a, b = c - half, c + WINDOW - half
        if a < 0 or b > len(x):
            continue
        v = x[a:b]
        if not np.isfinite(v).all() or v.std() <= 1e-12:
            continue
        if det:                                           # the same removal PR^det performs
            u = np.arange(len(v), dtype=float)
            v = v - np.polyval(np.polyfit(u, v, 1), u)
            if v.std() <= 1e-12:
                continue
        z = (v - v.mean()) / v.std()
        e1 = MG.all_estimators(z, cfg)
        e2 = MG.all_estimators(z, cfg2)
        out.append(dict(run=run, column=col, mid_step=mid,
                        max_E=cfg.max_E, tau=cfg.tau, k=cfg.k_neighbors,
                        theiler=cfg.theiler, detrend=bool(det),
                        MG=e1["MG"], MG_2E=e2["MG"],
                        ident_ratio=(e2["MG"] / e1["MG"]) if e1["MG"] else np.nan,
                        PRdelay=e1["PRdelay"], roughness=e1["roughness"],
                        oscillations=oscillations(v),
                        degenerate=bool(e1["degenerate"]),
                        PR_det=w.fn_PR_pos_det, PR_par_det=w.PR_pos_det,
                        pnorm=w.pnorm, t_gen=w.t_gen, t_mem=w.t_mem))
    return out


def main():
    base, _ = load_frozen()
    runs = sorted(p.stem.replace("_train", "") for p in FINE.glob("*_train.csv"))
    jobs = [(r, c, cell, det, base)
            for r in runs for c in COLUMNS for cell in GRID for det in DETREND]
    print(f"{len(runs)} runs x {len(COLUMNS)} columns x {len(GRID)} cells x "
          f"{len(DETREND)} detrends = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=12, verbose=5, batch_size=4)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(5)
    df.to_csv(OUT / "matched_windows.csv.gz", index=False, compression="gzip")
    print(f"\n{len(df)} rows in {time.time() - t0:.0f}s -> {OUT/'matched_windows.csv.gz'}")

    h = df[(df.max_E == HEADLINE["max_E"]) & (df.tau == HEADLINE["tau"])
           & (df.k == HEADLINE["k_neighbors"]) & (df.theiler == HEADLINE["theiler"])
           & (~df.detrend) & (df.column.isin(("weight_norm", "train_loss")))]
    h[["run", "column", "mid_step", "MG", "PRdelay", "roughness", "PR_det", "t_gen"]] \
        .to_csv(OUT / "headline_trace.csv", index=False)


if __name__ == "__main__":
    main()

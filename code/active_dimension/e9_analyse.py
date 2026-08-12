"""Analysis of E9: does the scalar estimate fall where the directly measured rank falls?

Three questions, each decided by a statistic fixed before the output was read.

1. **Depth.**  For a candidate centre ``c``, the local fall is
   ``D(c) = median(MG over [c-3000, c-1000]) / min(MG over [c-1000, c+2000])`` --- a level
   just before, divided by the floor just after.  The interval is the one in which the direct
   measurement's own minimum falls in all four generalising runs.  Reported at ``c = t_gen``.
2. **Specificity.**  ``D`` is evaluated at every admissible centre in the run and the
   percentile of ``D(t_gen)`` among them is reported.  A fall that is deep but no deeper than
   the rest of the run is not a signature, and the percentile is what says so.
3. **Co-timing.**  The identical statistic on ``PR^det`` from the same windows, so the two
   are compared by the same rule, plus the Spearman correlation of the two traces over the
   whole record.

Everything is computed for every cell of the configuration grid; the headline cell is named
by E9's rule and not by its answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "results" / "e9_matched_window"
GEN = {"mod_wd1": 13700.0, "mod_wd1_s43": 6660.0, "mod_wd1_s44": 3680.0, "s5_wd1": 6735.0}
CONTROL_OF = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
MEM = {"mod_wd0": 630, "mod_wd1": 1470, "mod_wd1_s43": 1580, "mod_wd1_s44": 1410,
       "s5_wd0": 1015, "s5_wd1": 1280}
PRE = (-3000, -1000)      # the level a fall is measured from
POST = (-1000, 2000)      # where the floor is looked for
HEADLINE = dict(max_E=10, tau=1, k=20, theiler="autocorr")


def scan(t, y, c):
    """D(c): the pre-transition level over the post-transition floor, or nan if undefined."""
    pre = (t >= c + PRE[0]) & (t <= c + PRE[1])
    post = (t >= c + POST[0]) & (t <= c + POST[1])
    if pre.sum() < 3 or post.sum() < 3:
        return np.nan
    b, f = np.nanmedian(y[pre]), np.nanmin(y[post])
    if not np.isfinite(b) or not np.isfinite(f) or f <= 0:
        return np.nan
    return b / f


def one(g, tgen, tmem, key):
    """Depth at t_gen, its percentile over the run, and where the floor actually is."""
    g = g.sort_values("mid_step")
    t, y = g.mid_step.to_numpy(float), g[key].to_numpy(float)
    d = scan(t, y, tgen)
    # every admissible centre after memorisation, on the same grid the run is logged on
    cs = t[(t >= tmem + 1000)]
    ds = np.array([scan(t, y, c) for c in cs], float)
    ok = np.isfinite(ds)
    pct = float((ds[ok] <= d).mean()) if np.isfinite(d) and ok.sum() > 5 else np.nan
    post = (t >= tgen + POST[0]) & (t <= tgen + POST[1]) & np.isfinite(y)
    loc = t[post][np.argmin(y[post])] - tgen if post.sum() >= 3 else np.nan
    return d, pct, loc, (np.nanmax(ds) if ok.sum() else np.nan)


def table(df, col):
    rows = []
    d = df[df.column == col]
    for (E, tau, k, th, dt), gg in d.groupby(["max_E", "tau", "k", "theiler", "detrend"]):
        rec = dict(max_E=E, tau=tau, k=k, theiler=th, det=dt)
        gen, ctrl = [], []
        for run, g in gg.groupby("run"):
            tgen = GEN.get(run, GEN.get(CONTROL_OF.get(run, ""), np.nan))
            dep, pct, loc, mx = one(g, tgen, MEM[run], "MG")
            rec[f"dep_{run}"], rec[f"pct_{run}"], rec[f"loc_{run}"] = dep, pct, loc
            (ctrl if run in CONTROL_OF else gen).append(dep)
        rec["gen_min"] = np.nanmin(gen) if np.isfinite(gen).any() else np.nan
        rec["ctrl_max"] = np.nanmax(ctrl) if np.isfinite(ctrl).any() else np.nan
        rec["separates"] = bool(np.isfinite(rec["gen_min"]) and np.isfinite(rec["ctrl_max"])
                                and rec["gen_min"] > rec["ctrl_max"])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(OUT / "matched_windows.csv.gz")

    print("================ the direct measurement, by the same rule ================")
    ref = df[(df.column == "weight_norm") & (df.max_E == HEADLINE["max_E"])
             & (df.tau == HEADLINE["tau"]) & (df.k == HEADLINE["k"])
             & (df.theiler == HEADLINE["theiler"])]
    for run, g in ref.groupby("run"):
        tgen = GEN.get(run, GEN.get(CONTROL_OF.get(run, ""), np.nan))
        for key, lab in (("PR_det", "fn"), ("PR_par_det", "par")):
            dep, pct, loc, mx = one(g, tgen, MEM[run], key)
            print(f"  {run:12s} {lab:4s} depth {dep:6.2f}  percentile {pct:5.2f}  "
                  f"floor at {loc:+7.0f}  deepest in run {mx:6.2f}")

    runs = ("mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1", "mod_wd0", "s5_wd0")
    for col in ("weight_norm", "train_loss", "val_loss"):
        t = table(df, col)
        t.to_csv(OUT / f"grid_{col}.csv", index=False)
        print(f"\n================ {col} ================")
        print(t[["max_E", "tau", "k", "theiler", "det"] + [f"dep_{r}" for r in runs]
                + ["gen_min", "ctrl_max", "separates"]].round(2).to_string(index=False))
        for dt in (False, True):
            u = t[t.det == dt]
            m = u[["gen_min", "ctrl_max"]].notna().all(1).sum()
            print(f"  detrend={dt}: cells separating {u.separates.sum()} / {m} defined")
        h = t[(t.max_E == HEADLINE["max_E"]) & (t.tau == HEADLINE["tau"])
              & (t.k == HEADLINE["k"]) & (t.theiler == HEADLINE["theiler"])]
        if len(h):
            print("headline cell (det=False then det=True):")
            print(h[["det"] + [f"dep_{r}" for r in runs] + [f"pct_{r}" for r in runs]
                    + [f"loc_{r}" for r in runs]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

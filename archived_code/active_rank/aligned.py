"""Is anything localised at generalisation, or is it all just training progress?

The three ``mod_wd1`` seeds generalise at 3680, 6660 and 13700.  A statistic genuinely
tied to generalisation survives averaging after alignment on ``t_gen``; one that is a
monotone function of training progress smears out.  The matched ``wd0`` twin -- same task,
same architecture, same schedule, no weight decay and no generalisation -- is plotted on
the same axes as the null: it also slows to a halt, so it shows what the statistic does
when the trajectory stops moving *without* anything being learned.

    python aligned.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
GROK = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
CTRL = {"mod_wd1": "mod_wd0", "mod_wd1_s43": "mod_wd0", "mod_wd1_s44": "mod_wd0",
        "s5_wd1": "s5_wd0"}
STATS = ["fn_PR_step", "fn_PR_step5", "fn_PR_step20", "PR_step20", "PR_pos_det", "move"]


def milestones(log, thresh=0.95):
    out = []
    for col in ("train_acc", "val_acc"):
        hit = log.index[log[col] >= thresh]
        out.append(int(log.step.iloc[hit[0]]) if len(hit) else None)
    return tuple(out)


def main():
    win = pd.read_csv(RES / "rank_windows.csv")
    tg = {}
    for run in win.run.unique():
        tg[run] = milestones(pd.read_csv(RES / f"{run}_train.csv"))[1]

    # --- aligned average over the three mod_wd1 seeds -----------------------------
    grid = np.arange(-6000, 6001, 250)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
    for a, stat in zip(ax.ravel(), STATS):
        curves = []
        for run in ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44"]:
            g = win[win.run == run].sort_values("right_step")
            x = g.right_step.to_numpy() - tg[run]
            y = g[stat].to_numpy()
            a.plot(x, y, lw=.8, alpha=.45, label=f"{run} (t_gen={tg[run]})")
            curves.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
        m = np.nanmean(np.vstack(curves), axis=0)
        a.plot(grid, m, "k-", lw=2.2, label="mean of 3 seeds, aligned")
        c = win[win.run == "mod_wd0"].sort_values("right_step")
        a.plot(c.right_step - 10000, c[stat], "--", c="#d62728", lw=1.4,
               label="mod_wd0 (never groks), step-10000")
        a.axvline(0, color="k", lw=1, ls=":")
        a.set_title(stat, fontsize=10)
        a.set_xlabel("steps since generalisation")
        a.grid(alpha=.25)
        if stat in ("fn_PR_step", "move"):
            a.set_yscale("log")
    ax[0][0].legend(fontsize=6, frameon=False)
    fig.suptitle("Aligned on each run's own t_gen. A feature of generalisation survives "
                 "the alignment; a feature of training progress does not.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "aligned_on_tgen.png", dpi=150)
    plt.close(fig)

    # --- the matched-movement test ------------------------------------------------
    print("=== does the statistic fall at t_gen beyond what the no-grokking twin does? ===")
    print("    windows matched on total displacement, which falls ~10x either way\n")
    for run in GROK:
        g = win[win.run == run]
        c = win[win.run == CTRL[run]]
        lo, hi = c.move.quantile([0.10, 0.90])
        rows = []
        for stat in STATS[:-1]:
            gg = g[(g.move >= lo) & (g.move <= hi)]
            cc = c[(c.move >= lo) & (c.move <= hi)]
            if len(gg) < 3 or len(cc) < 3:
                rows.append((stat, np.nan, np.nan, np.nan)); continue
            rows.append((stat, gg[stat].median(), cc[stat].median(),
                         gg[stat].median() / cc[stat].median()))
        print(f"-- {run} vs {CTRL[run]} (move in [{lo:.2g}, {hi:.2g}], "
              f"n={len(g[(g.move>=lo)&(g.move<=hi)])} vs {len(c[(c.move>=lo)&(c.move<=hi)])}) --")
        for s, a_, b_, r_ in rows:
            print(f"     {s:16s} grok {a_:8.2f}   control {b_:8.2f}   ratio {r_:6.2f}")
        print()
    print(f"-> {FIG/'aligned_on_tgen.png'}")


if __name__ == "__main__":
    raise SystemExit(main())

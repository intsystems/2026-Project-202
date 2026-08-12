"""Locate and size the rank collapse at generalisation.

``aligned.py`` shows a sharp, reproducible dip in the trajectory's participation ratio
around ``t_gen``.  Two things have to be settled before it can be called a result:

**Where is it?**  ``analyze_rank`` labels each window by its right edge, which delays any
feature by up to a whole window.  Here windows are labelled by their **centre**, so the
position of the dip is the position of the event and not an artefact of the labelling.

**Is it just the trajectory slowing down?**  Total displacement per window falls ~10x at
``t_gen`` and stays low.  The discriminating fact is that the rank dip **recovers** while
the displacement does not: a statistic that merely tracked displacement would stay down.
That comparison is quantified here, together with the matched no-weight-decay control,
whose displacement collapses far harder and earlier without generalising at all.

    python dip.py                 # the finely logged runs the paper reports
    python dip.py --results results

The default is ``results_fine``, which is the set the paper's section 7.1 is computed from;
``STATS`` lists the columns it reports, so that re-running this script reproduces the committed
``rank_dip.csv`` rather than a different set of statistics.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

HERE = Path(__file__).resolve().parent
GROK = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
CTRL = ["mod_wd0", "s5_wd0"]
# the run whose t_gen defines the aligned window for each control
CTRL_REF = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
STATS = ["fn_PR_pos_det", "fn_PR_step5", "PR_pos_det", "PR_step5"]


def milestones(log, thresh=0.95):
    out = []
    for col in ("train_acc", "val_acc"):
        hit = log.index[log[col] >= thresh]
        out.append(int(log.step.iloc[hit[0]]) if len(hit) else None)
    return tuple(out)


def load():
    win = pd.read_csv(RES / "rank_windows.csv")
    win["centre"] = (win.left_step + win.right_step) / 2.0
    info = {}
    for run in win.run.unique():
        log = pd.read_csv(RES / f"{run}_train.csv")
        info[run] = milestones(log)
    return win, info


def main():
    global RES, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_fine")
    args = ap.parse_args()
    RES = HERE / args.results
    FIG = HERE / "figures"
    FIG.mkdir(exist_ok=True)

    win, info = load()
    print("run            t_mem   t_gen")
    for r, (m, g) in info.items():
        print(f"{r:14s} {str(m):>6s}  {str(g):>6s}")

    # ---- where is the dip, and how deep, per run --------------------------------
    print("\n=== the dip, per run.  'plateau' is the median over the 4000 steps before")
    print("    t_gen; 'min' is the deepest window centre within +/-4000 of t_gen;")
    print("    'recovered' is the median over 3000..6000 steps after t_gen ===")
    rows = []
    for run in GROK:
        tg = info[run][1]
        g = win[win.run == run].sort_values("centre")
        for s in STATS + ["move"]:
            pre = g[(g.centre >= tg - 4000) & (g.centre < tg)][s]
            near = g[(g.centre >= tg - 4000) & (g.centre <= tg + 4000)]
            post = g[(g.centre >= tg + 3000) & (g.centre <= tg + 6000)][s]
            if not len(near) or not len(pre):
                continue
            i = near[s].idxmin()
            rows.append(dict(run=run, stat=s, plateau=pre.median(),
                             dip=near[s].min(), at=near.centre[i] - tg,
                             recovered=post.median() if len(post) else np.nan,
                             depth=pre.median() / max(near[s].min(), 1e-9)))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "rank_dip.csv", index=False)
    for s in STATS + ["move"]:
        print(f"\n-- {s} --")
        print(tab[tab.stat == s][["run", "plateau", "dip", "at", "recovered", "depth"]]
              .round(2).to_string(index=False))

    # ---- the same statistic on the controls, which never generalise --------------
    print("\n=== controls: deepest dip anywhere in the run, and the end level ===")
    crows = []
    for run in CTRL:
        g = win[win.run == run].sort_values("centre")
        early = g[g.centre < g.centre.quantile(0.2)]
        for s in STATS + ["move"]:
            crows.append(dict(run=run, stat=s, early=early[s].median(),
                              min=g[s].min(), at=g.centre[g[s].idxmin()],
                              end=g[s].iloc[-8:].median(),
                              depth=early[s].median() / max(g[s].min(), 1e-9)))
    ctab = pd.DataFrame(crows)
    ctab.to_csv(RES / "rank_dip_controls.csv", index=False)
    for s in STATS:
        print(f"\n-- {s} --")
        print(ctab[ctab.stat == s][["run", "early", "min", "at", "end", "depth"]]
              .round(2).to_string(index=False))

    # ---- the controls again, in the window the generalising runs define ----------
    # A control has no t_gen, so the depth above is measured over its whole budget and
    # is not comparable with the grokking rows.  This is the like-for-like number.
    print("\n=== controls, aligned on the matched run's t_gen (same window as above) ===")
    arows = []
    for run in CTRL:
        tg = info[CTRL_REF[run]][1]
        g = win[win.run == run].sort_values("centre")
        for s in STATS + ["move"]:
            pre = g[(g.centre >= tg - 4000) & (g.centre < tg)][s]
            near = g[(g.centre >= tg - 4000) & (g.centre <= tg + 4000)][s]
            if not len(pre) or not len(near):
                continue
            arows.append(dict(run=run, stat=s, plateau=pre.median(), dip=near.min(),
                              depth=pre.median() / max(near.min(), 1e-9)))
    atab = pd.DataFrame(arows)
    atab.to_csv(RES / "rank_dip_controls_aligned.csv", index=False)
    print(atab.round(3).to_string(index=False))

    # ---- figure: centre-aligned, all four grokking runs + both controls ----------
    grid = np.arange(-5000, 6001, 200)
    fig, ax = plt.subplots(1, len(STATS) + 1, figsize=(3.05 * (len(STATS) + 1), 3.4),
                           sharex=True)
    for a, s in zip(ax, STATS + ["move"]):
        curves = []
        for run in GROK:
            g = win[win.run == run].sort_values("centre")
            x = g.centre.to_numpy() - info[run][1]
            a.plot(x, g[s], lw=.9, alpha=.5)
            curves.append(np.interp(grid, x, g[s].to_numpy(), left=np.nan, right=np.nan))
        a.plot(grid, np.nanmean(np.vstack(curves), 0), "k-", lw=2.4,
               label="mean of the 4 grokking runs")
        for run, off in zip(CTRL, (10000, 7000)):
            c = win[win.run == run].sort_values("centre")
            a.plot(c.centre - off, c[s], "--", lw=1.2, alpha=.9,
                   label=f"{run} (never groks)")
        a.axvline(0, color="k", lw=1, ls=":")
        a.set_title(s, fontsize=10)
        a.set_xlabel("steps since generalisation")
        a.set_xlim(-5000, 6000)
        a.grid(alpha=.25)
        if s == "move":
            a.set_yscale("log")
    ax[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Participation ratio of the parameter and function trajectories, windows "
                 "labelled by their centre and aligned on each run's own t_gen", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "rank_dip.png", dpi=150)
    plt.close(fig)
    print(f"\n-> {FIG/'rank_dip.png'}")


if __name__ == "__main__":
    raise SystemExit(main())

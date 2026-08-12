"""Align the rank statistics to each run's own milestones, and test the movement confound.

A plain before/after split at ``t_gen`` is not interpretable: "before" mixes the initial
memorisation transient with the plateau, the three seeds grok at 3680 / 6660 / 13700 so the
two sides have very different lengths, and the total displacement per window falls by an
order of magnitude across the transition.  Any participation ratio that moves could
therefore be moving because the trajectory stopped moving.

So three phases per run, from the run's own log:

``early``    step < t_mem                  the memorisation transient
``plateau``  t_mem <= step < t_gen         memorised, not yet generalising
``post``     step >= t_gen                 after generalisation

and the comparison that matters is **plateau vs post**, with ``move`` printed beside every
statistic and a within-run rank correlation against it.

    python phases.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
STATS = ["PR_pos_det", "PR_step", "PR_step5", "PR_step20",
         "fn_PR_pos_det", "fn_PR_step", "fn_PR_step5", "fn_PR_step20"]


def milestones(log, thresh=0.95):
    out = []
    for col in ("train_acc", "val_acc"):
        hit = log.index[log[col] >= thresh]
        out.append(int(log.step.iloc[hit[0]]) if len(hit) else None)
    return tuple(out)


def main():
    win = pd.read_csv(RES / "rank_windows.csv")
    rows, corr = [], []
    for run, g in win.groupby("run"):
        log = pd.read_csv(RES / f"{run}_train.csv")
        t_mem, t_gen = milestones(log)
        # a window belongs to a phase only if it lies ENTIRELY inside it
        def phase(r):
            L, R = r.left_step, r.right_step
            if t_mem is not None and R < t_mem:
                return "early"
            if t_mem is not None and L >= t_mem and (t_gen is None or R < t_gen):
                return "plateau"
            if t_gen is not None and L >= t_gen:
                return "post"
            return "straddle"
        g = g.assign(phase=g.apply(phase, axis=1))
        for ph, gg in g.groupby("phase"):
            if ph == "straddle":
                continue
            rows.append(dict(run=run, phase=ph, n=len(gg), t_mem=t_mem, t_gen=t_gen,
                             move=gg.move.median(),
                             **{s: gg[s].median() for s in STATS}))
        for s in STATS:                       # is the statistic just tracking movement?
            ok = np.isfinite(g[s]) & np.isfinite(g.move)
            if ok.sum() > 5:
                corr.append(dict(run=run, stat=s,
                                 rho_vs_move=float(spearmanr(g[s][ok], g.move[ok]).statistic)))
    tab = pd.DataFrame(rows).sort_values(["run", "phase"])
    tab.to_csv(RES / "rank_phases.csv", index=False)
    print("=== median per phase (a window counts only if it lies entirely inside) ===")
    print(tab.round(2).to_string(index=False))

    print("\n=== plateau -> post, the comparison the simplification story predicts ===")
    piv = tab[tab.phase.isin(("plateau", "post"))]
    for s in STATS + ["move"]:
        p = piv.pivot_table(index="run", columns="phase", values=s)
        if {"plateau", "post"} <= set(p.columns):
            p["ratio"] = p["post"] / p["plateau"]
            print(f"\n-- {s} --")
            print(p.round(2).to_string())

    c = pd.DataFrame(corr).pivot_table(index="stat", columns="run", values="rho_vs_move")
    c.to_csv(RES / "rank_move_confound.csv")
    print("\n=== within-run Spearman(statistic, total displacement in the window) ===")
    print("    a large |rho| means the statistic is reporting how much the trajectory")
    print("    moved, not how many directions it moved in")
    print(c.round(2).to_string())


if __name__ == "__main__":
    raise SystemExit(main())

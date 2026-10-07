"""Figures for the S1 report (base runs, no intervention; all seeds)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS, RES = HERE / "runs", HERE / "results"
COL = {"A3W256": "#2a78d6", "A3W64": "#eb6834", "S1W256": "#1baf7a", "S3W256": "#eda100",
       "A1W128": "#e87ba4", "S2W128": "#008300"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False})


def load(fam):
    return pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob(f"mon_{fam}_*.csv"))], ignore_index=True)


def main():
    panels = [("acc_mean", "online accuracy of task"), ("MG|pnorm|span", "MG, param-norm log, span window"),
              ("MG|pnorm|within", "MG, param-norm log, within-task"), ("MG|loss|span", "MG, loss log, span window"),
              ("dormant_0.0", "dormant fraction (ReDo, tau=0)"), ("srank", "srank of last hidden layer")]
    for fam in ("F1", "F2"):
        df = load(fam)
        fig, axs = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        for ax, (m, title) in zip(axs.flat, panels):
            for c, g in df.groupby("cond"):
                mu = g.groupby("j")[m].mean()
                ax.plot(mu.index, mu.values, color=COL[c], lw=2, label=c)
            ax.set_title(title, loc="left")
            ax.set_xlabel("task")
        axs.flat[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"{fam}: base runs without intervention (mean over seeds)", x=0.01, ha="left")
        fig.savefig(HERE / f"fig_{fam}_trajectories.png", dpi=130)
        plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()

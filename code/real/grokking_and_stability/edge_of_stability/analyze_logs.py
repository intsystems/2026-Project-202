"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment_runner.plotting import FAINT, GREY, POINTER, RECURRENT, TRANSIENT
from experiment_runner.plotting import draw


def fig_eos():
    """Full-batch descent at the edge of stability, and what the logging stride hides."""
    runs = pd.read_csv(HERE / 'logs/eos_runs.csv').sort_values(["lr", "seed"])

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.78),
                             gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]})
    ax, bx, cx = axes

    # (a) the stability ratio along training, one line per rate, seed 1 only so the
    # panel stays readable; the seeds agree to within the linewidth (see eos_runs.csv).
    ax.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    ax.text(30500, 1.0, "$2/\\eta$", va="center", ha="left", **POINTER)
    for _, m in runs[runs.seed == 1].iterrows():
        sh = pd.read_csv(HERE / "logs" / f'{m["key"]}_sharp.csv')
        # Colour is the regime the run turns out to be in, as everywhere else: the
        # sub-threshold rates descend monotonically and are transient; the rates that
        # pin at 2/eta are the candidate recurrent regime this appendix is about.
        eos = pd.isna(m["diverged_at"]) and m["eta_lam_over_2_median_tail"] > 0.9
        c = RECURRENT if eos else (GREY if pd.notna(m["diverged_at"]) else TRANSIENT)
        ls = "-" if eos else ((0, (1, 2)) if pd.notna(m["diverged_at"]) else (0, (4, 2)))
        ax.plot(sh["step"], sh["eta_lam_over_2"], ls=ls, color=c, lw=1.0,
                clip_on=True, zorder=3 if eos else 2)
    ax.set_xlim(0, 30000)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("optimiser step")
    ax.set_ylabel(r"$\eta\lambda_{\max}/2$")
    ax.set_xticks([0, 15000, 30000])
    ax.set_xticklabels(["0", "15k", "30k"])

    # (b) sixty consecutive steps deep in the edge-of-stability phase, against the same
    # series read at the stride every full-batch log in this repository uses.
    #
    # Detrended, and that is not cosmetic. Over sixty steps the descent is much larger
    # than the two-cycle riding on it, so the raw loss is a straight line to the eye at
    # every window and the panel would show nothing -- an earlier draft of this figure
    # did exactly that. Removing the linear fit over the window shown puts the two
    # series on the scale of the thing being compared. The window is a typical one, not
    # a chosen one: its rise fraction, 0.492, is the run's own median.
    tr = pd.read_csv(HERE / "logs/eos_lr2e+06_s1_train.csv")
    seg = tr.iloc[20000:20060]
    step = seg["step"].to_numpy(float)
    loss = seg["train_loss"].to_numpy(float)
    resid = (loss - np.polyval(np.polyfit(np.arange(60.0), loss, 1),
                               np.arange(60.0))) * 1e9
    bx.axhline(0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    bx.plot(step, resid, "-", color=RECURRENT, lw=0.9, marker="o", ms=1.8,
            mec="none", label="every step", zorder=3)
    bx.plot(step[::10], resid[::10], ls=(0, (4, 2)), color=TRANSIENT, lw=1.0,
            marker="s", ms=3.0, mec="white", mew=0.5, label="stride 10", zorder=4)
    bx.set_xlabel("optimiser step")
    bx.set_ylabel("detrended loss ($10^{-9}$)")
    bx.set_xticks([20000, 20030, 20060])
    bx.set_xticklabels(["20k", "+30", "+60"])
    # headroom for the legend: the two-cycle fills the axes, and the stride-10 series
    # rides its upper branch, so an unexpanded top puts the legend on top of the data.
    bx.set_ylim(-3.0, 4.6)
    bx.set_yticks([-2, 0, 2])
    bx.legend(loc="upper right", handlelength=1.8)

    # (c) what the two non-monotonicity statistics do as the log is decimated.
    diag = pd.read_csv(HERE / 'logs/eos_diagnostics.csv')
    g = diag[(diag.column == "train_loss") & (diag.segment == "post") & (diag.tau == 1)]
    g = g.groupby(["lr", "subsample"]).agg(
        rises=("rises", "median"), crossings=("crossings", "median")).reset_index()
    eos_lrs = sorted(runs[(runs.eta_lam_over_2_median_tail > 0.9)
                          & (runs.diverged_at.isna())].lr.unique())
    for lr in eos_lrs:
        h = g[g.lr == lr].sort_values("subsample")
        if h.empty:
            continue
        cx.plot(h["subsample"], h["rises"], "-", color=RECURRENT, lw=0.9,
                marker="o", ms=2.6, mec="white", mew=0.4, alpha=0.85)
    sub = sorted(g["subsample"].unique())
    cx.axhline(0.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    cx.set_xscale("log")
    cx.set_xticks(sub)
    cx.set_xticklabels([str(s) for s in sub])
    cx.minorticks_off()
    cx.set_ylim(-0.04, 0.56)
    cx.set_xlabel("logging stride")
    cx.set_ylabel("fraction of steps rising")
    cx.text(sub[-1], 0.02, "monotone", ha="right", va="bottom", **POINTER)

    fig.tight_layout(w_pad=1.5)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_eos,))

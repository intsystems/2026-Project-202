"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiment_runner.plotting import CODE, RECURRENT, STOCHASTIC, TRANSIENT
from experiment_runner.plotting import draw


def fig_exclusion():
    """Why the exclusion decides the value on a transient and leaves a torus alone."""
    d = pd.read_csv(HERE / 'logs/sweep_windows.csv')
    d = d[(d.observer == "w_fro") & (d.theiler_label != "uncapped")]
    arms = [("fast", "recurrent, fast", RECURRENT, "-"),
            ("slow", "recurrent, slow", RECURRENT, (0, (3.5, 2))),
            ("transient", "transient", TRANSIENT, (0, (1, 1.6)))]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.78))
    handles = []
    for arm, label, colour, dash in arms:
        one = (d[d.arm == arm].groupby("theiler_used")
               .agg(MG=("MG", "median"), near=("frac_near_ref", "median"))
               .sort_index())
        x = one.index.to_numpy()
        ax.plot(x, one.MG.to_numpy(), color=colour, ls=dash, lw=1.1)
        bx.plot(x, one.near.to_numpy(), color=colour, ls=dash, lw=1.1)
        handles.append(Line2D([], [], color=colour, ls=dash, lw=1.1, label=label))

    for panel in (ax, bx):
        panel.set_xscale("symlog", linthresh=1.0)
        panel.set_xlim(0, 200)
        panel.set_xticks([0, 1, 10, 100])
        panel.set_xticklabels(["0", "1", "10", "100"])
        panel.set_xlabel("Theiler exclusion (samples)", labelpad=1.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(["1", "3", "10", "30"])
    ax.set_ylim(0.9, 45)
    ax.set_ylabel("components")
    ax.set_title("(a) the estimate", loc="left", pad=2.5)
    bx.set_ylabel("fraction")
    bx.set_ylim(-0.04, 1.0)
    bx.set_title("(b) neighbours that are returns", loc="left", pad=2.5)

    fig.tight_layout(rect=[0, 0.135, 1, 1], w_pad=1.5)
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=2.2, columnspacing=1.6)
    return fig


def fig_traces():
    """One raw scalar log per regime -- what the estimator is actually reading."""
    syn = pd.read_csv(HERE / 'logs/example_traces.csv')
    real = pd.read_csv(CODE / 'real/grokking_and_stability/transformer_grokking/logs/mod_wd1_train.csv')

    def z(a):
        a = np.asarray(a, float)
        return (a - a.mean()) / a.std()

    # Each panel spans what that regime needs to be legible: the recurrent arm's drive
    # period is 16 samples, so 400 samples is 25 cycles, while the transient and the
    # real run are shown whole. Standardised, because the estimator standardises.
    panels = [
        ("recurrent", z(syn["recurrent"].to_numpy()[:400]), np.arange(400),
         RECURRENT, "deterministic, recurrent", "sample"),
        ("transient", z(syn["transient"].to_numpy()), np.arange(len(syn)),
         TRANSIENT, "deterministic, transient", "sample"),
        ("stochastic", z(real["weight_norm"].to_numpy()),
         real["step"].to_numpy() / 1000.0, STOCHASTIC, "stochastically forced",
         "step ($10^3$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.78))
    for ax, (_, y, x, c, title, xlab) in zip(axes, panels):
        ax.plot(x, y, "-", color=c, lw=0.6, zorder=3)
        ax.set_title(title, pad=3.0)
        ax.set_xlabel(xlab)
        ax.set_ylim(-3.4, 3.4)
        ax.set_yticks([-2, 0, 2])
    axes[0].set_ylabel("standardised")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.tight_layout(w_pad=1.2)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_exclusion, fig_traces,))

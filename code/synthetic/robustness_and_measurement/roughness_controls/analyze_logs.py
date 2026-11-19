"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from experiment_runner.plotting import FAINT, GREY, RECURRENT, WIDTH
from experiment_runner.plotting import draw


def fig_switch():
    """A known change in the number of phases at matched roughness, tracked as it happens."""
    d = pd.read_csv(HERE / 'logs/geometry_switch_trace.csv')
    centres = np.sort(d.centre.unique())
    x = centres / 1000.0
    truth = d.groupby("centre").truth.first().reindex(centres).to_numpy(dtype=float)

    fig, (ax, rx) = plt.subplots(2, 1, figsize=(WIDTH, 1.86), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1.0]})
    # Named above each row rather than beside it: a row of a two-row figure is shorter than
    # either word set on end, and both ran off the canvas.
    for slot, column, ylabel in ((ax, "MG", "(a) the estimate, in components"),
                                 (rx, "roughness", "(b) the roughness")):
        stack = np.vstack([d[d.seed == s].sort_values("centre")[column].to_numpy()
                           for s in sorted(d.seed.unique())])
        slot.fill_between(x, stack.min(axis=0), stack.max(axis=0), facecolor=RECURRENT,
                          alpha=0.20, lw=0, zorder=3)
        slot.plot(x, np.median(stack, axis=0), "-", color=RECURRENT, lw=1.3, zorder=4)
        slot.set_title(ylabel, loc="left", pad=2.5)

    # The truth is a level the schedule holds, drawn as a band behind the estimate and only
    # where it exists. Over a ramp there are two systems inside one window and no number
    # the estimate could be right or wrong about, so those windows are struck out instead:
    # the old panel drew the estimate through them beside an unbroken reference.
    ax.plot(x, truth, "-", color=GREY, lw=5.5, alpha=0.42, solid_capstyle="butt", zorder=2)
    half = 0.5 * float(np.diff(x).min())
    missing = np.isnan(truth)
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        for slot in (ax, rx):
            slot.axvspan(x[lo] - half, x[hi - 1] + half, facecolor="none", edgecolor=FAINT,
                         hatch="////", lw=0.0, zorder=1)

    ax.set_ylim(0.0, 5.2)
    ax.set_yticks([0, 2, 4])
    # From zero, so that the flat trace is read against the fourfold change above it rather
    # than against its own noise: over every window of every seed the roughness lies
    # between 0.085 and 0.094.
    rx.set_ylim(0.0, 0.16)
    rx.set_yticks([0.0, 0.1])
    rx.set_xticks([10, 20, 30])
    rx.set_xlabel("window centre ($10^3$ optimiser steps)", labelpad=1.5)
    # Last, and on the shared axis: an axvspan added after a set_xlim brings the frame back
    # to the origin, which left a third of this one empty.
    rx.set_xlim(x.min(), x.max())

    handles = [Line2D([], [], color=GREY, lw=5.5, alpha=0.42, label="phases forced"),
               Line2D([], [], color=RECURRENT, lw=1.3,
                      label=r"$\hat{d}_{\mathrm{MG}}$, median of eight seeds"),
               Patch(facecolor="none", edgecolor=FAINT, hatch="////",
                     label="a ramp reaches in: no truth")]
    fig.tight_layout(rect=[0, 0.155, 1, 1.03], h_pad=0.4)
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.5, handletextpad=0.45, columnspacing=1.2)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_switch,))

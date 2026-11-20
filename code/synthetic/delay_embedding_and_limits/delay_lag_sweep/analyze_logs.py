"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import pandas as pd
import matplotlib.pyplot as plt
from experiment_runner.plotting import FAINT, GREY, POINTER, RECURRENT
from experiment_runner.plotting import draw


def fig_tau():
    """The delay-lag sweep: the usable span is about four tenths of a period.

    One torus of period 400 at E_max = 20, the configuration the paper's table
    reports. All of it is the recurrent regime, so the figure is monochrome and
    the six ranks are separated by marker shape.
    """
    d = pd.read_csv(HERE / 'logs/tau_sensitivity.csv')
    d = d[(d.period == 400) & (d.max_E == 20)]
    # tau='acorr' picks the lag from the autocorrelation, so its span differs
    # between the two seeds and it is not a point on a common x grid; the six
    # fixed lags 1..32 are. Reported in the caption.
    d = d[d.tau != "acorr"].copy()
    d["span_periods"] = d.span_periods.astype(float)

    rs = sorted(d.r.unique())
    marks = ["o", "s", "^", "v", "D", "X"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.2))

    for r, mk in zip(rs, marks):
        g = d[d.r == r]
        a = g.groupby("span_periods").MG.agg(["median", "min", "max"]).sort_index()
        x = a.index.values
        for panel, scale in ((ax, 1.0), (bx, float(r))):
            panel.vlines(x, a["min"] / scale, a["max"] / scale, color=RECURRENT,
                         lw=0.7, alpha=0.55, zorder=2)
            panel.plot(x, a["median"] / scale, "-", color=RECURRENT, lw=1.0,
                       marker=mk, ms=3.0, mec="white", mew=0.5, zorder=3,
                       label=f"$r = {r}$" if panel is ax else None)

    for panel in (ax, bx):
        panel.axvspan(0.30, 0.50, color=GREY, alpha=0.10, lw=0, zorder=0)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(0.038, 1.95)
        panel.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        panel.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
        panel.set_xticks([], minor=True)
        panel.set_xlabel("delay span (periods)")

    ax.set_ylim(1.15, 90)
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_yticks([], minor=True)
    ax.set_ylabel("estimated dimension")
    ax.set_title("(a) the estimate", loc="left")

    bx.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
    bx.set_ylim(0.26, 11)
    bx.set_yticks([0.3, 0.5, 1, 2, 5, 10])
    bx.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    bx.set_yticks([], minor=True)
    bx.set_ylabel("estimate / true rank")
    bx.set_title("(b) relative to the truth", loc="left")
    bx.text(0.40, 9.6, "usable", ha="center", va="top", **POINTER)

    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=1.8)
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.8, columnspacing=1.2)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_tau,))

"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import pandas as pd
import matplotlib.pyplot as plt
from experiment_runner.plotting import FAINT, POINTER, RECURRENT
from experiment_runner.plotting import draw


def fig_ceiling():
    """The tracking ceiling against each hypothesis's knob, and against its prediction."""
    d = pd.read_csv(HERE / 'logs/ceiling_summary.csv')
    d = d[d.arm == "frozen"]
    e = d[d.sweep == "E"].sort_values("max_E")
    n = d[d.sweep == "N"].sort_values("N")
    n56 = d[d.sweep == "N_E56"].sort_values("N")

    # Every arm here is the constructed recurrent system, so the figure is monochrome by
    # the rule in the module docstring; the two predictions are reference lines.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.95))

    ax.plot(e.max_E, e.takens, ls=(0, (3.5, 2.5)), color=FAINT, lw=0.9, zorder=1)
    ax.axhline(e.eckmann_ruelle.iloc[0], color=FAINT, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
    ax.plot(e.max_E, e["r_track_1"], "-", color=RECURRENT, lw=1.1, marker="o", ms=3.0,
            mec="white", mew=0.5, zorder=4, clip_on=False)
    ax.set_xscale("log")
    ax.set_xticks(e.max_E)
    ax.set_xticklabels([str(int(v)) for v in e.max_E])
    ax.minorticks_off()
    ax.set_ylim(0, 29)
    ax.set_xlabel("$E_{\\max}$   (record fixed at 8000)")
    ax.set_ylabel("rank tracked")
    ax.text(56, 27.5, "$E_{\\max}/2$", ha="right", va="top", **POINTER)
    ax.text(10.4, 8.6, "$2\\log_{10}N$", ha="left", va="bottom", **POINTER)

    bx.plot(n.N, n.eckmann_ruelle, ls=(0, (1, 2.5)), color=FAINT, lw=0.9, zorder=1)
    bx.axhline(10.0, color=FAINT, lw=0.9, ls=(0, (3.5, 2.5)), zorder=1)
    bx.plot(n.N, n["r_track_1"], "-", color=RECURRENT, lw=1.1, marker="o", ms=3.0,
            mec="white", mew=0.5, label="$E_{\\max}=20$", zorder=4)
    bx.plot(n56.N, n56["r_track_1"], ls=(0, (4, 2)), color=RECURRENT, lw=1.1, marker="s",
            ms=3.2, mfc="white", mec=RECURRENT, mew=0.9, label="$E_{\\max}=56$", zorder=5)
    bx.set_xscale("log")
    bx.set_xticks([1e3, 1e4, 1e5])
    bx.minorticks_off()
    bx.set_ylim(0, 14)
    bx.set_xlabel("record length   ($E_{\\max}$ fixed)")
    bx.set_ylabel("rank tracked")
    # The dotted line is the hypothesis this panel refutes, so it is the one labelled;
    # the dashed line is Takens for the E_max = 20 arm only and would be 28 for the
    # other, which is off scale, so the caption names it rather than a pointer here.
    bx.text(1.05e3, 5.7, "$2\\log_{10}N$", ha="left", va="top", **POINTER)
    bx.legend(loc="lower right", handlelength=2.0)

    fig.tight_layout(w_pad=1.8)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_ceiling,))

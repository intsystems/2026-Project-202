"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiment_runner.plotting import FAINT, GREY, RECURRENT
from experiment_runner.plotting import draw


def fig_aniso():
    """Anisotropy: the estimate follows the count, not the participation ratio.

    One panel per r. The CSV column is `rho`, but the paper renamed the
    amplitude decay factor to q, rho being the Spearman correlation everywhere
    else, so the axis says q. Everything here is one r-torus, i.e. the
    recurrent regime, so the figure is monochrome: the estimate is in the
    regime colour and every reference quantity is grey.
    """
    d = pd.read_csv(HERE / 'logs/aniso_summary.csv')
    rs = sorted(d.r.unique())

    fig, axes = plt.subplots(1, len(rs), figsize=(5.5, 2.2), sharey=True)
    for ax, r in zip(axes, rs):
        g = d[d.r == r].sort_values("rho")
        # the closed form (sum a^2)^2 / sum a^4 and the measured covariance PR
        # agree to two decimals: draw the prediction as a line and the
        # measurement as open markers on top of it, so the reader sees both
        ax.plot(g.rho, g.pr_pred, "-", color=GREY, lw=0.9, alpha=0.75, zorder=2)
        ax.plot(g.rho, g.pr_pos, "s", mfc="white", mec=GREY, mew=0.9, ms=3.2,
                ls="none", zorder=3)
        ax.axhline(r, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
        # MG is the median of three seeds and MG_sd their standard deviation
        ax.errorbar(g.rho, g.MG, yerr=g.MG_sd, color=RECURRENT, lw=1.2,
                    marker="o", ms=3.0, mec="white", mew=0.5, elinewidth=0.8,
                    capsize=1.6, capthick=0.8, zorder=4)
        ax.set_title(f"$r = {r}$", loc="left")
        ax.set_xlim(0.44, 1.06)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel("$q$")
    axes[0].set_ylim(0, 9.6)
    axes[0].set_yticks([0, 2, 4, 6, 8])
    axes[0].set_ylabel("components")

    handles = [
        Line2D([], [], color=RECURRENT, ls="-", marker="o", ms=3.0,
               mec="white", mew=0.5, label="estimate (3 seeds, $\\pm$ s.d.)"),
        Line2D([], [], color=GREY, ls="-", marker="s", mfc="white", mec=GREY,
               mew=0.9, ms=3.2, label="participation ratio"),
        Line2D([], [], color=FAINT, ls=(0, (1, 2.5)), lw=0.7,
               label="active dimension"),
    ]
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=0.9)
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.4)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_aniso,))

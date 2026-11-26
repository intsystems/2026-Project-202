"""Draw the article's regression and recurrent-generator comparison."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style import INK, figure_style, pt


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[2] / "code" / "synthetic" / "poster_examples"
WIDTH, FRACTION = 5.5, .98
BLUE, ORANGE, GREEN = "#2563EB", "#D97706", "#059669"
GRAY = "#687782"
ARMS = [("H4", "1 phase", BLUE), ("M4", "2 phases", ORANGE), ("T4", "4 phases", GREEN)]


def plot_components():
    regression = pd.read_csv(EXPERIMENTS / "known_linear_modes/logs/stationary_summary.csv")
    loss = regression.query('window == 8192 and observer == "loss"').sort_values("r")
    runs = pd.read_csv(EXPERIMENTS / "four_tone_rnn/logs/runs.csv").sort_values("seed")
    runs = runs[runs["learned"]]

    def paper(size):
        return pt(size, WIDTH, FRACTION)

    fig, (linear, mg, pr) = plt.subplots(1, 3, figsize=(WIDTH, 1.87))
    fig.subplots_adjust(left=.072, right=.97, top=.86, bottom=.30, wspace=.16)
    linear.plot(loss["r"], loss["median"], "o-", color=BLUE, lw=paper(1.2), ms=paper(4), label="MG@loss")
    linear.plot([1, 6], [1, 6], "--", color=GRAY, lw=paper(.9), label="Exact")
    linear.set(title="Linear regression", xlabel="Independent modes", ylabel="Estimate",
               xlim=(.75, 6.25), xticks=[1, 2, 4, 6], ylim=(.7, 7.3), yticks=[2, 4, 6])
    linear.legend(loc="upper left", frameon=False, handlelength=1.35, borderaxespad=.35)

    for ax, metric, title in ((mg, "MG_n0", "MG@neuron activation"),
                              (pr, "PR", "Full-state covariance PR")):
        for i, (name, _, color) in enumerate(ARMS):
            values = runs.loc[runs["arm"] == name, metric]
            jitter = np.linspace(-.12, .12, len(values))
            ax.scatter(i + jitter, values, s=paper(3.2) ** 2, color=color, linewidths=0, zorder=3)
            ax.hlines(values.median(), i - .20, i + .20, colors=INK, lw=paper(1.5), zorder=4)
        ax.set(title=title, xlabel="Four output spectral lines", xlim=(-.36, 2.36), ylim=(0, 8),
               xticks=range(len(ARMS)), xticklabels=[f"{name}\n{phases}" for name, phases, _ in ARMS],
               yticks=[0, 2, 4, 6, 8])
        ax.legend(handles=[
            Line2D([], [], ls="none", marker="o", color=GRAY, ms=paper(3.2), label="One seed"),
            Line2D([], [], color=INK, lw=paper(1.5), label="Median"),
        ], loc="upper left", frameon=False, ncol=2, handlelength=1, handletextpad=.35,
            columnspacing=.8, borderaxespad=.3)

    for ax in (linear, mg, pr):
        ax.grid(axis="y", alpha=.32)
        ax.set_axisbelow(True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with figure_style(WIDTH, FRACTION):
        fig = plot_components()
        fig.savefig(args.output / "components.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()

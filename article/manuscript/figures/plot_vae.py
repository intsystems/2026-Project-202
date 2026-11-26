"""Build the manuscript's text VAE figure from the recorded experiment."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from threadpoolctl import threadpool_limits

from figure_style import ANNOTATION, TEXT_WIDTH, figure_style, pt


FIGURES = Path(__file__).resolve().parent
VAE = FIGURES.parents[2] / "code/real/scalar_log_diagnostics/text_vae_latent_code"
sys.path.insert(0, str(VAE))
from analyze_logs import load_branch, measure


FRACTION = .96
WIDTH = TEXT_WIDTH * FRACTION
SWITCH = 1024
WINDOW = 512
END = 3072
ZOOM_END = 1408
ARMS = (
    ("base", "#707B87", "Lower penalty"),
    ("regularized", "#B65117", "Higher penalty"),
    ("protected", "#257770", "Free bits"),
)


def plot_vae(run_dir):
    fig = plt.figure(figsize=(WIDTH, 3.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[2.45, 1])
    nll = fig.add_subplot(grid[0, 0])
    zoom = fig.add_subplot(grid[0, 1])
    mg = fig.add_subplot(grid[1, :])
    info = fig.add_subplot(grid[2, :], sharex=mg)

    for arm, color, label in ARMS:
        log, mi = load_branch(arm, run_dir)
        assert np.array_equal(log.step.to_numpy(), np.arange(END)) and mi.step.max() == END
        probe = log.probe_nll.to_numpy()
        ends = range(WINDOW, len(probe) + 1, 128)
        scores = [measure(probe[end - WINDOW:end], WINDOW) for end in ends]
        assert not any(score["degenerate"] for score in scores)
        nll.plot(log.step, log.probe_nll, color=color, lw=.85, label=label)
        segment = log[log.step.between(SWITCH, ZOOM_END)]
        zoom.plot(segment.step, segment.probe_nll, color=color, lw=.85)
        mg.plot(ends, [score["MG"] for score in scores], color=color, lw=1.15, marker="o", ms=2.1)
        info.plot(mi.step, mi.MI, color=color, lw=1.15, marker="o", ms=2.1)

    for ax in (nll, mg, info):
        ax.axvline(SWITCH, color="#58646E", ls="--", lw=.8)
        ax.set(xlim=(0, END), xticks=[0, SWITCH, 2048, END])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=.18)
    nll.set(ylabel="NLL", ylim=(4, 8), yticks=[4.5, 7.5])
    mg.set(ylabel="MG@NLL", ylim=(3, 9), yticks=[4, 8])
    mg.axvspan(SWITCH, SWITCH + WINDOW, color="#707B87", alpha=.12, lw=0, zorder=0)
    info.set(ylabel="$I(X; Z)$\n(nats)", ylim=(-.25, 6), yticks=[0, 5], xlabel="Optimization step")
    for ax in (nll, mg):
        ax.tick_params(labelbottom=False)
    fig.align_ylabels([nll, mg, info])

    zoom.set(xlim=(SWITCH, ZOOM_END), ylim=(4.5, 5.3), xticks=[SWITCH, ZOOM_END], yticks=[4.6, 5.2])
    zoom.set_title("After update 1,024", fontsize=pt(ANNOTATION, WIDTH, FRACTION))
    zoom.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    zoom.yaxis.tick_right()
    zoom.grid(axis="y", alpha=.18)
    for spine in zoom.spines.values():
        spine.set_visible(True)
    for guide in mark_inset(nll, zoom, loc1=2, loc2=3, fc="none", ec="#8496A1", lw=.65):
        guide.set_in_layout(False)
    fig.legend(*nll.get_legend_handles_labels(), loc="outside upper center",
               ncol=len(ARMS), frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--run-dir", type=Path, help="Directory containing the base, regularized, and freebits runs.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with figure_style(WIDTH, FRACTION), threadpool_limits(limits=1):
        fig = plot_vae(args.run_dir)
        fig.savefig(args.output / "vae_real_log.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()

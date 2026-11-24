"""Illustrate the MG estimator on a periodic signal and a recorded VAE loss."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_application_traces import BLUE, EXPERIMENTS, FIGURES, GRAY

sys.path.insert(0, str(EXPERIMENTS.parent))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.embedding import reconstruct


VAE_CONFIG = EstimatorConfig(window=512, max_E=20, tau=1, k_neighbors=20,
                             theiler=39, theiler_cap=39)
INK = "#20262B"
BACKGROUND = "#BAC5CC"


def draw_example(fig, grid, column, title, index, signal, xlabel, ylabel):
    cfg = VAE_CONFIG
    rec = reconstruct(signal, cfg, seed=123)
    query = len(rec.points) // 2
    span = (cfg.max_E - 1) * rec.tau
    offsets = np.abs(np.arange(len(rec.points)) - query)
    excluded = (offsets > 0) & (offsets <= rec.theiler)
    admissible = np.flatnonzero(offsets > rec.theiler)
    distances = np.linalg.norm(rec.points[admissible] - rec.points[query], axis=1)
    by_distance = np.argsort(distances)
    neighbors = admissible[by_distance[:cfg.k_neighbors]]

    trace = fig.add_subplot(grid[0, column:column + 2])
    trace.plot(index, signal, color=GRAY, lw=.85)
    segment = slice(query, query + span + 1)
    trace.plot(index[segment], signal[segment], color=BLUE, lw=1.35)
    trace.scatter(index[[query, query + span]], signal[[query, query + span]],
                  color=BLUE, s=9, zorder=3)
    trace.set(title=title, xlabel=xlabel, ylabel=ylabel)
    trace.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    trace.yaxis.set_major_locator(MaxNLocator(3))

    geometry = fig.add_subplot(grid[1, column], aspect="equal")
    projection = rec.points[:, [0, -1]]
    geometry.scatter(*projection[admissible].T, s=7, color=BACKGROUND, linewidths=0)
    geometry.scatter(*projection[excluded].T, s=12, marker="x", color=GRAY, linewidths=.6)
    geometry.scatter(*projection[neighbors].T, s=16, color=BLUE,
                     edgecolor="white", linewidth=.22, zorder=3)
    geometry.scatter(*projection[query], s=28, facecolor="white",
                     edgecolor=INK, linewidth=.95, zorder=4)
    low, high = projection.min(), projection.max()
    pad = .06 * (high - low)
    geometry.set(title="Delay projection", xlabel=r"$z_t$", ylabel=r"$z_{t+(E-1)\tau}$",
                 xlim=(low - pad, high + pad), ylim=(low - pad, high + pad))
    geometry.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    geometry.yaxis.set_major_locator(MaxNLocator(4, integer=True))

    radius = fig.add_subplot(grid[1, column + 1])
    k = cfg.k_neighbors
    ranked = np.maximum(distances[by_distance[:2 * k]], cfg.floor_distance)
    ranks = np.arange(1, len(ranked) + 1)
    radius.plot(ranks, ranked, color=BACKGROUND, lw=1.2)
    radius.scatter(ranks[:k], ranked[:k], color=BLUE, s=8, linewidths=0)
    radius.axhline(ranked[k - 1], ls=":", lw=.75, color=GRAY)
    radius.axvline(k, ls=":", lw=.75, color=GRAY)
    radius.annotate(r"$r_{ik}$", (k, ranked[k - 1]), xytext=(4, -5),
                    textcoords="offset points", va="top", color=INK)
    radius.set(title="Distances", xlabel=r"Rank $j$", ylabel=r"$r_{ij}$",
               xlim=(0, ranks[-1] + 1), xticks=[1, k, ranks[-1]])
    radius.yaxis.set_major_locator(MaxNLocator(4))
    for ax in (trace, geometry, radius):
        ax.grid(axis="y", alpha=.16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logs = EXPERIMENTS / "scalar_log_diagnostics/text_vae_latent_code/logs"
    log = pd.read_csv(logs / "vae_base_log.csv", float_precision="round_trip")
    window = log[log.step.between(2048, 2559)]
    np.testing.assert_array_equal(window.step, np.arange(2048, 2560))
    sample = np.arange(VAE_CONFIG.window)
    periodic = np.sin(2 * np.pi * sample / (70 * np.sqrt(2)) + .2)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "axes.axisbelow": True})
    fig = plt.figure(figsize=(6.4, 2.95), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=(.78, 1.12), width_ratios=(1.1, .85, 1.1, .85))
    draw_example(fig, grid, 0, "Periodic signal", sample, periodic, "Sample index", "Signal")
    draw_example(fig, grid, 2, "Text reconstruction loss", window.step.to_numpy(),
                 window.probe_nll.to_numpy(), "Optimization step", "Loss")
    fig.legend(handles=[
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", ls="none", ms=4.5,
               label="Search center (query)"),
        Line2D([], [], marker="x", color=GRAY, ls="none", ms=4.5,
               label="Excluded in time"),
        Line2D([], [], marker="o", color=BLUE, ls="none", ms=4.5,
               label=r"$k$ nearest eligible neighbors"),
    ], loc="outside lower center", ncol=3, frameon=False, handletextpad=.3, columnspacing=1.2)
    fig.savefig(args.output / "icomp_method.pdf")


if __name__ == "__main__":
    main()

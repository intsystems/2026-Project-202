"""Plot activity, delay geometry, and MG across FORCE training checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import NullLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_force_log_geometry import FIGURES, LOGS, MG_CONFIG, load_trace, windowed_mg


BUNDLE = LOGS / "force_training_traces.npz"
DISPLAY_STEPS = (0, 150, 500, 40000)
DISPLAY_COLORS = ("#BD520E", "#9056A0", "#257770", "#0072B2")


def read_run(rollout_dir):
    if rollout_dir is None:
        with np.load(BUNDLE) as bundle:
            return bundle["steps"], bundle["time"], bundle["activity"]
    metadata = json.loads((rollout_dir / "metadata.json").read_text(encoding="utf-8"))
    steps = np.sort(metadata["checkpoints"])
    traces = [load_trace(step, rollout_dir) for step in steps]
    return steps, traces[0][0], np.stack([activity for _, activity in traces])


def checkpoint_summary(scores):
    """Summarize the valid windows in each fixed-weight simulation."""
    summary = np.full((len(scores), 3), np.nan)
    for index, windows in enumerate(scores):
        valid = windows[np.isfinite(windows)]
        if valid.size:
            summary[index] = valid.min(), np.median(valid), valid.max()
    return summary


def plot_progress(ax, steps, summary, phase_indices, horizon, scale):
    shown = steps <= horizon
    time = steps[shown] / scale
    lower, median, upper = summary[shown].T
    ax.fill_between(time, lower, upper, color="#DCE5EE", linewidth=0)
    ax.plot(time, median, color="#3C5269", lw=1.0, marker="o", markersize=2.0)
    ax.axhline(1, color="#89939E", lw=0.7, ls="--")
    for index, color in zip(phase_indices, DISPLAY_COLORS):
        if shown[index] and np.isfinite(summary[index, 1]):
            ax.scatter(steps[index] / scale, summary[index, 1], s=22,
                       color=color, zorder=4, edgecolors="white", linewidths=0.5)
    ax.set_xlim(-0.025 * horizon / scale, 1.025 * horizon / scale)


def plot_transition(steps, time, activity, summary):
    index_by_step = {step: index for index, step in enumerate(steps)}
    phase_indices = [index_by_step[step] for step in DISPLAY_STEPS]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.6, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "axes.axisbelow": True})
    fig = plt.figure(figsize=(6.0, 4.15), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1.22, 1.42])

    span = (MG_CONFIG.max_E - 1) * MG_CONFIG.tau
    short = time <= 125
    for col, (index, color) in enumerate(zip(phase_indices, DISPLAY_COLORS)):
        iteration = steps[index]
        trace = fig.add_subplot(grid[0, col])
        trace.plot(time[short], activity[index, short], color=color, lw=0.75)
        trace.set(title=f"{iteration:,} iterations", xlim=(0, 125), ylim=(-1.05, 1.05),
                  xticks=[0, 60, 120], yticks=[-1, 0, 1], xlabel="Simulation time")
        trace.title.set_color(color)
        trace.grid(alpha=0.16)
        delay = fig.add_subplot(grid[1, col], aspect="equal")
        window = activity[index, :MG_CONFIG.window]
        standardized = (window - window.mean()) / window.std()
        delay.scatter(standardized[:-span], standardized[span:], s=1.8, alpha=0.19,
                      linewidths=0, color=color)
        delay.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4),
                  xticks=[-3, 0, 3], yticks=[-3, 0, 3], xlabel=r"$z_t$")
        delay.grid(alpha=0.16)
        if col == 0:
            trace.set_ylabel("Neuron activation")
            delay.set_ylabel(r"$z_{t+(E-1)\tau}$")
        else:
            trace.tick_params(labelleft=False)
            delay.tick_params(labelleft=False)

    early = fig.add_subplot(grid[2, :2])
    full = fig.add_subplot(grid[2, 2:], sharey=early)
    for ax, title, horizon, scale in ((early, "First 1,000 iterations", 1000, 1),
                                      (full, "Full training run", 40000, 1000)):
        plot_progress(ax, steps, summary, phase_indices, horizon, scale)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    early.set(xticks=[0, 250, 500, 750, 1000], xlabel="Training iteration",
              ylabel="MG@neuron activation", yscale="log")
    early.set_yticks([0.5, 1, 3, 10, 30, 100], labels=["0.5", "1", "3", "10", "30", "100"])
    early.yaxis.set_minor_locator(NullLocator())
    full.set(xticks=[0, 10, 20, 30, 40], xlabel="Training iteration (thousands)")
    full.tick_params(labelleft=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--rollout-dir", type=Path, help="Use new checkpoint rollouts instead of the bundled traces.")
    parser.add_argument("--save-traces", type=Path, help="Export the scalar traces used by the figure.")
    parser.add_argument("--summary", type=Path, help="Write checkpoint MG summaries to CSV.")
    args = parser.parse_args()
    steps, time, activity = read_run(args.rollout_dir)
    scores = np.stack([windowed_mg(trace, reject_degenerate=True)[1] for trace in activity])
    summary = checkpoint_summary(scores)
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plot_transition(steps, time, activity, summary)
    fig.savefig(args.output / "force_training_transition.pdf")
    plt.close(fig)
    if args.summary is not None:
        np.savetxt(args.summary, np.column_stack((steps, summary)), delimiter=",",
                   header="training_iteration,mg_min,mg_median,mg_max",
                   comments="", fmt=("%d", "%.12g", "%.12g", "%.12g"))
    if args.save_traces is not None:
        np.savez_compressed(args.save_traces, steps=steps, time=time, activity=activity)


if __name__ == "__main__":
    main()

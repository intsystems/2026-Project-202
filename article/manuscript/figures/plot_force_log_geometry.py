"""Build the FORCE figure from saved neuron logs or newly generated rollouts."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent
CODE = FIGURES.parents[2] / "code"
LOGS = CODE / "real/scalar_log_diagnostics/force_rnn_cycle/logs"
sys.path.insert(0, str(CODE))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate as compute_mg

# Match the rolling MG configuration in force_rnn_cycle/analyze.py.
MG_CONFIG = EstimatorConfig(window=4096, max_E=20, tau=4, k_neighbors=20,
                            theiler=156, theiler_cap=156, spectral_bins=())
WINDOW_STRIDE = 512


def windowed_mg(activity, *, reject_degenerate=False):
    ends = np.arange(MG_CONFIG.window, len(activity) + 1, WINDOW_STRIDE)
    with threadpool_limits(limits=1):
        estimates = [compute_mg(activity[end - MG_CONFIG.window:end], MG_CONFIG, seed=123)
                     for end in ends]
    scores = np.asarray([np.nan if reject_degenerate and result.degenerate else result.MG
                         for result in estimates])
    return ends, scores


def load_trace(checkpoint, rollout_dir):
    if rollout_dir is None:
        log = np.genfromtxt(LOGS / f"force_checkpoint_{checkpoint:05d}_log.csv",
                            delimiter=",", names=True, encoding="utf-8")
        return log["evaluation_time"], log["neuron_0_activity"]
    metadata = json.loads((rollout_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(rollout_dir / f"rollout_{checkpoint:05d}.npz") as rollout:
        activity = rollout["activity"] if "activity" in rollout else np.tanh(rollout["states"][:, 0])
    return np.arange(len(activity)) * metadata["dt"], activity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--rollout-dir", type=Path, help="Directory containing a trained run's NPZ rollouts.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = MG_CONFIG
    span = (config.max_E - 1) * config.tau
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 4.35), layout="constrained")
    checkpoints = ((0, "#C04E00", "Before training"), (40000, "#0072B2", "After training"))
    for column, (checkpoint, color, title) in enumerate(checkpoints):
        time, activity = load_trace(checkpoint, args.rollout_dir)
        window_ends, scores = windowed_mg(activity)
        window_times = window_ends * (time[1] - time[0])
        trace, projection, estimate = axes[:, column]
        trace.set_title(title, fontweight="medium")
        trace.plot(time, activity, color=color, linewidth=0.6)
        trace.set_ylim(-1.05, 1.05)

        samples = activity[:config.window]
        standardized = (samples - samples.mean()) / samples.std()
        projection.scatter(standardized[:-span], standardized[span:], s=2,
                           color=color, alpha=0.22, linewidths=0)
        projection.set_xlim(-3.5, 3.5)
        projection.set_ylim(-3.5, 3.5)
        projection.set_aspect("equal", adjustable="box")
        projection.set_xlabel(r"$z_t$")
        projection.set_xticks([-3, 0, 3])
        projection.set_yticks([-3, 0, 3])

        estimate.axvspan(0, window_times[0], color="#E8EDF0", zorder=0)
        estimate.axhline(1, color="#66727D", linestyle="--", linewidth=0.7)
        estimate.plot(window_times, scores, color=color,
                      linewidth=1.1, marker="o", markersize=2.8)
        estimate.set_ylim(0, 6.7)
        estimate.set_yticks([0, 2, 4, 6])

        for panel in (trace, estimate):
            panel.set_xlim(time[0], window_times[-1])
            panel.set_xticks([0, 400, 800])
            panel.set_xlabel("Rollout time")
        for panel in (trace, projection, estimate):
            panel.spines[["top", "right"]].set_visible(False)
            panel.set_axisbelow(True)
            panel.grid(alpha=0.17)

    axes[0, 0].set_ylabel("Neuron activation")
    axes[1, 0].set_ylabel(r"$z_{t+(E-1)\tau}$")
    axes[2, 0].set_ylabel("MG@neuron activation")
    fig.savefig(args.output / "force_log_geometry.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

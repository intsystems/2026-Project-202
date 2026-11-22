"""Rebuild the ResNet, MNIST, and grokking figures from the bundled CSVs."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIGURES = Path(__file__).resolve().parent
EXPERIMENTS = FIGURES.parents[2] / "code" / "real"
BLUE = "#0072B2"
GRAY = "#707B87"
ORANGE = "#B65117"
PURPLE = "#88488F"


def read(logs, file):
    return pd.read_csv(logs / file)


def plot_resnet():
    logs = EXPERIMENTS / "scalar_log_diagnostics/resnet_freeze/logs"
    figure, axes = plt.subplots(3, 2, sharex=True, sharey="row", figsize=(6.1, 3.8))
    figure.subplots_adjust(left=.13, right=.98, top=.84, bottom=.13, hspace=.20, wspace=.14)
    freeze = 4000
    repeat = 20
    conditions = (("base", GRAY), ("freeze_head", BLUE))
    for column, (mode, title) in enumerate((("scratch", "From scratch"),
                                           ("finetune", "ImageNet initialization"))):
        norm, mg, rank = axes[:, column]
        norm.set_title(title)
        norms = read(logs, f"{mode}_param_norm.csv")
        windows = read(logs, f"{mode}_windows.csv")
        detection = read(logs, f"{mode}_detector_per_run.csv")
        result = detection.query("seed == @repeat and arm == 'freeze_head' and stat == 'MG'").iloc[0]
        assert result.hit and not result.false_alarm
        for arm, color in conditions:
            norm.plot(norms.step, norms[f"{arm}_s{repeat}"], color=color, lw=1.0)
            trace = windows.query("arm == @arm and seed == @repeat").sort_values("end")
            mg.plot(trace.end, trace.MG, color=color, marker="o", lw=1.15, ms=2.2)
            rank.plot(trace.end, trace.update_PR, color=color, marker="o", lw=1.15, ms=2.2)
            if arm == "freeze_head":
                alarm_step = int(freeze + result.delay)
                alarm_mg = trace.loc[trace.end == alarm_step, "MG"].item()
                mg.scatter(alarm_step, alarm_mg, s=38, facecolors="white", edgecolors=BLUE, zorder=6)
        for axis in axes[:, column]:
            axis.axvline(freeze, ls="--", lw=.95, color=ORANGE)
            axis.grid(axis="y", alpha=.17)
        rank.set(xlim=(0, 10000), xticks=[0, 4000, 8000], xlabel="Optimization step")
    for axis, label in zip(axes[:, 0], ("Weight norm", "MG@weight norm", "Update PR")):
        axis.set_ylabel(label)
    figure.legend(handles=[
        Line2D([], [], color=GRAY, label="Unchanged training"),
        Line2D([], [], color=BLUE, label="Backbone frozen"),
        Line2D([], [], color=ORANGE, ls="--", label="Freeze"),
        Line2D([], [], color=BLUE, ls="none", marker="o", markerfacecolor="white", label="Alarm"),
    ], loc="upper center", ncol=4, frameon=False)
    return figure


def plot_mnist():
    logs = EXPERIMENTS / "model_selection_and_stopping/mnist_early_stopping/logs"
    stops = read(logs, "decisions_10_17.csv").query("noise == 0.4 and action == 'stop'")
    steps = stops.pivot(index="seed", columns="method", values="step")
    # Show the first held-out repeat in which the probe-loss rule stops earlier.
    repeat = steps.index[steps.level < steps.MG][0]
    sample = stops[stops.seed == repeat].set_index("method")
    signal = read(logs, f"seed{repeat}_logs.csv")
    features = read(logs, f"seed{repeat}_features.csv")
    scores = features.query("obs == 'norm'").sort_values("step")
    metrics = read(logs, f"seed{repeat}_metrics.csv").set_index("step")
    rules = read(logs, "frozen_rules.csv").query("decision == 'stop_0.4'").set_index("method")
    assert rules.loc["MG", "obs"] == "norm" and rules.loc["MG", "sign"] == -1
    threshold = rules.loc["MG", "threshold"]
    assert scores.loc[scores.MG > threshold, "step"].iloc[0] == sample.loc["MG", "step"]
    probe = features.query("obs == 'probe'").sort_values("step")
    assert rules.loc["level", "obs"] == "probe" and rules.loc["level", "sign"] == 1
    assert probe.loc[probe.level <= rules.loc["level", "threshold"], "step"].iloc[0] == sample.loc["level", "step"]

    figure = plt.figure(figsize=(6.5, 4.6), layout="constrained")
    grid = figure.add_gridspec(4, 2, width_ratios=(1.75, 1), height_ratios=(.65, 1, 1, 1.2))
    norm = figure.add_subplot(grid[0, 0])
    mg = figure.add_subplot(grid[1, 0], sharex=norm)
    loss = figure.add_subplot(grid[2, 0], sharex=norm)
    accuracy = figure.add_subplot(grid[3, 0], sharex=norm)
    outcomes = figure.add_subplot(grid[:2, 1])
    timing = figure.add_subplot(grid[2:, 1], sharex=outcomes)

    norm.plot(signal.step, signal.norm, color=GRAY, lw=1)
    norm.set(title="One held-out repeat", ylabel="Weight norm")
    mg.plot(scores.step, scores.MG, color=BLUE, lw=1, marker="o", ms=3)
    mg.axhline(threshold, color=BLUE, lw=.9, ls=":")
    mg.set_ylabel("MG@weight norm")
    loss.plot(signal.step, signal.probe, color=PURPLE, alpha=.3, lw=.65)
    loss.plot(probe.step, probe.level, color=PURPLE, lw=1, marker="s", ms=2.8, label="Window mean")
    loss.axhline(rules.loc["level", "threshold"], color=PURPLE, lw=.9, ls=":")
    loss.set_ylabel("Probe loss")
    loss.legend(loc="upper right", frameon=False)

    accuracy.plot(metrics.index, metrics.test_acc * 100, color=GRAY, lw=1.2,
                  marker=".", label="Test")
    accuracy.plot(metrics.index, metrics.train_noisy_acc * 100, color=ORANGE, lw=1,
                  marker=".", ls="--", label="Train (noisy labels)")
    methods = (("level", "Probe loss", PURPLE, "s"), ("MG", "MG", BLUE, "o"),
               ("fixed", "Fixed step", GRAY, "D"))
    decision_scores = {"level": (loss, probe.set_index("step").level),
                       "MG": (mg, scores.set_index("step").MG)}
    for method, _, color, marker in methods[:2]:
        step = int(sample.loc[method, "step"])
        test_acc = metrics.loc[step, "test_acc"]
        assert np.isclose(test_acc, sample.loc[method, "test_acc"])
        for axis in (norm, mg, loss, accuracy):
            axis.axvline(step, color=color, lw=.75, ls="--", alpha=.7)
        score_axis, values = decision_scores[method]
        score_axis.scatter(step, values.loc[step], s=30, marker=marker, color=color, zorder=5)
        accuracy.scatter(step, test_acc * 100, s=27, marker=marker, color=color, zorder=5)
    accuracy.set(xlim=(0, 2048), ylim=(0, 100), xticks=[0, 256, 512, 1024, 1536, 2048],
                 xlabel="Optimization step", ylabel="Accuracy (%)")
    accuracy.legend(loc="lower right", frameon=False)
    for axis in (norm, mg, loss, outcomes):
        axis.tick_params(labelbottom=False)

    method_keys = [method for method, _, _, _ in methods]
    method_steps = steps[method_keys]
    test_results = 100 * stops.pivot(index="seed", columns="method", values="test_acc")[method_keys]
    for values in test_results.to_numpy():
        outcomes.plot(range(len(methods)), values, lw=.65, color=GRAY, alpha=.38)
    for position, (method, _, color, marker) in enumerate(methods):
        outcomes.scatter(np.full(len(test_results), position), test_results[method],
                         color=color, marker=marker, s=19, zorder=3)
        frequencies = method_steps[method].value_counts()
        timing.scatter(np.full(len(frequencies), position), frequencies.index,
                       s=24, color=color, marker=marker)
        for step, count in frequencies.items():
            timing.annotate(str(count), (position, step), xytext=(5, 0),
                            textcoords="offset points", va="center", color=color)
    outcomes.set(title="All held-out repeats", ylabel="Test accuracy at stop (%)")
    timing.set(ylabel="Stopping step", yticks=sorted(np.unique(method_steps)),
               ylim=(200, 570), xlim=(-.24, len(methods) - .76),
               xticks=range(len(methods)), xticklabels=[label for _, label, _, _ in methods])
    for axis in (norm, mg, loss, accuracy, outcomes, timing):
        axis.grid(axis="y", alpha=.17)
    return figure


def plot_grokking():
    logs = EXPERIMENTS / "grokking_and_stability/transformer_grokking/logs"
    aligned = read(logs, "headline_trace.csv").query("column == 'weight_norm'")
    ranks = read(logs, "rank_windows.csv")
    milestones = read(logs, "rank_milestones.csv").set_index("run")
    event = milestones.loc["mod_wd1", "t_gen"]
    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(6.1, 3.5))
    figure.subplots_adjust(left=.12, right=.965, top=.80, bottom=.15, hspace=.35, wspace=.29)
    norm, mg, rank, accuracy = axes.flat
    cases = (("mod_wd1", "mod_wd1_train.csv", BLUE),
             ("mod_wd0", "sketched__mod_wd0_train.csv", GRAY))
    for run, training_file, color in cases:
        training = read(logs, training_file)
        scalar = aligned[aligned.run == run].sort_values("mid_step")
        geometry = ranks[ranks.run == run].sort_values("centre")
        assert np.array_equal(scalar.mid_step, geometry.centre)
        assert np.allclose(scalar.PR_det, geometry.fn_PR_pos_det)
        norm.plot(training.step, training.weight_norm, color=color, lw=1.05)
        mg.plot(scalar.mid_step, scalar.MG, color=color, lw=1.05)
        rank.plot(geometry.centre, geometry.fn_PR_pos_det, color=color, lw=1.05)
        accuracy.plot(training.step, training.val_acc * 100, color=color, lw=1.1)
        accuracy.plot(training.step, training.train_acc * 100, color=color, lw=.85, ls="--", alpha=.85)
    for axis, title in zip(axes.flat, ("Weight norm", "MG@weight norm", "Detrended logit trajectory PR",
                                     "Train and validation accuracy (%)")):
        axis.set_title(title, loc="left")
        axis.axvline(event, color=ORANGE, ls=":", lw=1.0)
        axis.grid(axis="y", alpha=.17)
        axis.set_xlim(0, 20000)
    accuracy.set_ylim(-3, 103)
    accuracy.legend(handles=[Line2D([], [], color="black", label="Validation"),
                             Line2D([], [], color="black", ls="--", label="Train")],
                    loc="center left", bbox_to_anchor=(.20, .60), frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
        axis.set_xlabel("Optimization step")
    figure.legend(handles=[
        Line2D([], [], color=BLUE, label="With weight decay"),
        Line2D([], [], color=GRAY, label="Zero decay"),
        Line2D([], [], color=ORANGE, ls=":", label="Generalization"),
    ], ncol=3, frameon=False, loc="upper center")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.7, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "axes.axisbelow": True})
    for filename, plot in (("resnet_freeze_trace.pdf", plot_resnet),
                           ("mnist_stop_trace.pdf", plot_mnist),
                           ("grokking_matched_traces.pdf", plot_grokking)):
        figure = plot()
        figure.savefig(args.output / filename)
        plt.close(figure)


if __name__ == "__main__":
    main()

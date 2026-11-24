"""Draw four article figures from saved selection and warning records."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from plot_application_traces import BLUE, GRAY, ORANGE, PURPLE, EXPERIMENTS

FIGURES = Path(__file__).resolve().parent
DATA = FIGURES / "data"
sys.path.insert(0, str(EXPERIMENTS.parent))
from real.model_selection_and_stopping.rnn_forecast_selection.forecast import forecast


def plot_walker_selection():
    logs = EXPERIMENTS / "model_selection_and_stopping/walker2d_policy_selection/logs"
    selections = pd.read_csv(logs / "checkpoint_selections.csv").query("seed > 231")
    outcomes = selections.pivot(index="seed", columns="method", values="reward")
    methods = [
        ("MG", "MG@action norm", BLUE),
        ("entropy", "Spectral entropy", GRAY),
        ("fixed_pilot", "Fixed policy choice", GRAY),
    ]
    fig, ax = plt.subplots(figsize=(5.9, 1.75), layout="constrained")
    ax.axvline(0, color="#8B969F", ls="--", lw=.95, zorder=1)
    offsets = np.linspace(-.24, .24, len(outcomes))
    for row, (key, _, color) in enumerate(methods):
        gain = outcomes[key] - outcomes["reward"]
        ax.scatter(gain, row + offsets, s=22, facecolor="white", edgecolor=color,
                   linewidth=.9, zorder=3)
        ax.vlines(gain.mean(), row - .29, row + .29, color=color, lw=2.25, zorder=4)
    ax.set(xlabel="Reward difference in perturbed trials\nrelative to selection by unperturbed reward",
           yticks=range(len(methods)), yticklabels=[label for _, label, _ in methods],
           ylim=(2.5, -.5), xticks=[-.5, 0, .5, 1], xlim=(-.51, 1.04))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(axis="y", alpha=.19)
    ax.legend(handles=[
        Line2D([], [], color=GRAY, marker="o", markerfacecolor="white", ls="none", ms=4.5,
               label="Training repeat"),
        Line2D([], [], color=GRAY, marker="|", markeredgewidth=2, ls="none", ms=8,
               label="Mean difference"),
    ], loc="lower right", bbox_to_anchor=(1, 1.00), ncol=2, frameon=False,
              borderaxespad=0, handletextpad=.2, columnspacing=1.1)
    return fig


def plot_ssl():
    runs = pd.read_csv(DATA / "ssl_probe_scores.csv")
    by_config = runs.groupby(["family", "configuration"])[["gradient_norm_mg", "probe_accuracy"]].mean()
    families = [("ss", "SimSiam", BLUE, "o"), ("byol", "BYOL", PURPLE, "s"),
                ("vic", "VICReg", ORANGE, "^"), ("clr", "SimCLR", "#257770", "D")]
    fig, full = plt.subplots(figsize=(5.9, 2.85), layout="constrained")
    detail = full.inset_axes([.40, .12, .56, .56])
    for ax in (full, detail):
        for key, label, color, marker in families:
            group = by_config.loc[key]
            ax.scatter(group.gradient_norm_mg, 100 * group.probe_accuracy, s=29,
                       c=color, marker=marker, label=label, edgecolor="white", linewidth=.45)
        ax.grid(alpha=.17)
    full.set(xlabel="MG@gradient norm", ylabel="Linear classifier accuracy (%)", ylim=(6, 102),
             xlim=(7.75, 15.65))
    full.legend(frameon=False, loc="center left", bbox_to_anchor=(.02, .57),
                handletextpad=.3, borderaxespad=.15)
    left, right = 11.35, 15.4
    bottom, top = 84.8, 98.2
    detail.set(xlim=(left, right), ylim=(bottom, top), xticks=[12, 13, 14, 15], yticks=[85, 90, 95])
    detail.tick_params(labelsize=7, length=2, pad=2)
    for spine in detail.spines.values():
        spine.set(visible=True, color="#8496A1", linewidth=.65)
    frame = "#8496A1"
    full.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                             fill=False, lw=.75, edgecolor=frame))
    for edge, side in [(left, 0), (right, 1)]:
        full.add_artist(ConnectionPatch(
            xyA=(edge, bottom), xyB=(side, 1), coordsA=full.transData, coordsB=detail.transAxes,
            axesA=full, axesB=detail, color=frame, lw=.68, zorder=2,
        ))
    return fig


def plot_slingshot():
    windows = pd.read_csv(DATA / "slingshot_windows.csv")
    event = json.loads((DATA / "slingshot_example.json").read_text(encoding="utf-8"))
    onset = event["onset"]
    relative = windows.step - onset
    scored = windows.step >= event["first_scored_step"]
    for time in event["onsets"]:
        scored &= ~windows.step.between(time, time + event["horizon"])

    fig, (loss, mg, roughness) = plt.subplots(
        3, 1, sharex=True, figsize=(6.2, 3.15), layout="constrained",
        height_ratios=[1.12, 1, 1],
    )
    loss.fill_between(relative, windows.min_log10_loss, windows.max_log10_loss, color=GRAY, alpha=.18)
    loss.plot(relative, windows.mean_log10_loss, color=GRAY, lw=1.1)
    loss.set_ylabel(r"$\log_{10}$ training loss")
    loss.annotate("Spike onset", (0, .9), xycoords=loss.get_xaxis_transform(),
                  xytext=(5, 0), textcoords="offset points", va="center")
    for ax, signal, method, label, color in [
        (mg, windows.mg, "MG", "MG@log loss", BLUE),
        (roughness, windows.roughness, "roughness", "Roughness", PURPLE),
    ]:
        threshold = event["thresholds"][method]
        eligible = scored & relative.between(-event["horizon"], -1)
        first = relative[eligible & (signal <= threshold)].iloc[0]
        assert -first == event["leads"][method]
        ax.plot(relative, signal, color=color, alpha=.24, lw=1)
        ax.plot(relative, signal.where(scored), color=color, lw=1.2)
        ax.axhline(threshold, color=color, ls=":", lw=.9)
        value = signal.loc[relative == first].item()
        ax.scatter(first, value, color=color, s=21, zorder=5)
        ax.axvline(first, color=color, ls="--", lw=.7, alpha=.75)
        ax.annotate(f"{-first} steps before", (first, .89), xycoords=ax.get_xaxis_transform(),
                    xytext=(5, 0), textcoords="offset points", va="center", color=color, fontsize=7.5)
        ax.set_ylabel(label)
    for ax in (loss, mg, roughness):
        ax.axvspan(-event["horizon"], 0, color="#DCE5EE", alpha=.45, zorder=0)
        ax.axvline(0, color="#303941", lw=.8)
        ax.grid(axis="y", alpha=.14)
    roughness.set(xlabel="Optimization step relative to spike onset", xlim=(-650, 350),
                  xticks=[-600, -400, -200, 0, 200])
    return fig


def plot_forecast():
    logs = EXPERIMENTS / "model_selection_and_stopping/rnn_forecast_selection/logs"
    comparisons = pd.read_csv(logs / "query_fresh_comparisons.csv")
    model_names = {"knn8": "5-NN, lag 8", "knn48": "5-NN, lag 48", "ridge48": "Ridge, lag 48"}
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.2, 2.6), layout="constrained")
    for ax, arm, title in zip(axes, ("M4", "T4"), ("Mixed frequencies", "Independent frequencies")):
        repeat = 681
        signal = pd.read_csv(logs / f"fresh_seed{repeat}__{arm}.csv", float_precision="round_trip")
        origin = 6144
        prefix = signal.loc[signal["sample"] < origin, "neuron_2"].to_numpy()
        target = signal.loc[signal["sample"] >= origin, "neuron_2"].to_numpy()
        assert len(prefix) == 2048 and len(target) == 32
        selected = comparisons.query("seed == @repeat and arm == @arm").set_index("method")
        history = 96
        time = np.arange(-1, len(target))
        ax.plot(np.arange(-history, 0), prefix[-history:], color=GRAY, lw=1.05)
        ax.plot(time, np.r_[prefix[-1], target], color="#20262B", lw=1.4)
        for method, color, style, label in [
            ("MG", BLUE, "--", "MG@neuron activity"),
            ("validation", ORANGE, "-.", "Held-out search"),
        ]:
            model = selected.loc[method, "model"]
            prediction = forecast(prefix, len(target), model)
            error = np.mean((prediction - target)**2) / prefix.var()
            np.testing.assert_allclose(error, selected.loc[method, "error"], rtol=1e-9)
            ax.plot(time, np.r_[prefix[-1], prediction], color=color, ls=style, lw=1.35,
                    label=f"{label} ({model_names[model]})")
        ax.axvline(0, color=GRAY, lw=.75, ls=":")
        ax.axvspan(0, len(target), color="#DCE5EE", alpha=.35, zorder=0)
        ax.set(title=title, xlabel="Sample relative to forecast start", xlim=(-history, len(target)))
        ax.grid(axis="y", alpha=.15)
        ax.legend(loc="upper left", frameon=False, fontsize=6.6, handlelength=2)
    axes[0].set_ylabel("Neuron activity")
    for ax in axes:
        ax.set_ylim(-1.05, 1.68)
    fig.legend(handles=[
        Line2D([], [], color=GRAY, label="Observed activity"),
        Line2D([], [], color="#20262B", label="Measured continuation"),
    ], loc="outside lower center", ncol=2, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "axes.axisbelow": True})
    with threadpool_limits(limits=1):
        for name, plot in [
            ("walker2d_selection", plot_walker_selection),
            ("ssl_probe_scatter", plot_ssl),
            ("slingshot_warning", plot_slingshot),
            ("single_neuron_forecasts", plot_forecast),
        ]:
            fig = plot()
            fig.savefig(args.output / f"{name}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()

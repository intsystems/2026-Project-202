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
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from figure_style import figure_style
from plot_application_traces import BLUE, GRAY, ORANGE, PURPLE, EXPERIMENTS

FIGURES = Path(__file__).resolve().parent
DATA = FIGURES / "data"
SSL_SIZE = (5.9, 2.85)
SLINGSHOT_SIZE = (6.2, 3.05)
FORECAST_SIZE = (6.3, 2.9)
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
    fig, full = plt.subplots(figsize=SSL_SIZE, layout="constrained")
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
    detail.tick_params(length=2, pad=2)
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
    shown = relative.between(-650, 350)
    visible_steps = relative[shown]
    left, right = visible_steps.iloc[[0, -1]]
    scored = windows.step >= event["first_scored_step"]
    fig, (loss, scores) = plt.subplots(
        2, 1, sharex=True, figsize=SLINGSHOT_SIZE, layout="constrained",
        height_ratios=[1, 1.12],
    )
    fig.get_layout_engine().set(hspace=.1, w_pad=.07)
    for spike in event["onsets"]:
        recovery_end = spike + event["horizon"]
        scored &= ~windows.step.between(spike, recovery_end)
        start, end = max(left, spike - onset), min(right, recovery_end - onset)
        if start < end:
            scores.axvspan(start, end, color="#EDF0F2", lw=0, zorder=0)
            scores.text((start + end) / 2, .04, "Recovery", transform=scores.get_xaxis_transform(),
                        ha="center", va="bottom", color="#58646E")
    eligible = scored & relative.between(-event["horizon"], -1)

    loss.vlines(visible_steps, windows.min_log10_loss[shown], windows.max_log10_loss[shown],
                color="#9EABB4", lw=.85)
    loss.plot(visible_steps, windows.mean_log10_loss[shown], color="#566572", lw=1.1,
              marker="o", markersize=2.5)
    loss.set_ylabel(r"$\log_{10}$ loss")
    loss.yaxis.set_major_locator(MultipleLocator(1))
    loss.annotate("Spike onset", (0, 1), xycoords=loss.get_xaxis_transform(),
                  xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    scores.axhline(1, color="#69747C", ls="--", lw=.8)
    for method, signal, label, color, marker, offset, valign in (
        ("MG", windows.mg, "MG", BLUE, "o", (12, 13), "bottom"),
        ("roughness", windows.roughness, "Roughness", ORANGE, "D", (-20, -13), "top"),
    ):
        threshold = event["thresholds"][method]
        first_index = windows.index[eligible & (signal <= threshold)][0]
        first = relative.at[first_index]
        assert -first == event["leads"][method]
        ratio = signal / threshold
        scores.plot(visible_steps, ratio[shown], color=color, lw=1.15)
        value = ratio.at[first_index]
        scores.scatter(first, value, s=25, marker=marker, color=color,
                       edgecolors="white", linewidths=.55, zorder=4)
        scores.annotate(f"{label}\n{-first} steps before onset", (first, value),
                        xytext=offset, textcoords="offset points", ha="center",
                        va=valign, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=.65, shrinkB=3))

    for ax in (loss, scores):
        ax.axvline(0, color="#39444D", lw=.85)
        ax.grid(axis="y", color="#C9D0D5", lw=.5, alpha=.55)
    scores.set(xlabel="Training steps relative to spike onset", ylabel="Score / threshold",
               yticks=[.5, 1, 1.5], xticks=[-600, -400, -200, 0, 200],
               xlim=(left, right))
    fig.align_ylabels()
    return fig


def plot_forecast():
    logs = EXPERIMENTS / "model_selection_and_stopping/rnn_forecast_selection/logs"
    comparisons = pd.read_csv(logs / "query_fresh_comparisons.csv")
    model_names = {"knn8": "5-NN, 8 lags", "knn48": "5-NN, 48 lags", "ridge48": "Ridge, 48 lags"}
    rules = [
        ("MG", "MG-predicted error", BLUE, "--"),
        ("validation", "Validation error", ORANGE, (0, (1.2, 1.4))),
    ]
    signals = [("M4", "Two frequencies and their harmonics"), ("T4", "Four independent frequencies")]
    repeat = 681
    origin = 6144
    history_steps = 16
    actual_color = "#253039"
    fig = plt.figure(figsize=FORECAST_SIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=.07)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.15, 1), wspace=.07)
    axes = [fig.add_subplot(grid[0, col]) for col in range(2)]
    key = fig.add_subplot(grid[1, :])
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.set_axis_off()
    key_color = "#485660"
    key.text(.072, .64, "Select by", va="center", color=key_color)
    key.axhline(.51, color="#AEB9BF", lw=.6)
    for row, (_, criterion, color, style) in enumerate(rules):
        y = .34 - row * .24
        key.plot([0, .055], [y, y], color=color, ls=style, lw=1.55)
        key.text(.072, y, criterion, va="center")

    key_columns = [(.355, .61, "Mixed (M4)"), (.69, 1, "Independent (T4)")]
    for ax, (arm, title), (model_col, nmse_col, heading) in zip(axes, signals, key_columns):
        signal = pd.read_csv(logs / f"fresh_seed{repeat}__{arm}.csv", float_precision="round_trip")
        prefix = signal.loc[signal["sample"] < origin, "neuron_2"].to_numpy()
        target = signal.loc[signal["sample"] >= origin, "neuron_2"].to_numpy()
        assert len(prefix) == 2048 and len(target) == 32
        horizon = len(target)
        selected = comparisons.query("seed == @repeat and arm == @arm").set_index("method")
        best_rule = selected.loc[["MG", "validation"], "error"].idxmin()

        # Step zero is the last available observation. Both forecasts start from it.
        ax.axvspan(0, horizon, color="#EBF1F4", lw=0, zorder=0)
        ax.axvline(0, color=GRAY, lw=.7, ls=":")
        ax.plot(np.arange(-history_steps, horizon + 1), np.r_[prefix[-(history_steps + 1):], target],
                color=actual_color, lw=1.6)
        ax.text(1.1, .96, "Forecast", transform=ax.get_xaxis_transform(), va="top", color="#485660")
        ax.annotate("Recorded", (horizon, target[-1]), xytext=(-2, -4), textcoords="offset points",
                    ha="right", va="top", color=actual_color)
        ax.set(title=title, xlabel="Steps from last observation", xlim=(-history_steps, horizon),
               xticks=[-history_steps, 0, 16, horizon])
        ax.margins(y=.22)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.tick_params(length=3, pad=2)
        ax.grid(axis="y", color="#D6DCE0", lw=.55)

        key.text(.5 * (model_col + nmse_col), .90, heading, ha="center", va="center", color=key_color)
        key.text(model_col, .64, "Model", va="center", color=key_color)
        key.text(nmse_col, .64, "Test NMSE", ha="right", va="center", color=key_color)
        for row, (method, _, color, style) in enumerate(rules):
            model = selected.loc[method, "model"]
            prediction = forecast(prefix, horizon, model)
            error = np.mean((prediction - target)**2) / prefix.var()
            np.testing.assert_allclose(error, selected.loc[method, "error"], rtol=1e-9)
            ax.plot(np.arange(horizon + 1), np.r_[prefix[-1], prediction], color=color, ls=style, lw=1.55)

            y = .34 - row * .24
            key.text(model_col, y, model_names[model], va="center")
            key.text(nmse_col, y, f"{error:.3f}", ha="right", va="center",
                     fontweight="bold" if method == best_rule else "normal")
    axes[0].set_ylabel("Neuron activation")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        # Walker2d is a standalone plot, not placed in the article.
        with matplotlib.rc_context({
            "font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": .6, "axes.axisbelow": True,
        }):
            fig = plot_walker_selection()
            fig.savefig(args.output / "walker2d_selection.pdf")
            plt.close(fig)
        for name, plot, size, fraction in [
            ("ssl_probe_scatter", plot_ssl, SSL_SIZE, .86),
            ("slingshot_warning", plot_slingshot, SLINGSHOT_SIZE, .92),
            ("single_neuron_forecasts", plot_forecast, FORECAST_SIZE, 1),
        ]:
            with figure_style(size[0], fraction):
                fig = plot()
                fig.savefig(args.output / f"{name}.pdf")
                plt.close(fig)


if __name__ == "__main__":
    main()

"""EDM control analysis for sharp generalization without a memorization gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
GROKKING_DIR = HERE.parent
sys.path.insert(0, str(GROKKING_DIR))

from grokking_analysis import get_E_cao, get_E_fnn, get_E_mle, get_E_simplex  # noqa: E402


METRICS = (
    "train_loss",
    "val_loss",
    "weight_norm",
    "grad_norm",
    "embed_grad_norm",
    "grad_cosine",
)
METHODS = {
    "FNN": get_E_fnn,
    "Cao": get_E_cao,
    "Simplex": get_E_simplex,
    "MLE": get_E_mle,
}
COLORS = {
    "FNN": "#2563EB",
    "Cao": "#059669",
    "Simplex": "#D97706",
    "MLE": "#7C3AED",
}


def linear_detrend(series: np.ndarray) -> np.ndarray:
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return series - (slope * x + intercept)


def first_stable_step(frame: pd.DataFrame, metric: str, threshold: float, patience: int) -> int | None:
    values = frame[metric].to_numpy(dtype=float) >= threshold
    for start in range(0, len(values) - patience + 1):
        if values[start:start + patience].all():
            return int(frame["step"].iloc[start + patience - 1])
    return None


def transition_metadata(frame: pd.DataFrame) -> dict:
    result = {
        "logging_step": int(np.median(np.diff(frame["step"]))),
        "train_acc_095_stable5": first_stable_step(frame, "train_acc", 0.95, 5),
        "val_acc_050_stable5": first_stable_step(frame, "val_acc", 0.50, 5),
        "val_acc_090_stable5": first_stable_step(frame, "val_acc", 0.90, 5),
        "val_acc_095_stable5": first_stable_step(frame, "val_acc", 0.95, 5),
        "last_step": int(frame["step"].iloc[-1]),
    }
    train = result["train_acc_095_stable5"]
    validation = result["val_acc_090_stable5"]
    result["val090_minus_train095_steps"] = None if train is None or validation is None else validation - train
    result["points_after_val095"] = (
        None if result["val_acc_095_stable5"] is None else
        int((frame["step"] >= result["val_acc_095_stable5"]).sum())
    )
    return result


def estimate(series: np.ndarray) -> dict[str, float]:
    estimates = {}
    for index, (name, function) in enumerate(METHODS.items()):
        np.random.seed(20260801 + index)
        value = function(series, 1)
        estimates[name] = float(value) if value is not None and np.isfinite(value) else np.nan
    return estimates


def sliding_analysis(frame: pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
    rows = []
    steps = frame["step"].to_numpy(dtype=int)
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        for start in range(0, len(values) - window_size + 1, stride):
            stop = start + window_size
            raw = values[start:stop]
            detrended = linear_detrend(raw)
            raw_estimates = estimate(raw)
            detrended_estimates = estimate(detrended)
            row = {
                "metric": metric,
                "window_size": window_size,
                "stride": stride,
                "tau": 1,
                "start_step": int(steps[start]),
                "center_step": int(steps[start + window_size // 2]),
                "end_step": int(steps[stop - 1]),
                "mean": float(raw.mean()),
                "std": float(raw.std()),
                "lag1": float(np.corrcoef(raw[:-1], raw[1:])[0, 1]),
            }
            for method in METHODS:
                row[f"{method.lower()}_raw"] = raw_estimates[method]
                row[f"{method.lower()}_detrended"] = detrended_estimates[method]
            rows.append(row)
    return pd.DataFrame(rows)


def phase_summary(windows: pd.DataFrame, transition: dict) -> pd.DataFrame:
    # Fully pre-transition windows end before validation first reaches 10%.
    # Transition windows are the last five available windows and necessarily
    # mix the rapid transition with its immediate approach.
    rows = []
    for metric, group in windows.groupby("metric", sort=False):
        group = group.sort_values("center_step")
        pre_candidates = group[group["end_step"] < 42_500]
        pre = pre_candidates.tail(5)
        transition_windows = group.tail(5)
        for method in METHODS:
            for series in ("raw", "detrended"):
                column = f"{method.lower()}_{series}"
                pre_mean = float(pre[column].mean())
                transition_mean = float(transition_windows[column].mean())
                rows.append({
                    "metric": metric,
                    "method": method,
                    "series": series,
                    "pre_windows": len(pre),
                    "transition_windows": len(transition_windows),
                    "pre_end_step_max": int(pre["end_step"].max()),
                    "transition_end_step_min": int(transition_windows["end_step"].min()),
                    "pre_mean": pre_mean,
                    "transition_mean": transition_mean,
                    "transition_minus_pre": transition_mean - pre_mean,
                })
    return pd.DataFrame(rows)


def mle_robustness(frame: pd.DataFrame, window_sizes: tuple[int, ...] = (50, 100, 150)) -> pd.DataFrame:
    rows = []
    steps = frame["step"].to_numpy(dtype=int)
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        for window_size in window_sizes:
            stride = max(5, window_size // 10)
            for start in range(0, len(values) - window_size + 1, stride):
                stop = start + window_size
                raw = values[start:stop]
                detrended = linear_detrend(raw)
                np.random.seed(20260811)
                raw_mle = get_E_mle(raw, 1)
                np.random.seed(20260812)
                detrended_mle = get_E_mle(detrended, 1)
                rows.append({
                    "metric": metric, "window_size": window_size, "stride": stride,
                    "start_step": int(steps[start]),
                    "center_step": int(steps[start + window_size // 2]),
                    "end_step": int(steps[stop - 1]),
                    "mle_raw": raw_mle, "mle_detrended": detrended_mle,
                })
    return pd.DataFrame(rows)


def add_transition_markers(axis: plt.Axes, transition: dict) -> None:
    axis.axvline(transition["val_acc_050_stable5"], color="#DC2626", linestyle="--", linewidth=1.2)
    axis.axvline(transition["val_acc_090_stable5"], color="#7F1D1D", linestyle=":", linewidth=1.4)


def plot_overview(frame: pd.DataFrame, transition: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame["step"], frame["train_loss"], label="train loss", color="#2563EB", alpha=0.75)
    axes[0].plot(frame["step"], frame["val_loss"], label="validation loss", color="#DC2626")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[1].plot(frame["step"], frame["train_acc"], label="batch train accuracy", color="#2563EB", alpha=0.75)
    axes[1].plot(frame["step"], frame["val_acc"], label="validation accuracy", color="#DC2626")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)
    for axis in axes:
        add_transition_markers(axis, transition)
        axis.grid(alpha=0.22)
    axes[0].set_title("Sharp generalization without a memorization gap")
    fig.tight_layout()
    fig.savefig(output / "01_training_transition.png", dpi=220)
    fig.savefig(output / "01_training_transition.pdf")
    plt.close(fig)


def plot_loss_methods(windows: pd.DataFrame, transition: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis, method in zip(axes.flat, METHODS):
        for metric, style in (("train_loss", "-"), ("val_loss", "--")):
            group = windows[windows["metric"] == metric]
            axis.plot(group["center_step"], group[f"{method.lower()}_raw"],
                      color=COLORS[method], linestyle=style, linewidth=1.7,
                      label=metric.replace("_", " "))
        add_transition_markers(axis, transition)
        axis.set_title(method)
        axis.set_ylabel("estimated dimension")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("window center (step)")
    fig.suptitle("Four fixed-$\\tau=1$ EDM diagnostics on loss observables (W=100)")
    fig.tight_layout()
    fig.savefig(output / "02_four_methods_loss.png", dpi=220)
    fig.savefig(output / "02_four_methods_loss.pdf")
    plt.close(fig)


def plot_mle_metrics(windows: pd.DataFrame, transition: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        group = windows[windows["metric"] == metric]
        axis.plot(group["center_step"], group["mle_raw"], color="#7C3AED", label="raw")
        axis.plot(group["center_step"], group["mle_detrended"], color="#4B5563",
                  linestyle="--", label="detrended")
        add_transition_markers(axis, transition)
        axis.set_title(metric.replace("_", " "))
        axis.set_ylabel("MLE ID")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("window center (step)")
    fig.suptitle("Levina--Bickel MLE: raw versus locally detrended (W=100)")
    fig.tight_layout()
    fig.savefig(output / "03_mle_all_metrics.png", dpi=220)
    fig.savefig(output / "03_mle_all_metrics.pdf")
    plt.close(fig)


def plot_delta_heatmap(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    vmax = max(1.0, float(summary["transition_minus_pre"].abs().quantile(0.95)))
    image = None
    for axis, series in zip(axes, ("raw", "detrended")):
        subset = summary[summary["series"] == series]
        matrix = np.array([
            [subset[(subset["metric"] == metric) & (subset["method"] == method)]["transition_minus_pre"].iloc[0]
             for method in METHODS]
            for metric in METRICS
        ])
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(METHODS)), METHODS, rotation=25)
        axis.set_title(series)
        for row in range(len(METRICS)):
            for column in range(len(METHODS)):
                value = matrix[row, column]
                axis.text(column, row, f"{value:+.1f}", ha="center", va="center",
                          color="white" if abs(value) > 0.55 * vmax else "black", fontsize=8)
    axes[0].set_yticks(range(len(METRICS)), [metric.replace("_", " ") for metric in METRICS])
    fig.colorbar(image, ax=axes, label="transition-window mean minus pre-transition mean", shrink=0.78)
    fig.suptitle("EDM changes near the accuracy transition")
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.14, top=0.88, wspace=0.12)
    fig.savefig(output / "04_method_delta_heatmap.png", dpi=220)
    fig.savefig(output / "04_method_delta_heatmap.pdf")
    plt.close(fig)


def plot_mle_robustness(robustness: pd.DataFrame, transition: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, metric in zip(axes, ("train_loss", "val_loss")):
        for window_size in sorted(robustness["window_size"].unique()):
            group = robustness[(robustness["metric"] == metric) & (robustness["window_size"] == window_size)]
            axis.plot(group["center_step"], group["mle_detrended"], label=f"W={window_size}")
        add_transition_markers(axis, transition)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("window center (step)")
        axis.set_ylabel("detrended MLE ID")
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("MLE window-length sensitivity")
    fig.tight_layout()
    fig.savefig(output / "05_mle_window_robustness.png", dpi=220)
    fig.savefig(output / "05_mle_window_robustness.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input_csv).sort_values("step").reset_index(drop=True)
    transition = transition_metadata(frame)
    windows = sliding_analysis(frame, args.window_size, args.stride)
    summary = phase_summary(windows, transition)
    robustness = mle_robustness(frame)
    windows.to_csv(args.output_dir / "tau1_all_methods_windows.csv", index=False)
    summary.to_csv(args.output_dir / "pre_vs_transition_summary.csv", index=False)
    robustness.to_csv(args.output_dir / "mle_window_robustness.csv", index=False)
    (args.output_dir / "transition_metadata.json").write_text(
        json.dumps(transition, indent=2), encoding="utf-8"
    )
    plot_overview(frame, transition, args.output_dir)
    plot_loss_methods(windows, transition, args.output_dir)
    plot_mle_metrics(windows, transition, args.output_dir)
    plot_delta_heatmap(summary, args.output_dir)
    plot_mle_robustness(robustness, transition, args.output_dir)
    print(json.dumps(transition, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

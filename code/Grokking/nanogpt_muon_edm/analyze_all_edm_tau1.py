"""Run all four legacy project EDM dimension methods with fixed tau=1.

The four estimators intentionally call the implementations from
``code/Grokking/search_for_optimal_parameters.py`` so that the results are
directly comparable with the project's existing grokking notebooks.
"""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "code" / "Grokking"))
from search_for_optimal_parameters import (  # noqa: E402
    cao_method,
    false_nearest_neighbors,
    find_optimal_E_simplex,
    mle_intrinsic_dimension,
)


TAU = 1
WINDOW_SIZE = 400
STRIDE = 40
MAX_E = 15
COLORS = {"FNN": "#1976D2", "Cao": "#00897B", "Simplex": "#E67E22", "MLE": "#7B1FA2"}
METHODS = ("FNN", "Cao", "Simplex", "MLE")


def linear_detrend(series: np.ndarray) -> np.ndarray:
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return series - (slope * x + intercept)


def estimate_dimensions(series: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Use the four original project estimators with their notebook settings."""
    # FNN and Cao add tiny random jitter internally; fixed state makes the report reproducible.
    np.random.seed(20260730)
    fnn_curve = false_nearest_neighbors(series, tau=TAU, max_m=MAX_E)
    below_threshold = np.flatnonzero(fnn_curve < 1.0)
    fnn_dimension = float(below_threshold[0] + 1 if len(below_threshold) else np.argmin(fnn_curve) + 1)

    np.random.seed(20260731)
    cao_dimension, cao_curve = cao_method(series, tau=TAU, max_E=MAX_E)
    simplex_dimension, simplex_curve = find_optimal_E_simplex(series, tau=TAU, max_E=MAX_E, Tp=1)

    np.random.seed(20260732)
    mle_dimension = mle_intrinsic_dimension(series, tau=TAU, max_E=MAX_E, k_neighbors=5)

    dimensions = {
        "FNN": fnn_dimension,
        "Cao": float(cao_dimension),
        "Simplex": float(simplex_dimension),
        "MLE": float(mle_dimension),
    }
    curves = {"fnn_percent": fnn_curve, "cao_e1": np.asarray(cao_curve), "simplex_rmse": np.asarray(simplex_curve)}
    return dimensions, curves


def sliding_analysis(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    steps = train["step"].to_numpy(dtype=int)
    loss = train["train_loss_per_token"].to_numpy(dtype=float)
    rows: list[dict] = []
    representative: dict[str, dict[str, np.ndarray]] = {}

    target_centers = {"early": 300, "middle": 1500, "late": 2130}
    for start in range(0, len(loss) - WINDOW_SIZE + 1, STRIDE):
        stop = start + WINDOW_SIZE
        raw = loss[start:stop]
        detrended = linear_detrend(raw)
        raw_dimensions, raw_curves = estimate_dimensions(raw)
        detrended_dimensions, detrended_curves = estimate_dimensions(detrended)
        center_step = int(steps[start + WINDOW_SIZE // 2])
        row = {
            "start_step": int(steps[start]),
            "end_step": int(steps[stop - 1]),
            "center_step": center_step,
            "tau": TAU,
            "loss_mean": float(np.mean(raw)),
            "loss_std": float(np.std(raw)),
            "lag1_autocorrelation": float(np.corrcoef(raw[:-1], raw[1:])[0, 1]),
        }
        for method in METHODS:
            row[f"{method.lower()}_raw"] = raw_dimensions[method]
            row[f"{method.lower()}_detrended"] = detrended_dimensions[method]
        rows.append(row)

        for phase, target in target_centers.items():
            if phase not in representative or abs(center_step - target) < abs(representative[phase]["center_step"] - target):
                representative[phase] = {
                    "center_step": center_step,
                    "start_step": int(steps[start]),
                    "end_step": int(steps[stop - 1]),
                    "raw_curves": raw_curves,
                    "detrended_curves": detrended_curves,
                }
    return pd.DataFrame(rows), representative


def add_schedule_markers(axis: plt.Axes, metadata: dict) -> None:
    cooldown_start = metadata["num_iterations"] * 0.55
    axis.axvline(cooldown_start, color="#6B7280", linestyle="--", linewidth=1.1)
    axis.axvline(metadata["num_iterations"], color="#6B7280", linestyle=":", linewidth=1.1)


def plot_dimensions(results: pd.DataFrame, metadata: dict, output_dir: Path, suffix: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    x = results["center_step"]
    for axis, method in zip(axes.flat, METHODS):
        column = f"{method.lower()}_{suffix}"
        axis.plot(x, results[column], color=COLORS[method], marker="o", markersize=3.6, linewidth=1.9)
        axis.set_title(method)
        axis.set_ylabel("estimated dimension $E$")
        axis.set_ylim(0.5, MAX_E + 0.7)
        axis.grid(alpha=0.2)
        add_schedule_markers(axis, metadata)
    for axis in axes[-1]:
        axis.set_xlabel("window center (optimization step)")
    fig.suptitle(title, y=0.99)
    fig.text(0.5, 0.012, "dashed: start of LR cooldown; dotted: extension start", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    stem = f"tau1_all_methods_{suffix}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    x = results["center_step"]
    for axis, method in zip(axes.flat, METHODS):
        axis.plot(x, results[f"{method.lower()}_raw"], color=COLORS[method], marker="o", markersize=3.4, linewidth=1.7, label="raw loss")
        axis.plot(x, results[f"{method.lower()}_detrended"], color="#4B5563", marker="o", markersize=3.1, linewidth=1.5, linestyle="--", label="detrended loss")
        axis.set_title(method)
        axis.set_ylabel("estimated dimension $E$")
        axis.set_ylim(0.5, MAX_E + 0.7)
        axis.grid(alpha=0.2)
        add_schedule_markers(axis, metadata)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("window center (optimization step)")
    fig.suptitle("Fixed $\\tau=1$: raw loss versus locally detrended loss", y=0.99)
    fig.text(0.5, 0.012, "dashed: start of LR cooldown; dotted: extension start", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_dir / "tau1_raw_vs_detrended.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "tau1_raw_vs_detrended.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_method_curves(representative: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.2))
    phase_order = ("early", "middle", "late")
    for row, phase in enumerate(phase_order):
        item = representative[phase]
        curves = item["raw_curves"]
        x_fnn = np.arange(1, len(curves["fnn_percent"]) + 1)
        x_cao = np.arange(1, len(curves["cao_e1"]) + 1)
        x_simplex = np.arange(1, len(curves["simplex_rmse"]) + 1)
        axes[row, 0].plot(x_fnn, curves["fnn_percent"], color=COLORS["FNN"], marker="o")
        axes[row, 0].axhline(1.0, color="#4B5563", linestyle="--", linewidth=1)
        axes[row, 0].set_ylabel(f"{phase}\nFNN, %")
        axes[row, 1].plot(x_cao, curves["cao_e1"], color=COLORS["Cao"], marker="o")
        axes[row, 1].set_ylabel("Cao $E_1$")
        axes[row, 2].plot(x_simplex, curves["simplex_rmse"], color=COLORS["Simplex"], marker="o")
        axes[row, 2].set_ylabel("simplex RMSE")
        for axis in axes[row]:
            axis.grid(alpha=0.2)
            axis.set_xlabel("embedding dimension $E$")
            axis.set_title(f"steps {item['start_step']}–{item['end_step']}")
    fig.suptitle("Underlying legacy-method criteria, fixed $\\tau=1$ (raw loss)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "tau1_method_criteria.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "tau1_method_criteria.pdf", bbox_inches="tight")
    plt.close(fig)


def make_summary(results: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    work = results.copy()
    cooldown_start = metadata["num_iterations"] * 0.55
    work["schedule_phase"] = np.where(work["center_step"] < cooldown_start, "constant LR", "LR cooldown")
    aggregations: dict[str, tuple[str, str]] = {
        "windows": ("center_step", "count"),
        "step_min": ("center_step", "min"),
        "step_max": ("center_step", "max"),
    }
    for method in METHODS:
        for kind in ("raw", "detrended"):
            column = f"{method.lower()}_{kind}"
            aggregations[f"mean_{column}"] = (column, "mean")
            aggregations[f"sd_{column}"] = (column, "std")
    summary = work.groupby("schedule_phase", sort=False).agg(**aggregations).reset_index()
    early = work.iloc[:12]
    late = work.iloc[-12:]
    changes = {
        method: {
            kind: {
                "early_mean": float(early[f"{method.lower()}_{kind}"].mean()),
                "late_mean": float(late[f"{method.lower()}_{kind}"].mean()),
                "late_minus_early": float(late[f"{method.lower()}_{kind}"].mean() - early[f"{method.lower()}_{kind}"].mean()),
            }
            for kind in ("raw", "detrended")
        }
        for method in METHODS
    }
    return summary, {"tau": TAU, "window_size": WINDOW_SIZE, "stride": STRIDE, "max_E": MAX_E, "early_late_changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parsed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.parsed_dir / "train_log.csv")
    with (args.parsed_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    results, representative = sliding_analysis(train)
    results.to_csv(args.output_dir / "tau1_all_methods_windows.csv", index=False)
    summary, diagnostics = make_summary(results, metadata)
    summary.to_csv(args.output_dir / "tau1_all_methods_summary.csv", index=False)
    with (args.output_dir / "tau1_all_methods_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    plot_dimensions(results, metadata, args.output_dir, "raw", "All four legacy EDM methods, fixed $\\tau=1$ (raw train loss)")
    plot_dimensions(results, metadata, args.output_dir, "detrended", "All four legacy EDM methods, fixed $\\tau=1$ (locally detrended train loss)")
    plot_comparison(results, metadata, args.output_dir)
    plot_method_curves(representative, args.output_dir)
    print(summary.to_string(index=False))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

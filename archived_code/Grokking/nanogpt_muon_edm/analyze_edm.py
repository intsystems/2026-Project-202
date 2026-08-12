"""EDM diagnostics for the parsed Muon/nanoGPT loss trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors


COLORS = {
    "train": "#2455A4",
    "val": "#C6413A",
    "id": "#6A3D9A",
    "id_detrended": "#E68613",
    "tau": "#16817A",
    "schedule": "#6B7280",
}


def delay_embedding(series: np.ndarray, dimension: int, tau: int) -> np.ndarray:
    n = len(series) - (dimension - 1) * tau
    if n <= 0:
        raise ValueError("Series is too short for the requested embedding")
    return np.column_stack([series[i * tau : i * tau + n] for i in range(dimension)])


def linear_detrend(series: np.ndarray) -> np.ndarray:
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return series - (slope * x + intercept)


def standardize(series: np.ndarray) -> np.ndarray:
    std = float(np.std(series))
    if std < 1e-12:
        return np.zeros_like(series, dtype=float)
    return (series - np.mean(series)) / std


def estimate_tau(series: np.ndarray, max_tau: int = 20) -> tuple[int, np.ndarray]:
    """Select the first local minimum of delayed mutual information."""
    x = standardize(series)
    values = []
    for tau in range(1, max_tau + 1):
        # kNN MI is less dependent on an arbitrary histogram bin count.
        mi = mutual_info_regression(
            x[:-tau, None], x[tau:], n_neighbors=5, random_state=0
        )[0]
        values.append(float(mi))
    values_array = np.asarray(values)
    for i in range(1, len(values_array) - 1):
        if values_array[i] <= values_array[i - 1] and values_array[i] < values_array[i + 1]:
            return i + 1, values_array
    return int(np.argmin(values_array)) + 1, values_array


def levina_bickel_id(
    series: np.ndarray,
    tau: int,
    embedding_dimension: int = 12,
    k_neighbors: int = 10,
    theiler: int = 12,
) -> tuple[float, float, int]:
    """Estimate intrinsic dimension while excluding temporal neighbours.

    Returns the trimmed mean, standard error across local estimates, and the
    number of local estimates retained.
    """
    embedded = delay_embedding(standardize(series), embedding_dimension, tau)
    if len(embedded) < k_neighbors + 2 * theiler + 5:
        return math.nan, math.nan, 0

    query_k = min(len(embedded), k_neighbors + 2 * theiler + 30)
    model = NearestNeighbors(n_neighbors=query_k, algorithm="auto").fit(embedded)
    distances, indices = model.kneighbors(embedded)
    local_ids: list[float] = []
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        keep = np.abs(row_indices - row) > theiler
        d = row_distances[keep][:k_neighbors]
        if len(d) < k_neighbors or d[-1] <= 1e-12:
            continue
        d = np.maximum(d, 1e-12)
        denom = np.log(d[-1] / d[:-1]).sum()
        if denom > 1e-12:
            local_ids.append((k_neighbors - 1) / denom)

    local = np.asarray(local_ids)
    local = local[np.isfinite(local)]
    if len(local) < 10:
        return math.nan, math.nan, int(len(local))
    low, high = np.quantile(local, [0.05, 0.95])
    trimmed = local[(local >= low) & (local <= high)]
    return float(np.mean(trimmed)), float(np.std(trimmed, ddof=1) / np.sqrt(len(trimmed))), int(len(trimmed))


def legacy_project_id(
    series: np.ndarray,
    tau: int = 1,
    embedding_dimension: int = 15,
    k_neighbors: int = 5,
) -> float:
    """Reproduce the estimator in search_for_optimal_parameters.py."""
    embedded = delay_embedding(np.asarray(series, dtype=float), embedding_dimension, tau)
    model = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(embedded)
    distances, _ = model.kneighbors(embedded)
    distances = np.maximum(distances[:, 1:], 1e-8)
    outer = distances[:, -1:]
    denominator = np.log(outer / distances[:, :-1]).sum(axis=1)
    denominator = np.maximum(denominator, 1e-5)
    local = (k_neighbors - 1) / denominator
    local = local[np.isfinite(local)]
    estimate = float(np.mean(local)) if len(local) else math.nan
    return float(embedding_dimension) if estimate > 2 * embedding_dimension else estimate


def sliding_analysis(
    steps: np.ndarray,
    loss: np.ndarray,
    window_size: int = 400,
    stride: int = 40,
    max_tau: int = 20,
    embedding_dimension: int = 12,
    k_neighbors: int = 10,
    theiler: int = 12,
) -> pd.DataFrame:
    rows = []
    for start in range(0, len(loss) - window_size + 1, stride):
        stop = start + window_size
        window = loss[start:stop]
        tau, _ = estimate_tau(window, max_tau=max_tau)
        raw_id, raw_se, raw_n = levina_bickel_id(
            window, tau, embedding_dimension, k_neighbors, theiler
        )
        residual = linear_detrend(window)
        residual_id, residual_se, residual_n = levina_bickel_id(
            residual, tau, embedding_dimension, k_neighbors, theiler
        )
        legacy_id = legacy_project_id(window)
        x = np.arange(window_size, dtype=float)
        slope = float(np.polyfit(x, window, 1)[0])
        lag1 = float(np.corrcoef(window[:-1], window[1:])[0, 1])
        rows.append(
            {
                "start_step": int(steps[start]),
                "end_step": int(steps[stop - 1]),
                "center_step": int(steps[start + window_size // 2]),
                "tau": tau,
                "mle_id_raw": raw_id,
                "mle_id_raw_se": raw_se,
                "mle_id_detrended": residual_id,
                "mle_id_detrended_se": residual_se,
                "mle_id_legacy_project": legacy_id,
                "raw_local_estimates": raw_n,
                "detrended_local_estimates": residual_n,
                "loss_mean": float(np.mean(window)),
                "loss_std": float(np.std(window)),
                "loss_slope_per_step": slope,
                "lag1_autocorrelation": lag1,
            }
        )
    return pd.DataFrame(rows)


def phase_label(step: float, metadata: dict) -> str:
    cooldown_start = metadata["num_iterations"] * (1 - 0.45)
    if step < cooldown_start:
        return "constant LR"
    if step < metadata["num_iterations"]:
        return "LR cooldown"
    return "extension"


def add_schedule_markers(axis: plt.Axes, metadata: dict) -> None:
    cooldown_start = metadata["num_iterations"] * (1 - 0.45)
    axis.axvline(cooldown_start, color=COLORS["schedule"], linestyle="--", linewidth=1.2)
    axis.axvline(metadata["num_iterations"], color=COLORS["schedule"], linestyle=":", linewidth=1.2)


def make_overview(train: pd.DataFrame, val: pd.DataFrame, metadata: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    smooth = train["train_loss_per_token"].rolling(41, center=True, min_periods=1).median()
    ax1.plot(train["step"], train["train_loss_per_token"], color=COLORS["train"], alpha=0.20, linewidth=0.7, label="train loss (per token)")
    ax1.plot(train["step"], smooth, color=COLORS["train"], linewidth=2.0, label="41-step rolling median")
    ax1.plot(val["step"], val["val_loss"], color=COLORS["val"], marker="o", linewidth=1.8, label="validation loss")
    add_schedule_markers(ax1, metadata)
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_yscale("log")
    ax1.legend(frameon=False, ncol=3, fontsize=9)
    ax1.grid(alpha=0.18)
    ax1.set_title("Muon / nanoGPT training trajectory")

    ax2.plot(train["step"], train["lr_multiplier"], color=COLORS["schedule"], linewidth=2)
    add_schedule_markers(ax2, metadata)
    ax2.set_ylabel("LR multiplier")
    ax2.set_xlabel("optimization step")
    ax2.set_ylim(0, 1.08)
    ax2.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "01_training_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "01_training_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def make_edm_trajectory(results: pd.DataFrame, metadata: dict, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    x = results["center_step"].to_numpy()
    for column, se_column, color, label in [
        ("mle_id_raw", "mle_id_raw_se", COLORS["id"], "raw train-loss series"),
        ("mle_id_detrended", "mle_id_detrended_se", COLORS["id_detrended"], "locally detrended series"),
    ]:
        y = results[column].to_numpy()
        se = results[se_column].to_numpy()
        axes[0].plot(x, y, marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
        axes[0].fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.13)
    axes[0].plot(
        x,
        results["mle_id_legacy_project"],
        color="#4B5563",
        linestyle="--",
        linewidth=1.4,
        label="legacy project estimator",
    )
    axes[0].set_ylabel("Levina–Bickel ID")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.18)
    axes[0].set_title("Sliding-window EDM diagnostics (W=400, stride=40)")

    axes[1].step(x, results["tau"], where="mid", color=COLORS["tau"], linewidth=1.8)
    axes[1].set_ylabel("DMI lag $\\tau$")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(alpha=0.18)

    axes[2].plot(x, results["loss_std"], color=COLORS["train"], linewidth=1.8, label="within-window SD")
    twin = axes[2].twinx()
    twin.plot(x, results["lag1_autocorrelation"], color=COLORS["val"], linewidth=1.5, label="lag-1 autocorrelation")
    axes[2].set_ylabel("loss SD")
    twin.set_ylabel("lag-1 ACF", color=COLORS["val"])
    axes[2].set_xlabel("window center (optimization step)")
    axes[2].grid(alpha=0.18)

    for axis in axes:
        add_schedule_markers(axis, metadata)
    fig.tight_layout()
    fig.savefig(out / "02_edm_trajectory.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "02_edm_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def make_phase_portraits(train: pd.DataFrame, results: pd.DataFrame, metadata: dict, out: Path) -> None:
    target_centers = [300, 1500, 2130]
    loss = train["train_loss_per_token"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for col, target in enumerate(target_centers):
        row = results.iloc[(results["center_step"] - target).abs().argmin()]
        start, end, tau = int(row["start_step"]), int(row["end_step"]), int(row["tau"])
        segment = standardize(loss[start : end + 1])
        residual = standardize(linear_detrend(loss[start : end + 1]))
        for axis, values, kind in [(axes[0, col], segment, "raw"), (axes[1, col], residual, "detrended")]:
            axis.plot(values[:-tau], values[tau:], color=COLORS["train"], alpha=0.5, linewidth=0.7)
            axis.scatter(values[:-tau], values[tau:], c=np.arange(len(values) - tau), cmap="viridis", s=7, alpha=0.75)
            axis.set_xlabel("$x_t$ (standardized)")
            if col == 0:
                axis.set_ylabel(f"{kind}: $x_{{t+\\tau}}$")
            axis.grid(alpha=0.15)
        axes[0, col].set_title(f"{phase_label(row['center_step'], metadata)}\nsteps {start}–{end}, $\\tau$={tau}")
    fig.suptitle("Delay-coordinate portraits across training", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "03_phase_portraits.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "03_phase_portraits.pdf", bbox_inches="tight")
    plt.close(fig)


def robustness_table(train: pd.DataFrame) -> pd.DataFrame:
    loss = train["train_loss_per_token"].to_numpy()
    phases = {"early": 300, "middle": 1500, "late": 2130}
    rows = []
    for phase, center in phases.items():
        for window_size in [300, 400, 500, 600]:
            half = window_size // 2
            start = max(0, min(len(loss) - window_size, center - half))
            segment = loss[start : start + window_size]
            for tau in [1, 2, 3, 5, 8]:
                raw_id, _, _ = levina_bickel_id(segment, tau, 12, 10, 12)
                det_id, _, _ = levina_bickel_id(linear_detrend(segment), tau, 12, 10, 12)
                rows.append({"phase": phase, "center_step": center, "window_size": window_size, "tau": tau, "mle_id_raw": raw_id, "mle_id_detrended": det_id})
    return pd.DataFrame(rows)


def embedding_sensitivity_table(train: pd.DataFrame) -> pd.DataFrame:
    loss = train["train_loss_per_token"].to_numpy()
    phases = {"early": 300, "middle": 1500, "late": 2130}
    rows = []
    window_size = 400
    for phase, center in phases.items():
        start = max(0, min(len(loss) - window_size, center - window_size // 2))
        segment = loss[start : start + window_size]
        selected_tau, _ = estimate_tau(segment, max_tau=20)
        for embedding_dimension in [4, 6, 8, 10, 12, 15]:
            for tau in sorted({1, selected_tau}):
                raw_id, raw_se, _ = levina_bickel_id(
                    segment, tau, embedding_dimension, 10, 12
                )
                det_id, det_se, _ = levina_bickel_id(
                    linear_detrend(segment), tau, embedding_dimension, 10, 12
                )
                rows.append(
                    {
                        "phase": phase,
                        "center_step": center,
                        "embedding_dimension": embedding_dimension,
                        "tau": tau,
                        "tau_source": "DMI" if tau == selected_tau else "fixed",
                        "mle_id_raw": raw_id,
                        "mle_id_raw_se": raw_se,
                        "mle_id_detrended": det_id,
                        "mle_id_detrended_se": det_se,
                    }
                )
    return pd.DataFrame(rows)


def make_robustness(robustness: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.7), sharex=True, sharey="row")
    for col, phase in enumerate(["early", "middle", "late"]):
        subset = robustness[robustness["phase"] == phase]
        for row, (metric, title) in enumerate([("mle_id_raw", "raw"), ("mle_id_detrended", "detrended")]):
            pivot = subset.pivot(index="window_size", columns="tau", values=metric)
            image = axes[row, col].imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="magma")
            axes[row, col].set_xticks(range(len(pivot.columns)), pivot.columns)
            axes[row, col].set_yticks(range(len(pivot.index)), pivot.index)
            axes[row, col].set_title(f"{phase}: {title}")
            axes[row, col].set_xlabel("lag $\\tau$")
            if col == 0:
                axes[row, col].set_ylabel("window size")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    axes[row, col].text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.03)
    fig.suptitle("Sensitivity of the intrinsic-dimension estimate", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "04_robustness.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "04_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def make_embedding_sensitivity(sensitivity: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for axis, phase in zip(axes, ["early", "middle", "late"]):
        subset = sensitivity[
            (sensitivity["phase"] == phase) & (sensitivity["tau_source"] == "DMI")
        ]
        x = subset["embedding_dimension"].to_numpy()
        axis.plot(x, subset["mle_id_raw"], marker="o", color=COLORS["id"], label="raw")
        axis.plot(x, subset["mle_id_detrended"], marker="o", color=COLORS["id_detrended"], label="detrended")
        axis.plot(x, x, color=COLORS["schedule"], linestyle=":", linewidth=1, label="$d=m$")
        tau = int(subset["tau"].iloc[0])
        axis.set_title(f"{phase}, DMI $\\tau$={tau}")
        axis.set_xlabel("embedding dimension $m$")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Levina–Bickel ID")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dependence on reconstruction-space dimension", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "05_embedding_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "05_embedding_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(results: pd.DataFrame, robustness: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    results = results.copy()
    results["schedule_phase"] = results["center_step"].map(lambda step: phase_label(step, metadata))
    summary = results.groupby("schedule_phase", sort=False).agg(
        windows=("center_step", "count"),
        step_min=("center_step", "min"),
        step_max=("center_step", "max"),
        median_tau=("tau", "median"),
        mean_id_raw=("mle_id_raw", "mean"),
        sd_id_raw=("mle_id_raw", "std"),
        mean_id_detrended=("mle_id_detrended", "mean"),
        sd_id_detrended=("mle_id_detrended", "std"),
        mean_id_legacy=("mle_id_legacy_project", "mean"),
        sd_id_legacy=("mle_id_legacy_project", "std"),
        mean_loss=("loss_mean", "mean"),
        mean_loss_std=("loss_std", "mean"),
    ).reset_index()

    early = results.iloc[: max(3, len(results) // 4)]
    late = results.iloc[-max(3, len(results) // 4) :]
    diagnostics = {
        "raw_id_change_early_to_late": float(late["mle_id_raw"].mean() - early["mle_id_raw"].mean()),
        "detrended_id_change_early_to_late": float(late["mle_id_detrended"].mean() - early["mle_id_detrended"].mean()),
        "raw_id_early_mean": float(early["mle_id_raw"].mean()),
        "raw_id_late_mean": float(late["mle_id_raw"].mean()),
        "detrended_id_early_mean": float(early["mle_id_detrended"].mean()),
        "detrended_id_late_mean": float(late["mle_id_detrended"].mean()),
        "legacy_id_early_mean": float(early["mle_id_legacy_project"].mean()),
        "legacy_id_late_mean": float(late["mle_id_legacy_project"].mean()),
        "tau_range": [int(results["tau"].min()), int(results["tau"].max())],
        "robustness_id_ranges": {
            phase: {
                metric: [float(group[metric].min()), float(group[metric].max())]
                for metric in ["mle_id_raw", "mle_id_detrended"]
            }
            for phase, group in robustness.groupby("phase")
        },
    }
    return summary, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parsed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--stride", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.parsed_dir / "train_log.csv")
    val = pd.read_csv(args.parsed_dir / "validation_log.csv")
    with (args.parsed_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    results = sliding_analysis(
        train["step"].to_numpy(),
        train["train_loss_per_token"].to_numpy(),
        window_size=args.window,
        stride=args.stride,
    )
    results.to_csv(args.output_dir / "edm_sliding_windows.csv", index=False)
    robust = robustness_table(train)
    robust.to_csv(args.output_dir / "edm_robustness.csv", index=False)
    embedding_sensitivity = embedding_sensitivity_table(train)
    embedding_sensitivity.to_csv(args.output_dir / "edm_embedding_sensitivity.csv", index=False)
    summary, diagnostics = summarize(results, robust, metadata)
    summary.to_csv(args.output_dir / "edm_phase_summary.csv", index=False)
    with (args.output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    make_overview(train, val, metadata, args.output_dir)
    make_edm_trajectory(results, metadata, args.output_dir)
    make_phase_portraits(train, results, metadata, args.output_dir)
    make_robustness(robust, args.output_dir)
    make_embedding_sensitivity(embedding_sensitivity, args.output_dir)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

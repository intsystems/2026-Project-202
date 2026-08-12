"""Batch fixed-tau EDM analysis and cross-optimizer comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_all_edm_tau1 import (
    METHODS,
    STRIDE,
    TAU,
    WINDOW_SIZE,
    make_summary,
    plot_comparison,
    plot_dimensions,
    plot_method_curves,
    sliding_analysis,
)
from parse_nanogpt_log import parse_log


OPTIMIZER_ORDER = [
    "Muon",
    "MuonUSign",
    "EF21-MuonUSign",
    "EF21-MuonSign",
    "EF21-SignMuon",
    "SignMuon",
    "MuonSign",
    "SignSGD",
]
PALETTE = {
    "Muon": "#111827",
    "MuonUSign": "#2563EB",
    "EF21-MuonUSign": "#0891B2",
    "EF21-MuonSign": "#059669",
    "EF21-SignMuon": "#65A30D",
    "SignMuon": "#D97706",
    "MuonSign": "#DC2626",
    "SignSGD": "#9333EA",
}
MARKERS = {name: marker for name, marker in zip(OPTIMIZER_ORDER, ["o", "s", "^", "D", "v", "P", "X", "*"])}


def optimizer_order(values: pd.Series) -> list[str]:
    """Known paper order followed by any newly added optimizer names."""
    present = list(dict.fromkeys(values.astype(str)))
    return [name for name in OPTIMIZER_ORDER if name in present] + sorted(set(present) - set(OPTIMIZER_ORDER))


def discover_logs(input_dir: Path) -> list[Path]:
    logs = []
    for pattern in ("*.txt", "*.log"):
        for path in input_dir.glob(pattern):
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                if any(line.startswith("RUNMETA ") for line in handle):
                    logs.append(path)
    return sorted(set(logs))


def save_parsed(parsed_dir: Path, metadata: dict, train: pd.DataFrame, validation: pd.DataFrame, progress: pd.DataFrame) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(parsed_dir / "train_log.csv", index=False)
    validation.to_csv(parsed_dir / "validation_log.csv", index=False)
    progress.to_csv(parsed_dir / "progress_log.csv", index=False)
    with (parsed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def plot_validation(validation: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    order = optimizer_order(validation["optimizer"])
    fallback_colors = plt.cm.tab20(np.linspace(0, 1, max(len(order), 1)))
    for color_index, optimizer in enumerate(order):
        subset = validation[validation["optimizer"] == optimizer]
        if subset.empty:
            continue
        axis.plot(
            subset["step"], subset["val_loss"], label=optimizer,
            color=PALETTE.get(optimizer, fallback_colors[color_index]), marker=MARKERS.get(optimizer, "o"), markersize=5, linewidth=1.8,
        )
    axis.set_xlabel("optimization step")
    axis.set_ylabel("validation loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.set_title("Validation trajectories across optimizers")
    fig.tight_layout()
    fig.savefig(output_dir / "01_validation_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "01_validation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # EF21-MuonSign is the only log that reports both the exact/server model
    # and the compressed broadcast model W.  Plotting only val_loss makes this
    # run look divergent even though W keeps improving, so expose both series.
    special = validation[validation["optimizer"] == "EF21-MuonSign"]
    if not special.empty and special["val_loss_W"].notna().any():
        fig, axis = plt.subplots(figsize=(9.2, 5.0))
        axis.plot(
            special["step"], special["val_loss"], color=PALETTE["EF21-MuonSign"],
            marker="D", linewidth=2, label="exact/server model",
        )
        axis.plot(
            special["step"], special["val_loss_W"], color="#374151",
            marker="o", linestyle="--", linewidth=2, label="compressed broadcast model W",
        )
        axis.set_xlabel("optimization step")
        axis.set_ylabel("validation loss")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.set_title("EF21-MuonSign: exact model versus compressed W")
        fig.tight_layout()
        fig.savefig(output_dir / "01b_ef21_muonsign_exact_vs_w.png", dpi=220, bbox_inches="tight")
        fig.savefig(output_dir / "01b_ef21_muonsign_exact_vs_w.pdf", bbox_inches="tight")
        plt.close(fig)



def plot_mle_trajectories(windows: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8), sharex=True)
    order = optimizer_order(windows["optimizer"])
    fallback_colors = plt.cm.tab20(np.linspace(0, 1, max(len(order), 1)))
    for axis, kind in zip(axes, ("raw", "detrended")):
        for color_index, optimizer in enumerate(order):
            subset = windows[windows["optimizer"] == optimizer]
            if subset.empty:
                continue
            axis.plot(
                subset["center_step"], subset[f"mle_{kind}"], label=optimizer,
                color=PALETTE.get(optimizer, fallback_colors[color_index]), marker=MARKERS.get(optimizer, "o"), markersize=3.5, linewidth=1.5,
            )
        axis.set_ylabel(f"MLE ID ({kind})")
        axis.set_ylim(8, 19)
        axis.grid(alpha=0.2)
        axis.axvline(2290 * 0.55, color="#6B7280", linestyle="--", linewidth=1)
        axis.axvline(2290, color="#6B7280", linestyle=":", linewidth=1)
    axes[0].legend(frameon=False, ncol=4, fontsize=7.5)
    axes[0].set_title("Levina–Bickel MLE trajectories, fixed $\\tau=1$")
    axes[-1].set_xlabel("window center (optimization step)")
    fig.tight_layout()
    fig.savefig(output_dir / "02_mle_trajectories.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "02_mle_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mle_small_multiples(windows: pd.DataFrame, output_dir: Path) -> None:
    order = optimizer_order(windows["optimizer"])
    ncols = 2
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.7 * nrows), sharex=True, sharey=True, squeeze=False)
    for axis, optimizer in zip(axes.flat, order):
        subset = windows[windows["optimizer"] == optimizer]
        color = PALETTE.get(optimizer, plt.cm.tab20(order.index(optimizer) / max(len(order) - 1, 1)))
        axis.plot(subset["center_step"], subset["mle_raw"], color=color, linewidth=1.8, label="raw")
        axis.plot(subset["center_step"], subset["mle_detrended"], color="#6B7280", linestyle="--", linewidth=1.5, label="detrended")
        axis.axvline(2290 * 0.55, color="#9CA3AF", linestyle="--", linewidth=0.9)
        axis.set_title(optimizer)
        axis.set_ylim(8, 19)
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(order):]:
        axis.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("window center")
    for axis in axes[:, 0]:
        axis.set_ylabel("MLE ID")
    fig.suptitle("Per-optimizer MLE ID: raw and detrended train loss", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "03_mle_small_multiples.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "03_mle_small_multiples.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_early_late_mle(early_late: pd.DataFrame, output_dir: Path) -> None:
    order = optimizer_order(early_late["optimizer"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharey=True)
    y = np.arange(len(order))
    for axis, kind in zip(axes, ("raw", "detrended")):
        indexed = early_late.set_index("optimizer").reindex(order)
        early = indexed[f"mle_{kind}_early"].to_numpy()
        late = indexed[f"mle_{kind}_late"].to_numpy()
        for pos, optimizer, left, right in zip(y, order, early, late):
            color = PALETTE.get(optimizer, plt.cm.tab20(pos / max(len(order) - 1, 1)))
            axis.plot([left, right], [pos, pos], color=color, linewidth=2)
            axis.scatter(left, pos, color=color, facecolor="white", s=42, zorder=3)
            axis.scatter(right, pos, color=color, s=42, zorder=3)
        axis.set_title(f"{kind}: open=early, filled=late")
        axis.set_xlabel("mean MLE ID")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    fig.suptitle("Early-to-late MLE ID shifts (first/last 12 windows)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_mle_early_late.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "04_mle_early_late.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(early_late: pd.DataFrame, output_dir: Path) -> None:
    order = optimizer_order(early_late["optimizer"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    indexed = early_late.set_index("optimizer").reindex(order)
    vmax = 5.0
    for axis, kind in zip(axes, ("raw", "detrended")):
        matrix = np.array([[indexed.loc[optimizer, f"{method.lower()}_{kind}_delta"] for method in METHODS] for optimizer in order])
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axis.set_xticks(range(len(METHODS)), METHODS)
        axis.set_title(kind)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                axis.text(col, row, f"{value:+.1f}", ha="center", va="center", color="white" if abs(value) > 2.2 else "black", fontsize=8)
    axes[0].set_yticks(range(len(order)), order)
    colorbar_axis = fig.add_axes([0.89, 0.20, 0.018, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("late minus early estimated dimension", rotation=90, labelpad=14)
    fig.suptitle("Direction of dimension change across all four methods")
    fig.subplots_adjust(left=0.17, right=0.84, top=0.87, bottom=0.12, wspace=0.12)
    fig.savefig(output_dir / "05_all_method_deltas.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "05_all_method_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_comparison(early_late: pd.DataFrame, runs: pd.DataFrame, output_dir: Path) -> None:
    """Compare the two logged training regimes without treating windows as replicates."""
    merged = early_late.copy()
    labels = {"lmo": "LMO family (lr=0.06)", "sign": "sign family (lr=0.03)"}
    colors = {"lmo": "#2563EB", "sign": "#D97706"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for axis, kind in zip(axes, ("raw", "detrended")):
        for family in ("lmo", "sign"):
            subset = merged[merged["family"] == family]
            early = subset[f"mle_{kind}_early"].mean()
            late = subset[f"mle_{kind}_late"].mean()
            axis.plot(["early", "late"], [early, late], marker="o", markersize=7,
                      linewidth=2.4, color=colors[family], label=labels[family])
        axis.set_title(kind)
        axis.set_ylabel("mean MLE estimate")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MLE by logged training regime ($\\tau=1$)")
    fig.tight_layout()
    fig.savefig(output_dir / "06_mle_regime_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "06_mle_regime_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def make_regime_summary(early_late: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    merged = early_late.merge(
        runs[["run_id", "final_val_loss", "best_val_loss", "train_time_ms"]],
        on="run_id", validate="one_to_one",
    )
    rows: list[dict[str, float | int | str]] = []
    for family, group in merged.groupby("family", sort=True):
        row: dict[str, float | int | str] = {
            "family": family,
            "lr": float(group["lr"].iloc[0]),
            "runs": int(len(group)),
            "final_val_loss_mean": float(group["final_val_loss"].mean()),
            "final_val_loss_median": float(group["final_val_loss"].median()),
            "best_val_loss_mean": float(group["best_val_loss"].mean()),
            "train_time_ms_mean": float(group["train_time_ms"].mean()),
        }
        for method in METHODS:
            for kind in ("raw", "detrended"):
                for period in ("early", "late", "delta"):
                    column = f"{method.lower()}_{kind}_{period}"
                    row[f"mean_{column}"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def early_late_rows(run_id: str, optimizer: str, family: str, lr: float, results: pd.DataFrame) -> dict:
    early = results.iloc[:12]
    late = results.iloc[-12:]
    row: dict[str, float | str] = {
        "run_id": run_id,
        "optimizer": optimizer,
        "family": family,
        "lr": lr,
    }
    for method in METHODS:
        for kind in ("raw", "detrended"):
            column = f"{method.lower()}_{kind}"
            row[f"{column}_early"] = float(early[column].mean())
            row[f"{column}_late"] = float(late[column].mean())
            row[f"{column}_delta"] = float(late[column].mean() - early[column].mean())
    return row


def make_cross_optimizer_summary(runs: pd.DataFrame, early_late: pd.DataFrame) -> pd.DataFrame:
    """Compact, machine-readable summary of the main batch conclusions.

    The 49 sliding windows within a run overlap by 90%, so they are not
    independent replicates.  This table deliberately reports descriptive
    counts and means rather than pseudoreplicated p-values.
    """
    rows: list[dict[str, float | int | str]] = []
    for kind in ("raw", "detrended"):
        for method in METHODS:
            column = f"{method.lower()}_{kind}_delta"
            values = early_late[column]
            rows.append(
                {
                    "series": kind,
                    "method": method,
                    "optimizers_decreasing": int((values < 0).sum()),
                    "optimizers_unchanged": int((values == 0).sum()),
                    "optimizers_increasing": int((values > 0).sum()),
                    "mean_late_minus_early": float(values.mean()),
                    "median_late_minus_early": float(values.median()),
                }
            )
    result = pd.DataFrame(rows)
    result.attrs["n_runs"] = int(len(runs))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = discover_logs(args.input_dir)
    if not logs:
        raise SystemExit("No RUNMETA logs found")

    run_rows = []
    all_validation = []
    all_windows = []
    early_late = []
    seen_run_ids: set[str] = set()
    seen_optimizers: set[str] = set()

    for log_path in logs:
        metadata, train, validation, progress = parse_log(log_path)
        optimizer = metadata["optimizer"]
        run_id = metadata["run_id"]
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id {run_id}")
        seen_run_ids.add(run_id)
        if optimizer in seen_optimizers:
            raise ValueError(
                f"More than one run supplied for optimizer {optimizer}; "
                "seed-aware aggregation is not implemented"
            )
        seen_optimizers.add(optimizer)
        run_dir = args.output_dir / "runs" / run_id
        parsed_dir = run_dir / "parsed"
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_parsed(parsed_dir, metadata, train, validation, progress)

        results, representative = sliding_analysis(train)
        results.insert(0, "run_id", run_id)
        results.insert(1, "optimizer", optimizer)
        results.insert(2, "family", metadata["family"])
        results.to_csv(run_dir / "tau1_all_methods_windows.csv", index=False)
        summary, diagnostics = make_summary(results, metadata)
        summary.insert(0, "run_id", run_id)
        summary.insert(1, "optimizer", optimizer)
        summary.to_csv(run_dir / "tau1_all_methods_summary.csv", index=False)
        with (run_dir / "tau1_diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)

        plot_dimensions(results, metadata, figures_dir, "raw", f"{optimizer}: all four methods, fixed $\\tau=1$ (raw loss)")
        plot_dimensions(results, metadata, figures_dir, "detrended", f"{optimizer}: all four methods, fixed $\\tau=1$ (detrended loss)")
        plot_comparison(results, metadata, figures_dir)
        plot_method_curves(representative, figures_dir)

        validation = validation.copy()
        validation.insert(0, "run_id", run_id)
        validation.insert(1, "optimizer", optimizer)
        all_validation.append(validation)
        all_windows.append(results)
        early_late.append(early_late_rows(run_id, optimizer, metadata["family"], metadata["lr"], results))
        run_rows.append(
            {
                "run_id": run_id,
                "optimizer": optimizer,
                "family": metadata["family"],
                "lr": metadata["lr"],
                "diverged": metadata["run_end"]["diverged"],
                "final_val_loss": float(validation.iloc[-1]["val_loss"]),
                "best_val_loss": float(validation["val_loss"].min()),
                "train_time_ms": metadata["run_end"]["train_time_ms"],
                "peak_memory_mib": metadata["run_end"]["peak_memory_mib"],
            }
        )
        print(f"Analyzed {run_id}")

    runs = pd.DataFrame(run_rows)
    order = optimizer_order(runs["optimizer"])
    order_index = {name: index for index, name in enumerate(order)}
    runs = runs.sort_values("optimizer", key=lambda x: x.map(order_index)).reset_index(drop=True)
    validation_all = pd.concat(all_validation, ignore_index=True)
    windows_all = pd.concat(all_windows, ignore_index=True)
    early_late_df = pd.DataFrame(early_late)
    early_late_df = early_late_df.sort_values("optimizer", key=lambda x: x.map(order_index)).reset_index(drop=True)

    runs.to_csv(args.output_dir / "optimizer_run_summary.csv", index=False)
    validation_all.to_csv(args.output_dir / "validation_all.csv", index=False)
    windows_all.to_csv(args.output_dir / "tau1_all_methods_windows_all.csv", index=False)
    early_late_df.to_csv(args.output_dir / "tau1_early_late_all_methods.csv", index=False)
    cross_optimizer = make_cross_optimizer_summary(runs, early_late_df)
    cross_optimizer.to_csv(args.output_dir / "cross_optimizer_direction_summary.csv", index=False)
    regime_summary = make_regime_summary(early_late_df, runs)
    regime_summary.to_csv(args.output_dir / "training_regime_summary.csv", index=False)

    plot_validation(validation_all, args.output_dir)
    plot_mle_trajectories(windows_all, args.output_dir)
    plot_mle_small_multiples(windows_all, args.output_dir)
    plot_early_late_mle(early_late_df, args.output_dir)
    plot_delta_heatmap(early_late_df, args.output_dir)
    plot_regime_comparison(early_late_df, runs, args.output_dir)
    print(runs.to_string(index=False))
    print(early_late_df[["optimizer", "mle_raw_early", "mle_raw_late", "mle_raw_delta", "mle_detrended_early", "mle_detrended_late", "mle_detrended_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()

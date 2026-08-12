"""Analyse a completed S_n sweep and compare EDM signatures across n.

Run locally after copying/syncing the Colab output directory.  The script uses
the four existing project estimators at fixed tau=1 and writes per-run windows,
early/late summaries, aggregate tables, and comparison figures.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grokking_analysis import get_E_cao, get_E_fnn, get_E_mle, get_E_simplex, get_tau_fixed


METHODS = {
    "FNN": get_E_fnn,
    "Cao": get_E_cao,
    "Simplex": get_E_simplex,
    "MLE": get_E_mle,
}


def discover_runs(input_root: Path) -> list[Path]:
    return sorted(path.parent for path in input_root.rglob("training_log.csv") if (path.parent / "metadata.json").exists())


def estimate_window(series: np.ndarray) -> dict[str, float]:
    results: dict[str, float] = {}
    for index, (name, estimator) in enumerate(METHODS.items()):
        np.random.seed(20260731 + index)
        value = estimator(series, get_tau_fixed(series))
        results[name] = float(value) if value is not None and np.isfinite(value) else np.nan
    return results


def analyse_run(log_path: Path, metadata: dict, window_size: int, stride: int) -> pd.DataFrame:
    frame = pd.read_csv(log_path)
    required = {"step", "train_loss"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{log_path} lacks columns {required - set(frame.columns)}")
    frame = frame.dropna(subset=["step", "train_loss"]).sort_values("step").reset_index(drop=True)
    if len(frame) < window_size:
        raise ValueError(f"{log_path} contains {len(frame)} log rows, fewer than window_size={window_size}")
    rows = []
    values = frame["train_loss"].to_numpy(dtype=float)
    steps = frame["step"].to_numpy(dtype=int)
    for start in range(0, len(values) - window_size + 1, stride):
        window = values[start:start + window_size]
        dims = estimate_window(window)
        rows.append({
            "n": metadata["n"], "seed": metadata["seed"], "run_dir": str(log_path.parent),
            "start_step": int(steps[start]), "end_step": int(steps[start + window_size - 1]),
            "center_step": int(steps[start + window_size // 2]), "tau": 1,
            "loss_mean": float(window.mean()), "loss_std": float(window.std()), **dims,
        })
    return pd.DataFrame(rows)


def early_late(windows: pd.DataFrame, count: int) -> dict[str, float | int]:
    early = windows.iloc[:count]
    late = windows.iloc[-count:]
    row: dict[str, float | int] = {"n": int(windows["n"].iloc[0]), "seed": int(windows["seed"].iloc[0]),
                                  "windows": len(windows)}
    for name in METHODS:
        row[f"{name.lower()}_early"] = float(early[name].mean())
        row[f"{name.lower()}_late"] = float(late[name].mean())
        row[f"{name.lower()}_delta"] = float(late[name].mean() - early[name].mean())
    return row


def plot_mle(windows: pd.DataFrame, out: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    for (n, seed), group in windows.groupby(["n", "seed"], sort=True):
        axis.plot(group["center_step"], group["MLE"], linewidth=1.8, marker="o", markersize=3,
                  label=f"$S_{n}$, seed={seed}")
    axis.set_xlabel("optimization step")
    axis.set_ylabel("MLE intrinsic-dimension estimate")
    axis.set_title("Fixed $\\tau=1$ MLE across symmetric groups")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_mle_by_n.png", dpi=220)
    fig.savefig(out / "01_mle_by_n.pdf")
    plt.close(fig)


def plot_deltas(summary: pd.DataFrame, out: Path) -> None:
    methods = list(METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, method in zip(axes.flat, methods):
        for seed, group in summary.groupby("seed", sort=True):
            axis.plot(group["n"], group[f"{method.lower()}_delta"], marker="o", linewidth=1.8,
                      label=f"seed={seed}")
        axis.axhline(0, color="#6B7280", linewidth=1)
        axis.set_title(method)
        axis.set_ylabel("late − early")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("n in $S_n$")
    fig.suptitle("Early-to-late estimated-dimension change by group")
    fig.tight_layout()
    fig.savefig(out / "02_all_method_deltas_by_n.png", dpi=220)
    fig.savefig(out / "02_all_method_deltas_by_n.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path, help="downloaded/synchronised Colab sweep root")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--window-size", type=int, default=300, help="number of logged rows, not optimizer steps")
    parser.add_argument("--stride", type=int, default=50, help="number of logged rows")
    parser.add_argument("--early-late-windows", type=int, default=5)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.input_root)
    if not runs:
        raise SystemExit("No completed training_log.csv + metadata.json runs found")
    all_windows = []
    summaries = []
    for run_dir in runs:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        windows = analyse_run(run_dir / "training_log.csv", metadata, args.window_size, args.stride)
        if len(windows) < 2 * args.early_late_windows:
            print(f"Skipping early/late summary for {run_dir}: only {len(windows)} EDM windows")
        else:
            summaries.append(early_late(windows, args.early_late_windows))
        all_windows.append(windows)
        windows.to_csv(args.output_root / f"windows_S{metadata['n']}_seed{metadata['seed']}.csv", index=False)
    windows_all = pd.concat(all_windows, ignore_index=True)
    windows_all.to_csv(args.output_root / "all_tau1_windows.csv", index=False)
    summary = pd.DataFrame(summaries).sort_values(["n", "seed"]).reset_index(drop=True)
    summary.to_csv(args.output_root / "early_late_summary.csv", index=False)
    plot_mle(windows_all, args.output_root)
    if not summary.empty:
        plot_deltas(summary, args.output_root)
    print(f"Analysed {len(runs)} runs; wrote results to {args.output_root}")


if __name__ == "__main__":
    main()

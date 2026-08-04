"""Detailed fixed-tau EDM analysis of the p=211, AdamW WD=0.5 run.

The script intentionally keeps the four project diagnostics (FNN, Cao,
simplex projection, and the legacy Levina--Bickel arithmetic aggregation) and
adds the MacKay--Ghahramani pooled-likelihood/harmonic aggregation on every
dimension plot.  All analyses use tau=1 in log-index units.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


HERE = Path(__file__).resolve().parent
GROKKING_DIR = HERE.parent
sys.path.insert(0, str(GROKKING_DIR))

from grokking_analysis import get_E_cao, get_E_fnn, get_E_simplex  # noqa: E402
from search_for_optimal_parameters import delay_embedding  # noqa: E402


TAU = 1
EMBEDDING_DIMENSION = 15
K_NEIGHBORS = 5
WINDOW_SIZE = 200
STRIDE = 10

METRICS = (
    "train_loss",
    "val_loss",
    "weight_norm",
    "gradient_norm",
    "update_norm",
    "gradient_cosine",
    "gradient_participation_ratio",
    "parameter_participation_ratio",
    "train_entropy",
    "val_entropy",
    "train_lossproj_r0",
    "val_lossproj_r0",
    "weightproj__W_U__r0",
    "gradproj__W_U__r0",
    "updateproj__W_U__r0",
    "weightproj__embed__W_E__r0",
    "gradproj__embed__W_E__r0",
)

CORE_METRICS = (
    "train_loss",
    "val_loss",
    "weight_norm",
    "gradient_norm",
    "update_norm",
    "gradient_cosine",
    "gradient_participation_ratio",
    "parameter_participation_ratio",
    "train_entropy",
    "val_entropy",
)

PROJECTION_METRICS = (
    "train_lossproj_r0",
    "val_lossproj_r0",
    "weightproj__W_U__r0",
    "gradproj__W_U__r0",
    "updateproj__W_U__r0",
    "weightproj__embed__W_E__r0",
    "gradproj__embed__W_E__r0",
)

METHODS = ("FNN", "Cao", "Simplex", "LB", "MG")
COLORS = {
    "FNN": "#2563EB",
    "Cao": "#059669",
    "Simplex": "#D97706",
    "LB": "#7C3AED",
    "MG": "#DC2626",
}

LABELS = {
    "train_loss": "train loss",
    "val_loss": "validation loss",
    "weight_norm": "weight norm",
    "gradient_norm": "gradient norm",
    "update_norm": "update norm",
    "gradient_cosine": "gradient cosine",
    "gradient_participation_ratio": "gradient participation ratio",
    "parameter_participation_ratio": "parameter participation ratio",
    "train_entropy": "train entropy",
    "val_entropy": "validation entropy",
    "train_lossproj_r0": "train loss projection r0",
    "val_lossproj_r0": "validation loss projection r0",
    "weightproj__W_U__r0": "unembedding weight projection r0",
    "gradproj__W_U__r0": "unembedding gradient projection r0",
    "updateproj__W_U__r0": "unembedding update projection r0",
    "weightproj__embed__W_E__r0": "embedding weight projection r0",
    "gradproj__embed__W_E__r0": "embedding gradient projection r0",
}


def standardize(series: np.ndarray) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / scale


def linear_detrend(series: np.ndarray) -> np.ndarray:
    x = np.arange(len(series), dtype=float)
    values = np.asarray(series, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    return values - (slope * x + intercept)


def lb_mg_dimension(
    series: np.ndarray,
    tau: int = TAU,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    k_neighbors: int = K_NEIGHBORS,
) -> tuple[float, float, float, float]:
    """Return LB arithmetic ID, MG pooled ID, local median, and local IQR.

    This deliberately matches the project's historical estimator: Euclidean
    kNN in an E=15 delay embedding and no Theiler exclusion.  MacKay--
    Ghahramani differs only in the statistically correct pooled aggregation:
    inverse of the mean local inverse dimension (harmonic mean of local IDs).
    """
    embedded = delay_embedding(standardize(series), embedding_dimension, tau)
    if len(embedded) < k_neighbors + 2:
        return (math.nan,) * 4
    distances, _ = KDTree(embedded).query(embedded, k=k_neighbors + 1)
    distances = np.maximum(distances[:, 1:], 1e-12)
    outer = distances[:, -1:]
    log_sums = np.log(outer / distances[:, :-1]).sum(axis=1)
    valid = np.isfinite(log_sums) & (log_sums > 1e-12)
    if not valid.any():
        return (math.nan,) * 4
    inverse_local = log_sums[valid] / (k_neighbors - 1)
    local = 1.0 / inverse_local
    lb = float(np.mean(local))
    mg = float(1.0 / np.mean(inverse_local))
    median = float(np.median(local))
    iqr = float(np.quantile(local, 0.75) - np.quantile(local, 0.25))
    return lb, mg, median, iqr


def estimate_all(series: np.ndarray) -> dict[str, float]:
    clean = standardize(series)
    if np.std(clean) < 1e-12:
        return {name: 1.0 for name in METHODS} | {"local_median": 1.0, "local_iqr": 0.0}
    out: dict[str, float] = {}
    for seed, (name, fn) in enumerate((
        ("FNN", get_E_fnn), ("Cao", get_E_cao), ("Simplex", get_E_simplex)
    )):
        np.random.seed(20260803 + seed)
        try:
            value = fn(clean, TAU)
            out[name] = float(value) if value is not None and np.isfinite(value) else math.nan
        except Exception:
            out[name] = math.nan
    lb, mg, median, iqr = lb_mg_dimension(clean)
    out.update(LB=lb, MG=mg, local_median=median, local_iqr=iqr)
    return out


def first_stable_onset(frame: pd.DataFrame, metric: str, threshold: float, patience: int) -> int | None:
    mask = (frame[metric].to_numpy(dtype=float) >= threshold)
    for start in range(len(mask) - patience + 1):
        if mask[start:start + patience].all():
            return int(frame["step"].iloc[start])
    return None


def first_flag_step(frame: pd.DataFrame, flag: str) -> int | None:
    selected = frame[frame[flag].astype(bool)]
    return None if selected.empty else int(selected.iloc[0]["step"])


def transition_metadata(frame: pd.DataFrame) -> dict[str, int | float | None]:
    memo = first_flag_step(frame, "stable_memorization")
    grok = first_flag_step(frame, "genuine_grokking")
    result: dict[str, int | float | None] = {
        "rows": int(len(frame)),
        "first_step": int(frame.step.iloc[0]),
        "last_step": int(frame.step.iloc[-1]),
        "logging_stride_steps": int(np.median(np.diff(frame.step))),
        "weight_decay": float(frame.weight_decay.median()),
        "stable_memorization_flag_step": memo,
        "grok_flag_step": grok,
        "train_acc_095_stable5_onset": first_stable_onset(frame, "train_acc", 0.95, 5),
        "train_acc_099_stable5_onset": first_stable_onset(frame, "train_acc", 0.99, 5),
        "val_acc_050_stable5_onset": first_stable_onset(frame, "val_acc", 0.50, 5),
        "val_acc_090_stable5_onset": first_stable_onset(frame, "val_acc", 0.90, 5),
        "val_acc_095_stable5_onset": first_stable_onset(frame, "val_acc", 0.95, 5),
        "val_acc_099_stable5_onset": first_stable_onset(frame, "val_acc", 0.99, 5),
        "final_train_acc": float(frame.train_acc.iloc[-1]),
        "final_val_acc": float(frame.val_acc.iloc[-1]),
        "final_train_loss": float(frame.train_loss.iloc[-1]),
        "final_val_loss": float(frame.val_loss.iloc[-1]),
        "elapsed_seconds": float(frame.elapsed_seconds.iloc[-1]),
    }
    result["flag_gap_steps"] = None if memo is None or grok is None else grok - memo
    t95 = result["train_acc_095_stable5_onset"]
    v95 = result["val_acc_095_stable5_onset"]
    result["threshold_gap_steps"] = None if t95 is None or v95 is None else int(v95) - int(t95)
    result["outcome"] = "genuine_grokking" if grok is not None else "not_grokked"
    return result


def sliding_analysis(frame: pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
    rows: list[dict] = []
    steps = frame.step.to_numpy(dtype=int)
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        for start in range(0, len(values) - window_size + 1, stride):
            stop = start + window_size
            raw = values[start:stop]
            detrended = linear_detrend(raw)
            raw_est = estimate_all(raw)
            detrended_est = estimate_all(detrended)
            row: dict[str, float | int | str] = {
                "metric": metric, "tau": TAU,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "k_neighbors": K_NEIGHBORS,
                "window_size": window_size, "stride": stride,
                "start_step": int(steps[start]),
                "center_step": int(steps[start + window_size // 2]),
                "end_step": int(steps[stop - 1]),
                "mean": float(np.mean(raw)), "std": float(np.std(raw)),
            }
            for key, value in raw_est.items():
                row[f"{key.lower()}_raw"] = value
            for key, value in detrended_est.items():
                row[f"{key.lower()}_detrended"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def phase_summary(windows: pd.DataFrame, meta: dict) -> pd.DataFrame:
    memo = int(meta["stable_memorization_flag_step"])
    grok_value = meta.get("grok_flag_step")
    grok = None if grok_value is None else int(grok_value)
    phases = {
        "fitting": lambda g: g[g.end_step < memo],
        "memorization_gap": (lambda g: g[g.start_step >= memo]) if grok is None else (lambda g: g[(g.start_step >= memo) & (g.end_step < grok)]),
        "transition": (lambda g: g.iloc[0:0]) if grok is None else (lambda g: g[(g.start_step < grok) & (g.end_step >= grok)]),
        "post_grok": (lambda g: g.iloc[0:0]) if grok is None else (lambda g: g[g.start_step >= grok]),
    }
    rows: list[dict] = []
    for metric, group in windows.groupby("metric", sort=False):
        for phase, selector in phases.items():
            selected = selector(group)
            for series in ("raw", "detrended"):
                row = {"metric": metric, "phase": phase, "series": series,
                       "windows": int(len(selected))}
                for method in METHODS:
                    column = f"{method.lower()}_{series}"
                    row[method] = float(selected[column].mean()) if len(selected) else math.nan
                rows.append(row)
    return pd.DataFrame(rows)


def window_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    steps = frame.step.to_numpy(dtype=int)
    for metric in ("train_loss", "val_loss", "weight_norm", "gradient_norm"):
        values = frame[metric].to_numpy(dtype=float)
        for window_size in (100, 150, 200, 250):
            stride = 20
            for start in range(0, len(values) - window_size + 1, stride):
                stop = start + window_size
                est = estimate_all(linear_detrend(values[start:stop]))
                rows.append({"metric": metric, "window_size": window_size,
                             "center_step": int(steps[start + window_size // 2]),
                             "lb": est["LB"], "mg": est["MG"]})
    return pd.DataFrame(rows)


def sensitivity_grid(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    phases = {
        "fitting": (2000, 10000),
        "memorization_gap": (20000, 40000),
        "transition": (41000, 48000),
        "post_grok": (46500, int(frame.step.iloc[-1])),
    }
    rows: list[dict] = []
    for metric in ("train_loss", "val_loss", "weight_norm", "gradient_norm"):
        for phase, (lo, hi) in phases.items():
            values = frame.loc[(frame.step >= lo) & (frame.step <= hi), metric].to_numpy(dtype=float)
            values = linear_detrend(values)
            for embedding_dimension in (6, 9, 12, 15):
                for k_neighbors in (5, 8, 10, 15):
                    if len(values) <= embedding_dimension + k_neighbors + 2:
                        continue
                    lb, mg, median, iqr = lb_mg_dimension(
                        values, embedding_dimension=embedding_dimension,
                        k_neighbors=k_neighbors,
                    )
                    rows.append({"metric": metric, "phase": phase,
                                 "embedding_dimension": embedding_dimension,
                                 "k_neighbors": k_neighbors, "lb": lb, "mg": mg,
                                 "local_median": median, "local_iqr": iqr})
    return pd.DataFrame(rows)


def add_events(axis: plt.Axes, meta: dict, labels: bool = False) -> None:
    memo = meta["stable_memorization_flag_step"]
    grok = meta["grok_flag_step"]
    if memo is not None:
        axis.axvline(memo, color="#2563EB", linestyle="--", linewidth=1.1,
                     label="stable memorization" if labels else None)
    if grok is not None:
        axis.axvline(grok, color="#DC2626", linestyle="--", linewidth=1.1,
                     label="grokking" if labels else None)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview(frame: pd.DataFrame, meta: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True)
    axes[0].semilogy(frame.step, frame.train_loss.clip(lower=1e-7), label="train loss", color="#2563EB")
    axes[0].semilogy(frame.step, frame.val_loss.clip(lower=1e-7), label="validation loss", color="#DC2626")
    axes[0].set_ylabel("cross-entropy (log scale)")
    axes[1].plot(frame.step, frame.train_acc, label="train accuracy", color="#2563EB")
    axes[1].plot(frame.step, frame.val_acc, label="validation accuracy", color="#DC2626")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].plot(frame.step, frame.weight_norm, label="weight norm", color="#7C3AED")
    axes[2].set_ylabel("weight norm")
    axes[2].set_xlabel("optimizer step")
    for index, axis in enumerate(axes):
        add_events(axis, meta, labels=index == 0)
        axis.legend(frameon=False, ncol=3, fontsize=8)
        axis.grid(alpha=0.22)
    wd = float(frame.weight_decay.median()) if "weight_decay" in frame else math.nan
    axes[0].set_title(f"p=211 modular addition, AdamW weight decay {wd:g}")
    fig.tight_layout()
    save_figure(fig, output, "01_training_overview")


def plot_four_methods_loss(windows: pd.DataFrame, meta: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    methods = ("FNN", "Cao", "Simplex", "LB")
    for axis, method in zip(axes.flat, methods):
        for metric, linestyle in (("train_loss", "-"), ("val_loss", "--")):
            group = windows[windows.metric == metric]
            axis.plot(group.center_step, group[f"{method.lower()}_detrended"],
                      color=COLORS[method], linestyle=linestyle,
                      label=LABELS[metric], linewidth=1.6)
        # User-requested MG curve is included on every dimension panel.
        mg = windows[windows.metric == "val_loss"]
        axis.plot(mg.center_step, mg.mg_detrended, color=COLORS["MG"],
                  linewidth=1.35, alpha=0.9, label="MG ID (validation loss)")
        add_events(axis, meta)
        axis.set_title(method + (" arithmetic" if method == "LB" else ""))
        axis.set_ylabel("estimated dimension")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("window center (optimizer step)")
    fig.suptitle("Four EDM diagnostics on linearly detrended losses; MG-ID overlaid")
    fig.tight_layout()
    save_figure(fig, output, "02_four_methods_loss_with_mg")


def plot_four_methods_raw_vs_detrended(windows: pd.DataFrame, meta: dict, output: Path) -> None:
    """Make the raw/detrended distinction explicit for every EDM method.

    The old report had a raw training overview next to a detrended ID panel,
    which made it easy to read the latter as a raw estimate.  This figure puts
    both estimates side by side and labels the preprocessing in the titles.
    """
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.8), sharex=True, sharey=True)
    methods = ("FNN", "Cao", "Simplex", "LB")
    for row, series, descriptor in ((0, "raw", "raw window"),
                                    (1, "detrended", "linear trend removed per window")):
        for axis, method in zip(axes[row], methods):
            for metric, linestyle in (("train_loss", "-"), ("val_loss", "--")):
                group = windows[windows.metric == metric]
                axis.plot(group.center_step, group[f"{method.lower()}_{series}"],
                          color=COLORS[method], linestyle=linestyle,
                          label=LABELS[metric], linewidth=1.25)
            # MG is the pooled kNN-MLE reference, shown in both rows.
            group = windows[windows.metric == "val_loss"]
            axis.plot(group.center_step, group[f"mg_{series}"], color=COLORS["MG"],
                      linewidth=1.25, alpha=0.9, label="MG val loss")
            add_events(axis, meta)
            axis.set_title(f"{method}: {descriptor}", fontsize=9)
            axis.grid(alpha=0.22)
            axis.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("estimated dimension")
    axes[1, 0].set_ylabel("estimated dimension")
    for axis in axes[1]:
        axis.set_xlabel("window center (optimizer step)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Raw versus detrended EDM estimates on train/validation loss", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output, "02b_four_methods_loss_raw_vs_detrended")


def plot_lb_mg_grid(windows: pd.DataFrame, metrics: tuple[str, ...], meta: dict,
                    output: Path, name: str, title: str) -> None:
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11.5, 3.1 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, metric in zip(axes, metrics):
        group = windows[windows.metric == metric]
        axis.plot(group.center_step, group.lb_detrended, color=COLORS["LB"],
                  label="Levina--Bickel (arithmetic)", linewidth=1.5)
        axis.plot(group.center_step, group.mg_detrended, color=COLORS["MG"],
                  label="MacKay--Ghahramani", linewidth=1.5)
        add_events(axis, meta)
        axis.set_title(LABELS[metric])
        axis.set_ylabel("intrinsic dimension")
        axis.grid(alpha=0.22)
    for axis in axes[len(metrics):]:
        axis.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes[-columns:]:
        if axis.get_visible():
            axis.set_xlabel("window center (optimizer step)")
    fig.suptitle(title + " (linear trend removed independently in each window)")
    fig.tight_layout()
    save_figure(fig, output, name)


def plot_raw_detrended(windows: pd.DataFrame, meta: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for axis, metric in zip(axes.flat, ("train_loss", "val_loss", "weight_norm", "gradient_norm")):
        group = windows[windows.metric == metric]
        axis.plot(group.center_step, group.lb_raw, color=COLORS["LB"], alpha=0.35,
                  label="LB raw")
        axis.plot(group.center_step, group.lb_detrended, color=COLORS["LB"],
                  label="LB detrended (linear trend removed)")
        axis.plot(group.center_step, group.mg_raw, color=COLORS["MG"], alpha=0.35,
                  label="MG raw")
        axis.plot(group.center_step, group.mg_detrended, color=COLORS["MG"],
                  label="MG detrended (linear trend removed)")
        add_events(axis, meta)
        axis.set_title(LABELS[metric])
        axis.set_ylabel("intrinsic dimension")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("window center (optimizer step)")
    fig.suptitle("Raw versus linearly detrended ID; both LB and MG aggregations")
    fig.tight_layout()
    save_figure(fig, output, "05_raw_vs_detrended_lb_mg")


def plot_phase_heatmaps(summary: pd.DataFrame, output: Path) -> None:
    phases = ("fitting", "memorization_gap", "transition", "post_grok")
    metrics = CORE_METRICS[:8]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    image = None
    matrices = []
    for method in ("LB", "MG"):
        subset = summary[summary.series == "detrended"]
        matrix = np.array([
            [subset[(subset.metric == metric) & (subset.phase == phase)][method].iloc[0]
             for phase in phases] for metric in metrics
        ])
        matrices.append(matrix)
    vmax = max(float(np.nanquantile(m, 0.95)) for m in matrices)
    for axis, method, matrix in zip(axes, ("LB", "MG"), matrices):
        image = axis.imshow(matrix, cmap="viridis", vmin=1, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(phases)), ["fitting", "gap", "transition", "post-grok"], rotation=20)
        axis.set_title(method + " detrended ID")
        for i in range(len(metrics)):
            for j in range(len(phases)):
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                          color="white" if matrix[i, j] > 0.55 * vmax else "black", fontsize=7)
    axes[0].set_yticks(range(len(metrics)), [LABELS[m] for m in metrics])
    fig.colorbar(image, ax=axes, label="phase-average intrinsic dimension", shrink=0.78)
    fig.suptitle("Phase summary: arithmetic LB versus MacKay--Ghahramani")
    fig.subplots_adjust(left=0.22, right=0.91, bottom=0.15, top=0.87, wspace=0.12)
    save_figure(fig, output, "06_phase_heatmaps_lb_mg")


def plot_window_robustness(robustness: pd.DataFrame, meta: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for axis, metric in zip(axes.flat, robustness.metric.unique()):
        for window_size in sorted(robustness.window_size.unique()):
            group = robustness[(robustness.metric == metric) & (robustness.window_size == window_size)]
            axis.plot(group.center_step, group.mg, label=f"MG W={window_size}", linewidth=1.1)
        # Include the corresponding LB reference on every panel.
        reference = robustness[(robustness.metric == metric) & (robustness.window_size == 200)]
        axis.plot(reference.center_step, reference.lb, color=COLORS["LB"], linestyle="--",
                  label="LB W=200", linewidth=1.4)
        add_events(axis, meta)
        axis.set_title(LABELS[metric])
        axis.set_ylabel("detrended intrinsic dimension")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("window center (optimizer step)")
    fig.suptitle("Window-length sensitivity of MG ID with LB reference")
    fig.tight_layout()
    save_figure(fig, output, "07_window_robustness_lb_mg")


def plot_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for axis, metric in zip(axes.flat, sensitivity.metric.unique()):
        subset = sensitivity[(sensitivity.metric == metric) &
                             (sensitivity.phase == "memorization_gap")]
        matrix = subset.pivot(index="embedding_dimension", columns="k_neighbors", values="mg")
        im = axis.imshow(matrix.values, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_xlabel("k neighbours")
        axis.set_ylabel("embedding E")
        axis.set_title(LABELS[metric] + " (MG, gap phase)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center",
                          color="white", fontsize=7)
        fig.colorbar(im, ax=axis, shrink=0.75)
    fig.suptitle("MacKay--Ghahramani sensitivity to embedding dimension and k")
    fig.tight_layout()
    save_figure(fig, output, "08_mg_E_k_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--fast", action="store_true", help="Skip expensive E/k sensitivity grid")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input_csv).sort_values("step").reset_index(drop=True)
    missing = [column for column in METRICS if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["gradient_cosine"] = frame.gradient_cosine.interpolate(limit_direction="both")

    meta = transition_metadata(frame)
    windows = sliding_analysis(frame, args.window_size, args.stride)
    summary = phase_summary(windows, meta)
    robustness = window_robustness(frame)
    sensitivity = pd.DataFrame() if args.fast else sensitivity_grid(frame, meta)

    windows.to_csv(args.output_dir / "tau1_edm_windows_all_methods.csv", index=False)
    summary.to_csv(args.output_dir / "phase_dimension_summary.csv", index=False)
    robustness.to_csv(args.output_dir / "window_robustness_lb_mg.csv", index=False)
    sensitivity.to_csv(args.output_dir / "E_k_sensitivity_lb_mg.csv", index=False)
    (args.output_dir / "transition_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    plot_overview(frame, meta, args.output_dir)
    plot_four_methods_loss(windows, meta, args.output_dir)
    plot_four_methods_raw_vs_detrended(windows, meta, args.output_dir)
    plot_lb_mg_grid(windows, CORE_METRICS, meta, args.output_dir,
                    "03_core_metrics_lb_mg",
                    "Core scalar observers: Levina--Bickel and MacKay--Ghahramani")
    plot_lb_mg_grid(windows, PROJECTION_METRICS, meta, args.output_dir,
                    "04_projection_metrics_lb_mg",
                    "Fixed random projections: Levina--Bickel and MacKay--Ghahramani")
    plot_raw_detrended(windows, meta, args.output_dir)
    plot_phase_heatmaps(summary, args.output_dir)
    plot_window_robustness(robustness, meta, args.output_dir)
    if not sensitivity.empty:
        plot_sensitivity(sensitivity, args.output_dir)

    print(json.dumps(meta, indent=2))
    print(summary[(summary.series == "detrended") &
                  (summary.metric.isin(("train_loss", "val_loss", "weight_norm", "gradient_norm")))].to_string(index=False))


if __name__ == "__main__":
    main()

"""Build the corrected EDM report for the p=211, weight-decay-zero ablation.

This builder is deliberately separate from the WD=0.5 report.  The original
WD=0 PDF reused prose and transition numbers from the grokking run even though
its figures were computed from the correct WD=0 CSV.  Everything in this file
is derived from the WD=0 source log and its saved EDM window table.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "training_log_p_211_wd_0.csv"
ANALYSIS = HERE / "edm_report_p211_wd0_tau1"
OUTPUT = ANALYSIS / "p211_wd0_detailed_edm_report_tau1_lb_mg.pdf"
WINDOWS_CSV = ANALYSIS / "tau1_edm_windows_all_methods.csv"
AUDIT_JSON = ANALYSIS / "corrected_report_audit.json"

TAU = 1
E = 15
K = 5
WINDOW_ROWS = 200
WINDOW_STRIDE_ROWS = 100

CORE = (
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
PROJECTIONS = (
    "train_lossproj_r0",
    "val_lossproj_r0",
    "weightproj__W_U__r0",
    "gradproj__W_U__r0",
    "updateproj__W_U__r0",
    "weightproj__embed__W_E__r0",
    "gradproj__embed__W_E__r0",
)
LABEL = {
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
    "weightproj__W_U__r0": "W_U weight projection r0",
    "gradproj__W_U__r0": "W_U gradient projection r0",
    "updateproj__W_U__r0": "W_U update projection r0",
    "weightproj__embed__W_E__r0": "embedding weight projection r0",
    "gradproj__embed__W_E__r0": "embedding gradient projection r0",
}
METHODS = ("FNN", "Cao", "Simplex", "LB", "MG")
COLORS = {
    "FNN": "#2563EB",
    "Cao": "#059669",
    "Simplex": "#D97706",
    "LB": "#7C3AED",
    "MG": "#DC2626",
}


def stable_onset(frame: pd.DataFrame, metric: str, threshold: float, patience: int = 5) -> int | None:
    mask = frame[metric].to_numpy(float) >= threshold
    for start in range(len(mask) - patience + 1):
        if mask[start : start + patience].all():
            return int(frame.step.iloc[start])
    return None


def first_flag(frame: pd.DataFrame, column: str) -> int | None:
    selected = frame[frame[column].astype(bool)]
    return None if selected.empty else int(selected.step.iloc[0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(frame: pd.DataFrame, windows: pd.DataFrame) -> dict:
    required = {"step", "weight_decay", "train_loss", "val_loss", "train_acc", "val_acc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    if not np.allclose(frame.weight_decay.to_numpy(float), 0.0):
        raise ValueError("The source is not a fixed WD=0 run")
    if bool(frame.genuine_grokking.astype(bool).any()):
        raise ValueError("The source unexpectedly contains a genuine_grokking flag")
    for field, expected in {
        "tau": TAU,
        "embedding_dimension": E,
        "k_neighbors": K,
        "window_size": WINDOW_ROWS,
        "stride": WINDOW_STRIDE_ROWS,
    }.items():
        values = windows[field].dropna().unique()
        if len(values) != 1 or int(values[0]) != expected:
            raise ValueError(f"Unexpected {field}: {values}; expected {expected}")

    # Verify that saved windows really address this source log.  This catches
    # accidental reuse of the WD=0.5 artifacts even when filenames look right.
    steps = frame.step.to_numpy(int)
    step_to_index = {int(step): idx for idx, step in enumerate(steps)}
    for row in windows.iloc[:: max(1, len(windows) // 25)].itertuples():
        start = step_to_index[int(row.start_step)]
        values = frame[row.metric].iloc[start : start + int(row.window_size)].to_numpy(float)
        if not np.isclose(values.mean(), float(row.mean), rtol=1e-9, atol=1e-9):
            raise ValueError(f"Window/source mismatch for {row.metric} at {row.start_step}")
        if int(steps[start + int(row.window_size) // 2]) != int(row.center_step):
            raise ValueError("Window center mismatch")

    memo = first_flag(frame, "stable_memorization")
    grok = first_flag(frame, "genuine_grokking")
    return {
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "windows_sha256": sha256(WINDOWS_CSV),
        "rows": int(len(frame)),
        "first_step": int(frame.step.iloc[0]),
        "last_step": int(frame.step.iloc[-1]),
        "logging_stride_steps": int(np.median(np.diff(frame.step))),
        "logging": {
            "median_step_interval": int(np.median(np.diff(frame.step))),
            "first_interval_steps": int(frame.step.iloc[1] - frame.step.iloc[0]),
        },
        "weight_decay": float(frame.weight_decay.median()),
        "stable_memorization_flag_step": memo,
        "train_acc_099_stable5_onset": stable_onset(frame, "train_acc", 0.99),
        "grok_flag_step": grok,
        "final_train_acc": float(frame.train_acc.iloc[-1]),
        "final_val_acc": float(frame.val_acc.iloc[-1]),
        "best_val_acc": float(frame.val_acc.max()),
        "final_train_loss": float(frame.train_loss.iloc[-1]),
        "final_val_loss": float(frame.val_loss.iloc[-1]),
        "max_val_loss": float(frame.val_loss.max()),
        "initial_weight_norm": float(frame.weight_norm.iloc[0]),
        "final_weight_norm": float(frame.weight_norm.iloc[-1]),
        "outcome": "memorized_but_not_generalized",
        "edm_protocol": {
            "tau_log_rows": TAU,
            "tau_optimizer_steps": int(np.median(np.diff(frame.step))) * TAU,
            "embedding_dimension": E,
            "k_neighbors": K,
            "window_rows": WINDOW_ROWS,
            "window_optimizer_steps_approx": WINDOW_ROWS * int(np.median(np.diff(frame.step))),
            "stride_rows": WINDOW_STRIDE_ROWS,
            "stride_optimizer_steps": WINDOW_STRIDE_ROWS * int(np.median(np.diff(frame.step))),
            "overlap_fraction": 1.0 - WINDOW_STRIDE_ROWS / WINDOW_ROWS,
            "detrending": "independent least-squares linear trend per window",
            "theiler_exclusion": False,
        },
    }


def add_memo(axis: plt.Axes, memo: int) -> None:
    axis.axvline(memo, color="#2563EB", linestyle="--", linewidth=1.1, label="stable memorization")


def optimizer_steps(values) -> np.ndarray:
    """Optimizer steps in their original, absolute units."""
    return np.asarray(values, dtype=float)


def format_step_axis(axis: plt.Axes, last_step: int = 200_000, *, window_center: bool = False) -> None:
    """Format time in *actual optimizer updates*, never in log-row indices.

    EDM estimates are attached to the optimizer step at the centre of their
    source window.  This is made explicit on the windowed figures.
    """
    axis.set_xlim(0.0, float(last_step))
    axis.xaxis.set_major_locator(MultipleLocator(25_000))
    if window_center:
        # Five-panel EDM figures are unreadable with six-digit tick labels.
        # Keep the plotted coordinate in optimizer steps, but display it in
        # explicitly declared units of 10^3 optimizer updates.
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:g}"))
    else:
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):,}"))
    axis.xaxis.get_offset_text().set_visible(False)
    # ``sharex=True`` hides labels on all but the last row by default.  Every
    # report panel is intended to be interpretable on its own, so explicitly
    # restore tick labels and put the physical unit on every subplot.
    axis.tick_params(axis="x", which="both", labelbottom=True)
    axis.set_xlabel(
        r"window-centre step $t_c$ [$10^3$ updates]"
        if window_center else
        "optimizer step $t$ [updates]"
    )


def format_dimension_axis(axis: plt.Axes, label: str | None = None) -> None:
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    if label is not None:
        axis.set_ylabel(label, fontsize=8)


def dimension_label(method: str) -> str:
    if method in ("FNN", "Cao", "Simplex"):
        return "embedding dimension $E$ [coord.]"
    # The previous full MacKay--Ghahramani label was wider than a two-column
    # panel.  The acronym is defined in the report, so use a compact unit-bearing
    # label here rather than allowing it to be clipped.
    return f"{method} intrinsic dimension $d$ [dim.]"


def robust_upper(values: np.ndarray) -> float:
    """A display limit for unstable arithmetic-LB curves.

    Values above it are explicitly marked, never silently discarded.  The
    underlying CSV and summary statistics retain the original values.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0
    q25, q75, q90 = np.quantile(finite, [0.25, 0.75, 0.90])
    cap = max(q90, q75 + 3.0 * (q75 - q25), 1.1 * np.median(finite), 1.0)
    return float(cap)


def plot_with_marked_overflow(axis: plt.Axes, x, y, cap: float, **kwargs) -> int:
    """Plot on a linear axis and mark values above the declared display cap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    visible = np.minimum(y, cap)
    line = axis.plot(x, visible, **kwargs)
    overflow = np.isfinite(y) & (y > cap)
    if overflow.any():
        color = kwargs.get("color", line[0].get_color())
        axis.scatter(x[overflow], np.full(overflow.sum(), cap), marker="^", s=19,
                     color=color, edgecolors="white", linewidths=0.35, zorder=4)
    return int(overflow.sum())


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ANALYSIS / f"{name}.png", dpi=210, bbox_inches="tight")
    fig.savefig(ANALYSIS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(frame: pd.DataFrame, meta: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.7, 8.6), sharex=True)
    x = optimizer_steps(frame.step)
    axes[0].semilogy(x, frame.train_loss.clip(lower=1e-8), color="#2563EB", label="train loss")
    axes[0].semilogy(x, frame.val_loss.clip(lower=1e-8), color="#DC2626", label="validation loss")
    axes[0].set_ylabel("cross-entropy [nats]\nlogarithmic y-axis")
    axes[1].plot(x, frame.train_acc, color="#2563EB", label="train accuracy")
    axes[1].plot(x, frame.val_acc, color="#DC2626", label="validation accuracy")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("accuracy [fraction]")
    axes[2].plot(x, frame.weight_norm, color="#7C3AED", label="weight norm")
    axes[2].set_ylabel(r"parameter norm $\|\theta\|_2$ [parameter units]")
    for axis in axes:
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, ncol=3)
        format_step_axis(axis, int(meta["last_step"]))
    axes[0].set_title("p=211 modular addition, fixed weight decay 0: memorization without generalization")
    fig.tight_layout()
    save(fig, "wd0_corrected_01_training_overview")


def plot_methods(windows: pd.DataFrame, meta: dict) -> None:
    # One panel per estimator is substantially easier to read in an A4 report
    # than a 2 x 5 matrix of very narrow panels. Colour identifies the source
    # metric; saturation identifies raw versus detrended preprocessing.
    # This is a full report page: use the available height instead of squeezing
    # five estimators into the old 80-mm strip.
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9.2), sharex=True)
    flat = axes.reshape(-1)
    for axis, method in zip(flat, METHODS):
        plotted = []
        for metric, base_colour, pale_colour, linestyle in (
            ("train_loss", "#2563EB", "#93C5FD", "-"),
            ("val_loss", "#DC2626", "#FCA5A5", "--"),
        ):
            group = windows[windows.metric == metric]
            for series, colour, alpha, suffix in (
                ("raw", pale_colour, 0.72, "raw"),
                ("detrended", base_colour, 1.0, "detrended"),
            ):
                values = group[f"{method.lower()}_{series}"].to_numpy(float)
                plotted.append((group, values, colour, alpha, linestyle, f"{LABEL[metric]}, {suffix}"))
        combined = np.concatenate([item[1] for item in plotted])
        cap = robust_upper(combined) if method == "LB" else float(np.nanmax(combined) * 1.08)
        overflow_count = 0
        for group, values, colour, alpha, linestyle, label in plotted:
            overflow_count += plot_with_marked_overflow(
                axis, optimizer_steps(group.center_step), values, cap,
                color=colour, alpha=alpha, linestyle=linestyle, linewidth=1.45, label=label,
            )
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(method, fontsize=10.5, fontweight="bold")
        axis.grid(alpha=0.2)
        axis.set_ylim(0, cap * 1.08)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=method in ("FNN", "Cao", "Simplex")))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.set_ylabel(dimension_label(method), fontsize=7.6, labelpad=3)
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        if overflow_count:
            axis.text(0.98, 0.96, f"▲ {overflow_count} above {cap:.1f}", transform=axis.transAxes,
                      ha="right", va="top", fontsize=6.8, color="#334155")
    flat[-1].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9, title="Encoding")
    fig.suptitle("Loss observables: estimator comparison", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, "Blue = train, red = validation; pale = raw, saturated = linearly detrended",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.015, 1, 0.92), h_pad=2.1, w_pad=1.4)
    save(fig, "wd0_corrected_02_all_methods_loss")


def plot_mg_readable_panels(
    windows: pd.DataFrame,
    metrics: tuple[str, ...],
    meta: dict,
    name: str,
    title: str,
) -> None:
    """Create a report-sized MG small-multiple figure (at most six panels)."""
    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11.7, 2.8 * rows), sharex=True)
    flat = np.asarray(axes).reshape(-1)
    for index, (axis, metric) in enumerate(zip(flat, metrics)):
        group = windows[windows.metric == metric]
        axis.plot(
            optimizer_steps(group.center_step), group.mg_detrended,
            color=COLORS["MG"], linewidth=1.65, label="MG, detrended",
        )
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(LABEL[metric], fontsize=10, fontweight="bold")
        format_dimension_axis(axis, dimension_label("MG"))
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        axis.grid(alpha=0.2)
    for axis in flat[len(metrics):]:
        axis.axis("off")
    flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


def plot_lb_mg(windows: pd.DataFrame, metrics: tuple[str, ...], meta: dict, name: str, title: str) -> None:
    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11.7, 2.7 * rows), sharex=True)
    flat = np.asarray(axes).reshape(-1)
    # Main figure: pooled MG only, because overlaying it with unstable
    # arithmetic LB creates a meaningless shared scale.
    for index, (axis, metric) in enumerate(zip(flat, metrics)):
        group = windows[windows.metric == metric]
        axis.plot(optimizer_steps(group.center_step), group.mg_detrended, color=COLORS["MG"], label="MG pooled", linewidth=1.45)
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(LABEL[metric], fontsize=9.5)
        format_dimension_axis(axis, dimension_label("MG"))
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        axis.grid(alpha=0.2)
    for axis in flat[len(metrics) :]:
        axis.axis("off")
    flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title + ": MacKay–Ghahramani pooled estimate; detrended windows")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name)

    # Companion arithmetic-LB figure with a declared robust display limit.
    fig, axes = plt.subplots(rows, cols, figsize=(11.7, 2.7 * rows), sharex=True)
    flat = np.asarray(axes).reshape(-1)
    for index, (axis, metric) in enumerate(zip(flat, metrics)):
        group = windows[windows.metric == metric]
        values = group.lb_detrended.to_numpy(float)
        cap = robust_upper(values)
        count = plot_with_marked_overflow(
            axis, optimizer_steps(group.center_step), values, cap,
            color=COLORS["LB"], label="LB arithmetic", linewidth=1.25,
        )
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(LABEL[metric], fontsize=9.5)
        axis.set_ylim(0, cap * 1.08)
        format_dimension_axis(axis, dimension_label("LB"))
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        axis.grid(alpha=0.2)
        if count:
            axis.text(0.98, 0.96, f"▲ {count} above {cap:.1f}", transform=axis.transAxes,
                      ha="right", va="top", fontsize=6.5, color=COLORS["LB"])
    for axis in flat[len(metrics):]:
        axis.axis("off")
    flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title + ": Levina–Bickel arithmetic estimate; triangles mark values above the displayed range")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name + "_lb_diagnostic")


def plot_raw_detrended(windows: pd.DataFrame, meta: dict) -> None:
    metrics = ("train_loss", "val_loss", "weight_norm", "gradient_norm")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.5), sharex=True)
    for index, (axis, metric) in enumerate(zip(axes.flat, metrics)):
        group = windows[windows.metric == metric]
        axis.plot(optimizer_steps(group.center_step), group.mg_raw, color="#FCA5A5", label="MG raw", linewidth=1.1)
        axis.plot(optimizer_steps(group.center_step), group.mg_detrended, color=COLORS["MG"], label="MG detrended", linewidth=1.4)
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(LABEL[metric])
        format_dimension_axis(axis, dimension_label("MG"))
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("MacKay–Ghahramani estimate: raw versus linearly detrended windows")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "wd0_corrected_05_raw_vs_detrended")


def period_summary(windows: pd.DataFrame) -> pd.DataFrame:
    centers = np.sort(windows.center_step.unique())
    chunks = np.array_split(centers, 3)
    periods = {
        "early": chunks[0],
        "middle": chunks[1],
        "late": chunks[2],
    }
    rows = []
    for metric in CORE + PROJECTIONS:
        group = windows[windows.metric == metric]
        for period, selected_centers in periods.items():
            selected = group[group.center_step.isin(selected_centers)]
            rows.append(
                {
                    "metric": metric,
                    "period": period,
                    "first_center_step": int(selected_centers[0]),
                    "last_center_step": int(selected_centers[-1]),
                    "windows": int(len(selected)),
                    "lb_detrended_median": float(selected.lb_detrended.median()),
                    "lb_detrended_iqr": float(selected.lb_detrended.quantile(0.75) - selected.lb_detrended.quantile(0.25)),
                    "mg_detrended_median": float(selected.mg_detrended.median()),
                    "mg_detrended_iqr": float(selected.mg_detrended.quantile(0.75) - selected.mg_detrended.quantile(0.25)),
                }
            )
    return pd.DataFrame(rows)


def plot_period_heatmap(summary: pd.DataFrame) -> None:
    metrics = CORE
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.4), sharey=True)
    for axis, method, title in (
        (axes[0], "lb", "LB median (linear colour scale)"),
        (axes[1], "mg", "MG median (linear colour scale)"),
    ):
        matrix = np.array(
            [
                [summary[(summary.metric == metric) & (summary.period == period)][f"{method}_detrended_median"].iloc[0] for period in ("early", "middle", "late")]
                for metric in metrics
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(3), ["early", "middle", "late"])
        axis.set_yticks(range(len(metrics)), [LABEL[m] for m in metrics])
        axis.tick_params(axis="y", labelleft=True, labelsize=7.5)
        axis.set_xlabel("observation period [equal window-count thirds]")
        axis.set_ylabel("logged scalar observable [metric]")
        axis.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center", fontsize=7,
                          color="white" if matrix[i, j] > np.nanmedian(matrix) else "black")
        fig.colorbar(image, ax=axis, shrink=0.76, label="median intrinsic dimension $d$ [dimensions]")
    fig.suptitle("Temporal summary after memorization: descriptive periods, not grokking phases")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "wd0_corrected_06_temporal_summary")


def plot_robustness(meta: dict) -> None:
    path = ANALYSIS / "window_robustness_lb_mg.csv"
    robust = pd.read_csv(path)
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.3), sharex=True)
    for index, (axis, metric) in enumerate(zip(axes.flat, ("train_loss", "val_loss", "weight_norm", "gradient_norm"))):
        for size in sorted(robust.window_size.unique()):
            group = robust[(robust.metric == metric) & (robust.window_size == size)]
            window_steps = int(size * meta["logging"]["median_step_interval"])
            axis.plot(
                optimizer_steps(group.center_step), group.mg, linewidth=1.0,
                label=f"MG, W={size} rows ({window_steps:,} steps)",
            )
        add_memo(axis, int(meta["stable_memorization_flag_step"]))
        axis.set_title(LABEL[metric])
        format_dimension_axis(axis, dimension_label("MG"))
        format_step_axis(axis, int(meta["last_step"]), window_center=True)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle("Window-length sensitivity of MG estimate; detrended windows")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "wd0_corrected_07_window_robustness")


def setup_fonts() -> None:
    candidates = [
        ("Arial", Path(r"C:\Windows\Fonts\arial.ttf")),
        ("Arial-Bold", Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ]
    for name, path in candidates:
        if not path.exists():
            raise FileNotFoundError(path)
        pdfmetrics.registerFont(TTFont(name, str(path)))


def build_pdf(meta: dict, summary: pd.DataFrame) -> None:
    setup_fonts()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyRu", parent=styles["BodyText"], fontName="Arial", fontSize=9.1, leading=12.0, alignment=TA_JUSTIFY, spaceAfter=2.3 * mm)
    small = ParagraphStyle("SmallRu", parent=body, fontSize=7.5, leading=9.2, alignment=TA_LEFT)
    title = ParagraphStyle("TitleRu", parent=styles["Title"], fontName="Arial-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#16253d"), alignment=TA_CENTER, spaceAfter=5 * mm)
    subtitle = ParagraphStyle("SubtitleRu", parent=body, fontSize=10.8, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#41546f"), spaceAfter=7 * mm)
    h1 = ParagraphStyle("H1Ru", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#163a68"), spaceBefore=1 * mm, spaceAfter=3 * mm)
    h2 = ParagraphStyle("H2Ru", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#254f7d"), spaceBefore=2 * mm, spaceAfter=2 * mm)
    caption = ParagraphStyle("CaptionRu", parent=small, fontSize=7.2, leading=9.0, textColor=colors.HexColor("#45566e"), spaceBefore=1 * mm, spaceAfter=3 * mm)
    callout = ParagraphStyle("CalloutRu", parent=body, fontSize=9.7, leading=12.7, textColor=colors.HexColor("#17365d"), leftIndent=4 * mm, rightIndent=4 * mm, borderColor=colors.HexColor("#98b6d4"), borderWidth=0.8, borderPadding=3.5 * mm, backColor=colors.HexColor("#edf5fc"), spaceBefore=2 * mm, spaceAfter=4 * mm)

    def p(text: str, style=body) -> Paragraph:
        return Paragraph(text, style)

    def image(name: str, width=181 * mm, height=None) -> Image:
        path = ANALYSIS / name
        obj = Image(str(path))
        if height is None:
            height = width * obj.imageHeight / obj.imageWidth
        obj.drawWidth = width
        obj.drawHeight = height
        return obj

    def page_header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.3)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(14 * mm, 9 * mm, "EDM-анализ: p=211, WD=0, non-grokking ablation")
        canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"стр. {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="Подробный EDM-анализ modular addition p=211, WD=0",
        author="Codex / 2026-Project-202",
        subject="Non-grokking ablation; EDM; intrinsic dimension; Levina-Bickel; MacKay-Ghahramani",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=page_header)])

    overview = [
        ["Параметр", "Значение"],
        ["Исходный лог", meta["source_file"]],
        ["Weight decay", "0.0 (фиксированный)"],
        ["Строк / шагов", f'{meta["rows"]} / {meta["first_step"]}–{meta["last_step"]}'],
        ["Стабильная меморизация", str(meta["stable_memorization_flag_step"])],
        ["Grokking", "не произошёл"],
        ["Финальная accuracy", f'train={meta["final_train_acc"]:.3f}, val={meta["final_val_acc"]:.3f}'],
        ["Лучшая val accuracy", f'{meta["best_val_acc"]:.4f}'],
        ["Финальный val loss", f'{meta["final_val_loss"]:.3f}'],
    ]
    overview_table = Table(overview, colWidths=[59 * mm, 100 * mm], repeatRows=1, hAlign="CENTER")
    overview_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 8.5),
        ("FONT", (0, 0), (-1, 0), "Arial-Bold", 8.7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aebdca")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story = [
        Spacer(1, 10 * mm),
        p("Подробный EDM-анализ", title),
        p("Modular addition, <i>p</i>=211 · OmniGrok-style Transformer · AdamW · fixed weight decay 0.0", subtitle),
        p(
            "<b>Статус эксперимента:</b> отрицательный контроль. Модель быстро меморизует train, но не генерализует за 200 000 шагов. "
            "Этот запуск нельзя описывать как delayed generalization и в нём нет grokking transition или post-grok фазы.",
            callout,
        ),
        overview_table,
        Spacer(1, 5 * mm),
        p(
            f'EDM рассчитан с τ={TAU} строка лога (≈{meta["edm_protocol"]["tau_optimizer_steps"]} optimizer steps), '
            f'W={WINDOW_ROWS} строк (≈{meta["edm_protocol"]["window_optimizer_steps_approx"]} шагов), '
            f'stride={WINDOW_STRIDE_ROWS} строк (≈{meta["edm_protocol"]["stride_optimizer_steps"]} шагов), '
            f'E={E}, k={K}. Перекрытие соседних окон равно 50%. В каждом detrended-окне независимо удалён линейный тренд.',
        ),
        p(f'<b>Контроль происхождения:</b> SHA-256 исходного CSV: <font size="6.6">{meta["source_sha256"]}</font>.', small),
        PageBreak(),
        p("1. Траектория обучения", h1),
        image("wd0_corrected_01_training_overview.png", 181 * mm, 133 * mm),
        p("Рис. 1. Сырые метрики обучения. Пунктир отмечает stable memorization; линии grokking нет, поскольку событие отсутствует.", caption),
        p(
            f'Train accuracy стабилизируется выше 0.99 около шага {meta["train_acc_099_stable5_onset"]}, однако validation accuracy остаётся около случайного уровня: '
            f'максимум {meta["best_val_acc"]:.4f}, финал {meta["final_val_acc"]:.4f}. Validation loss возрастает до {meta["final_val_loss"]:.1f}. '
            f'Weight norm, в отличие от WD=0.5, растёт с {meta["initial_weight_norm"]:.1f} до {meta["final_weight_norm"]:.1f}.',
        ),
        p(
            "Следовательно, это режим memorization without generalization. В дальнейшем временная ось делится только на ранний, средний и поздний участки наблюдения; эти участки не называются fitting/gap/transition/post-grok.",
        ),
        PageBreak(),
        p("2. Что именно оценивает EDM", h1),
        p(
            "Из каждой одномерной метрики x(t) строится delay embedding: X(t)=[x(t), x(t−τ), …, x(t−(E−1)τ)]. "
            "EDM-оценка характеризует локальную геометрию облака таких задержанных векторов, а не число параметров сети и не ранг отдельной матрицы.",
        ),
        p(
            "В отчёте сохранены FNN, Cao, Simplex и арифметическая локальная MLE Levina–Bickel. MacKay–Ghahramani (MG) использует pooled likelihood: сначала усредняется локальная обратная размерность, затем результат инвертируется. "
            "MG менее чувствительна к отдельным почти нулевым log-distance sums, из-за которых арифметическая LB может взрываться.",
        ),
        p(
            "Абсолютные значения следует считать диагностическими: соседние delay vectors перекрываются, Theiler exclusion не использован, а W=200 оставляет лишь 186 векторов при E=15. Поэтому основной объект сравнения — устойчивое изменение кривой при фиксированном протоколе, а не буквальное число активных параметров.",
            callout,
        ),
        p("3. Все методы на train/validation loss", h1),
        image("wd0_corrected_02_all_methods_loss.png", 181 * mm, 130 * mm),
        p("Figure 2. Each panel is one estimator. Colour distinguishes train/validation; saturation distinguishes raw/detrended. X: EDM window centre in 10^3 optimizer updates. FNN/Cao/Simplex Y: embedding dimension E; LB/MG Y: intrinsic dimension d. All Y axes are linear; triangles mark values above a labelled display cap.", caption),
        p(
            "К концу запуска MG-ID обеих loss-метрик снижается, однако validation accuracy не растёт. Это напрямую показывает, что dimension drop не является достаточным условием grokking: подобный сигнал может сопровождать насыщение, численное вырождение loss или упрощение наблюдаемой скалярной динамики без генерализации.",
        ),
        PageBreak(),
        p("4. Основные одномерные наблюдатели", h1),
        image("wd0_corrected_03a_core_primary.png", 181 * mm, 184 * mm),
        p("Figure 3a. Primary MG-ID curves for detrended windows. X: EDM window centre in 10^3 optimizer updates; Y: intrinsic dimension d on a linear scale. Arithmetic LB remains in separate diagnostic files and does not distort the MG scale.", caption),
        PageBreak(),
        p("4. Основные одномерные наблюдатели — продолжение", h1),
        image("wd0_corrected_03b_core_secondary.png", 181 * mm, 123 * mm),
        p("Figure 3b. Participation-ratio and entropy observers. Splitting the ten panels across two pages keeps labels legible without changing data or scales.", caption),
        PageBreak(),
        p("5. Интерпретация основных наблюдателей", h1),
        p(
            "MG-ID показывает нисходящий временной тренд для train loss, validation loss, weight norm, gradient norm, parameter participation ratio и энтропий. "
            "Но gradient participation ratio остаётся примерно стационарным, а gradient cosine меняется без единого направленного коллапса. Получается не единый согласованный переход динамической системы, а метрико-зависимое упрощение части наблюдаемых рядов.",
        ),
        p(
            "Арифметическая LB периодически принимает огромные значения — до сотен и тысяч, а для отдельных raw/projection окон ещё выше. Это не физический рост размерности. Причина — локальные оценки вида 1/s, где s является малой суммой логарифмов отношений kNN-расстояний. Один почти нулевой s создаёт очень большой локальный d, после чего арифметическое среднее взрывается. MG агрегирует s до инверсии и потому остаётся устойчивее.",
        ),
        p("Медианные MG-ID по трём равным временным участкам", h2),
    ]

    table_data = [["Метрика", "Ранний", "Средний", "Поздний"]]
    for metric in ("train_loss", "val_loss", "weight_norm", "gradient_norm", "gradient_cosine", "gradient_participation_ratio"):
        vals = [summary[(summary.metric == metric) & (summary.period == period)].mg_detrended_median.iloc[0] for period in ("early", "middle", "late")]
        table_data.append([LABEL[metric]] + [f"{value:.2f}" for value in vals])
    med_table = Table(table_data, colWidths=[72 * mm, 28 * mm, 28 * mm, 28 * mm], repeatRows=1)
    med_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 8),
        ("FONT", (0, 0), (-1, 0), "Arial-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aebdca")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        med_table,
        Spacer(1, 4 * mm),
        p(
            "Эта таблица особенно важна для критерия будущего grokking: validation-loss MG-ID падает примерно с 3.76 до 1.77, хотя генерализации нет. Следовательно, предиктор не должен срабатывать только по условию d(t)&lt;threshold или Δd&lt;0.",
            callout,
        ),
        PageBreak(),
        p("6. Фиксированные случайные проекции", h1),
        image("wd0_corrected_04_projection_metrics_lb_mg.png", 181 * mm, 205 * mm),
        p("Рис. 4. MG-ID detrended loss-, weight-, gradient- и update-проекций. X — центр EDM-окна в 10³ optimizer updates; Y — intrinsic dimension d [dimensions] на линейной шкале.", caption),
        p(
            "Проекции не образуют согласованного набора предикторов. Одна фиксированная случайная проекция может быть почти ортогональна информативному направлению, а gradient projection дополнительно содержит mini-batch noise. "
            "Поэтому отдельный r0 нельзя трактовать как надёжное измерение глобального состояния; нужен ансамбль проекций и агрегирование устойчивости эффекта между r.",
        ),
        PageBreak(),
        p("7. Raw и detrended", h1),
        image("wd0_corrected_05_raw_vs_detrended.png", 181 * mm, 116 * mm),
        p("Рис. 5. MG-ID: светлые линии — raw, насыщенные — линейно detrended. X — центр EDM-окна в 10³ optimizer updates; Y — intrinsic dimension d [dimensions], линейная шкала.", caption),
        p(
            "Detrending выполнен независимо в каждом окне обычной least-squares прямой. Он удаляет только линейный дрейф уровня, но не исправляет насыщение, ступени, гетероскедастичность, выбросы или временную корреляцию. Raw и detrended являются разными диагностическими представлениями и в отчёте не смешиваются.",
        ),
        PageBreak(),
        p("8. Временное резюме без вымышленных фаз", h1),
        image("wd0_corrected_06_temporal_summary.png", 181 * mm, 94 * mm),
        p("Рис. 6. Медианы по ранней, средней и поздней третям доступных окон. Это описательные периоды non-generalizing trajectory, а не grokking phases.", caption),
        PageBreak(),
        p("9. Робастность к длине окна", h1),
        image("wd0_corrected_07_window_robustness.png", 181 * mm, 113 * mm),
        p("Рис. 7. MG для W∈{100,150,200,250} строк лога, то есть примерно {5, 7.5, 10, 12.5}×10³ optimizer updates. Все оценки detrended; X — центр окна в 10³ optimizer updates, Y — intrinsic dimension d [dimensions] на линейной шкале.", caption),
        p(
            "Качественный поздний спад MG наблюдается при нескольких W, но это не делает его grokking-specific: validation accuracy при этом остаётся низкой. Размер окна заметно меняет абсолютный уровень и временную локализацию оценки, поэтому W, stride, τ, E, k и logging cadence должны быть одинаковыми во всех сравниваемых запусках.",
        ),
        PageBreak(),
        p("10. Вывод для критерия раннего предсказания", h1),
        p(
            "Этот ablation опровергает слишком простой критерий «размерность упала ⇒ скоро grokking». Падение ID наблюдается и в запуске, который меморизовал train, но не начал генерализовать. Поэтому dimension drop разумно считать только одним компонентом составного критерия.",
            callout,
        ),
        p(
            "Рабочий визуальный критерий должен требовать одновременно: (1) устойчивого падения MG-ID на нескольких последовательных trailing windows; (2) согласованности хотя бы между несколькими заранее выбранными наблюдателями; (3) отсутствия объяснения простым насыщением или численным floor/ceiling; "
            "(4) отличия от распределения таких же признаков на non-grokking controls; (5) достаточного lead time до заранее фиксированного порога validation accuracy.",
        ),
        p(
            "Для текущего проекта особенно полезна контрастная пара WD=0.5 versus WD=0. В WD=0.5 падение размерности связано с резким validation transition; в WD=0 оно происходит без transition. Следующий статистический шаг — измерять не наличие падения само по себе, а форму, скорость, согласованность и специфичность падения относительно отрицательных контролей.",
        ),
        PageBreak(),
        p("11. Ограничения", h1),
        p(
            "1. Один seed не позволяет оценить межзапусковую вариативность.<br/>"
            "2. Центрированные окна используют данные по обе стороны center step; для честного online predictor нужны trailing windows.<br/>"
            "3. Theiler exclusion отсутствует, поэтому временно соседние delay vectors могут искусственно уменьшать расстояния.<br/>"
            "4. W=200 и E=15 дают только 186 delay vectors.<br/>"
            "5. Арифметическая LB нестабильна при почти нулевых log-distance sums; MG предпочтительнее как основная кривая.<br/>"
            "6. Участки early/middle/late выбраны описательно и не являются подтверждёнными динамическими фазами.<br/>"
            "7. EDM скалярного лога оценивает размерность наблюдаемой временной динамики, а не число активных параметров сети.",
        ),
        p("12. Файлы воспроизводимости", h1),
        p(
            f'Исходный лог: <b>{SOURCE.name}</b><br/>'
            f'EDM-окна: <b>{WINDOWS_CSV.name}</b><br/>'
            'Исправленное временное резюме: <b>wd0_corrected_temporal_summary.csv</b><br/>'
            'Аудит источников и параметров: <b>corrected_report_audit.json</b><br/>'
            f'Сборщик отчёта: <b>{Path(__file__).name}</b>',
        ),
        p("Итог", h2),
        p(
            "WD=0 является чистым non-grokking control: сеть меморизует train, validation остаётся около случайного уровня, а weight norm растёт. Некоторые MG-ID кривые при этом заметно снижаются. "
            "Следовательно, падение размерности — потенциально полезный, но неспецифичный индикатор перестройки наблюдаемой динамики; для прогноза grokking необходим контроль ложных срабатываний и составной критерий.",
        ),
    ]
    doc.build(story)


def main() -> None:
    frame = pd.read_csv(SOURCE).sort_values("step").reset_index(drop=True)
    windows = pd.read_csv(WINDOWS_CSV)
    meta = validate_inputs(frame, windows)
    summary = period_summary(windows)
    summary.to_csv(ANALYSIS / "wd0_corrected_temporal_summary.csv", index=False)
    AUDIT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_training(frame, meta)
    plot_methods(windows, meta)
    plot_lb_mg(windows, CORE, meta, "wd0_corrected_03_core_metrics_lb_mg", "Core scalar observers: LB and MG")
    plot_mg_readable_panels(
        windows, CORE[:6], meta, "wd0_corrected_03a_core_primary",
        "Core observers: pooled MG estimate, detrended windows",
    )
    plot_mg_readable_panels(
        windows, CORE[6:], meta, "wd0_corrected_03b_core_secondary",
        "Participation-ratio and entropy observers: pooled MG estimate",
    )
    plot_lb_mg(windows, PROJECTIONS, meta, "wd0_corrected_04_projection_metrics_lb_mg", "Fixed random projections: LB and MG")
    plot_raw_detrended(windows, meta)
    plot_period_heatmap(summary)
    plot_robustness(meta)
    build_pdf(meta, summary)
    print(OUTPUT)


if __name__ == "__main__":
    main()

"""Figures for the grokking / dimensionality-collapse experiments.

Three families, matching the figures in ``icomp_article/grokking_en.tex``:

* :func:`plot_dimension_vs_accuracy` -- the diagnostic d_hat(t) vs. Train/Val
  accuracy plot (Fig. 3 and Fig. 5 of the paper).
* :func:`plot_presentation_panels` -- the three-panel set ``*_acc`` / ``*_loss``
  / ``*_norm`` used in Figs. 1 and 2.
* :func:`plot_smoothed_accuracy` / :func:`plot_smoothed_dynamics` -- the
  memorization-vs-generalization overview with t_mem / t_gen markers (Fig. 4a).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PAPER_STYLE = {
    "font.size": 14,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}

COLOR_E = "tab:purple"
COLOR_TRAIN = "tab:blue"
COLOR_VAL = "tab:red"
COLOR_NORM = "tab:green"

SOURCE_LABELS = {
    "weight_norm": r"$\|w\|_2$",
    "val_loss": r"$\mathcal{L}_{val}$",
    "train_loss": r"$\mathcal{L}_{train}$",
    "grad_norm": r"$\|g\|_2$",
    "embed_grad_norm": r"$\|g_{embed}\|_2$",
    "grad_cosine": r"$\cos(g_t, g_{t-1})$",
    "val_acc": "val accuracy",
    "train_acc": "train accuracy",
}


def source_label(metric):
    """Human-readable name of the 1D series a dimension estimate was built from."""
    return SOURCE_LABELS.get(metric, metric.replace("_", " "))


def add_accuracy_axis(ax, df, alpha_train=0.35, alpha_val=0.55, linewidth=2):
    """Twin axis carrying both accuracies -- val alone hides where memorization ended."""
    twin = ax.twinx()
    train, = twin.plot(df["step"], df["train_acc"], color=COLOR_TRAIN,
                       alpha=alpha_train, linewidth=linewidth, label="Train Acc")
    val, = twin.plot(df["step"], df["val_acc"], color=COLOR_VAL,
                     alpha=alpha_val, linewidth=linewidth, label="Val Acc")
    twin.set_ylabel("Accuracy", fontsize=12)
    twin.set_ylim(-0.05, 1.05)
    twin.grid(False)

    # Accuracy is context, the dimension curve is the subject: keep the host axis
    # (and its legend) on top of the twin instead of behind it.
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    return twin, [train, val]


E_SERIES_STYLES = (
    {"color": COLOR_E, "linestyle": "-"},        # Levina-Bickel (the paper's estimator)
    {"color": "#3b0f70", "linestyle": "-."},     # MacKay-Ghahramani pooling
    {"color": "#00838f", "linestyle": ":"},      # + Theiler window (different assumption)
    {"color": "#8c564b", "linestyle": (0, (5, 1, 1, 1, 1, 1))},
)
E_SERIES_MARKERS = ("o", "^", "s", "D")


def _save(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_traces(traces):
    """Accept a single ``DimensionTrace`` or a sequence of them."""
    return [traces] if hasattr(traces, "dimension") else list(traces)


def _dimension_axis(ax, traces, linewidth=3, markersize=4, solo_label="$E$ (MLE)"):
    """Draw every d_hat(t) on ``ax`` and scale the axis to span all of them.

    A single trace keeps the paper's plain ``$E$ (MLE)`` label; several traces are
    labelled by estimator so the corrections can be told apart.
    """
    traces = _as_traces(traces)
    lines = []
    for trace, style, marker in zip(traces, E_SERIES_STYLES, E_SERIES_MARKERS):
        label = solo_label if len(traces) == 1 else f"$E$ ({trace.label})"
        line, = ax.plot(
            trace.steps, trace.dimension,
            linewidth=linewidth, marker=marker, markersize=markersize, label=label, **style,
        )
        lines.append(line)

    ax.tick_params(axis="y", labelcolor=COLOR_E)
    valid = np.concatenate([t.dimension[~np.isnan(t.dimension)] for t in traces])
    if len(valid):
        ax.set_ylim(valid.min() - 0.05, valid.max() + 0.05)
    return lines


def _mark_grokking(ax, grok_step, y, text_x, label="Grokking", fontsize=12):
    ax.axvline(x=grok_step, color="black", linestyle=":", linewidth=2)
    ax.annotate(
        label, xy=(grok_step, y), xytext=(text_x, y),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6),
        fontsize=fontsize, fontweight="bold",
    )


def plot_dimension_vs_accuracy(traces, df, save_path, title=None, grok_threshold=0.95,
                               annotate_source=True):
    """Effective dimensionality against Train/Val accuracy (twin axes)."""
    traces = _as_traces(traces)
    steps = df["step"].to_numpy(dtype=np.float64)
    solo = traces[0] if len(traces) == 1 else None
    legend_loc = "center left" if solo else "best"

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Optimization Steps", fontsize=12)
    ylabel = (f"Effective Dimensionality $E$ ({solo.method})" if solo
              else "Effective Dimensionality $E$")
    if annotate_source:
        ylabel += "\nestimated from " + source_label(traces[0].metric)
    ax1.set_ylabel(ylabel, color=COLOR_E, fontsize=12, fontweight="bold")
    lines_E = _dimension_axis(
        ax1, traces, linewidth=2.5, markersize=5, solo_label="Dimensionality $E$"
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="black", fontsize=12)
    line_tr, = ax2.plot(steps, df["train_acc"], color=COLOR_TRAIN, alpha=0.3, label="Train Acc")
    line_val, = ax2.plot(steps, df["val_acc"], color=COLOR_VAL, linewidth=2.5, label="Val Acc")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_xlim(steps.min(), steps.max())
    ax2.set_xlim(steps.min(), steps.max())

    grok_mask = df["val_acc"].to_numpy() >= grok_threshold
    if np.any(grok_mask):
        grok_step = steps[grok_mask][0]
        _mark_grokking(
            ax2, grok_step, 0.2,
            text_x=min(grok_step + 500, steps.max() - 2000),
            label="Grokking Point", fontsize=11,
        )

    lines = [*lines_E, line_tr, line_val]
    ax1.legend(lines, [l.get_label() for l in lines], loc=legend_loc, framealpha=0.9)

    if title is None:
        methods = solo.method if solo else " vs. ".join(t.label for t in traces)
        title = (
            "Collapse of Dimensionality during Grokking\n"
            f"Method: {methods} (Metric: {traces[0].metric}, "
            f"W={traces[0].meta.get('window_size')})"
        )
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_presentation_panels(traces, df, outdir, prefix, grok_threshold=0.95,
                             annotate_source=True):
    """Write ``<prefix>_acc.png``, ``<prefix>_loss.png`` and ``<prefix>_norm.png``.

    Each panel shows the same d_hat(t) curve(s) against a different macroscopic
    observable, so the collapse can be read off against accuracy, loss and the
    weight norm independently. Passing several traces overlays them on one axis.
    """
    traces = _as_traces(traces)
    outdir = Path(outdir)
    steps = df["step"].to_numpy(dtype=np.float64)
    grok = df[df["val_acc"] >= grok_threshold]
    grok_step = None if grok.empty else float(grok["step"].iloc[0])
    text_x = None if grok_step is None else min(grok_step + 300, steps.max() - 2000)
    legend_loc = "center left" if len(traces) == 1 else "best"
    e_label = "Dimensionality $E$"
    if annotate_source:
        e_label += "\nestimated from " + source_label(traces[0].metric)
    written = {}

    with matplotlib.rc_context(PAPER_STYLE):
        # --- Dimensionality vs. loss ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlabel("Optimization Steps", fontsize=14, fontweight="bold")
        ax.set_ylabel(e_label, color=COLOR_E, fontsize=14, fontweight="bold")
        lines_E = _dimension_axis(ax, traces)

        ax_loss = ax.twinx()
        ax_loss.set_ylabel("Loss", color="black", fontsize=14, fontweight="bold")
        line_tr, = ax_loss.plot(steps, df["train_loss"], color=COLOR_TRAIN, alpha=0.3, label="Train Loss")
        line_val, = ax_loss.plot(steps, df["val_loss"], color=COLOR_VAL, linewidth=2.5, label="Val Loss")
        ax.set_xlim(steps.min(), steps.max())
        if grok_step is not None:
            _mark_grokking(ax_loss, grok_step, df["val_loss"].max() * 0.5, text_x)
        lines = [*lines_E, line_tr, line_val]
        ax.legend(lines, [l.get_label() for l in lines], loc=legend_loc, framealpha=0.9, fontsize=11)
        fig.tight_layout()
        written["loss"] = _save(fig, outdir / f"{prefix}_loss.png")

        # --- Dimensionality vs. weight norm ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlabel("Optimization Steps", fontsize=14, fontweight="bold")
        ax.set_ylabel(e_label, color=COLOR_E, fontsize=14, fontweight="bold")
        lines_E = _dimension_axis(ax, traces)

        ax_norm = ax.twinx()
        ax_norm.set_ylabel("Weight Norm", color=COLOR_NORM, fontsize=14, fontweight="bold")
        line_norm, = ax_norm.plot(
            steps, df["weight_norm"], color=COLOR_NORM, linewidth=2.5, linestyle="--", label="Weight Norm",
        )
        ax_norm.tick_params(axis="y", labelcolor=COLOR_NORM)
        norm_min, norm_max = df["weight_norm"].min(), df["weight_norm"].max()
        ax_norm.set_ylim(norm_min * 0.9, norm_max * 1.1)
        ax.set_xlim(steps.min(), steps.max())
        if grok_step is not None:
            _mark_grokking(ax_norm, grok_step, (norm_max + norm_min) / 2, text_x)
        lines = [*lines_E, line_norm]
        ax.legend(lines, [l.get_label() for l in lines], loc=legend_loc, framealpha=0.9, fontsize=11)
        fig.tight_layout()
        written["norm"] = _save(fig, outdir / f"{prefix}_norm.png")

        # --- Dimensionality vs. accuracy ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlabel("Optimization Steps", fontsize=14, fontweight="bold")
        ax.set_ylabel(e_label, color=COLOR_E, fontsize=14, fontweight="bold")
        lines_E = _dimension_axis(ax, traces)

        ax_acc = ax.twinx()
        ax_acc.set_ylabel("Accuracy", color="black", fontsize=14, fontweight="bold")
        line_tr, = ax_acc.plot(steps, df["train_acc"], color=COLOR_TRAIN, alpha=0.3, label="Train Acc")
        line_val, = ax_acc.plot(steps, df["val_acc"], color=COLOR_VAL, linewidth=2.5, label="Val Acc")
        ax_acc.set_ylim(-0.05, 1.05)
        ax.set_xlim(steps.min(), steps.max())
        if grok_step is not None:
            _mark_grokking(ax_acc, grok_step, 0.2, text_x)
        lines = [*lines_E, line_tr, line_val]
        ax.legend(lines, [l.get_label() for l in lines], loc=legend_loc, framealpha=0.9, fontsize=11)
        fig.tight_layout()
        written["acc"] = _save(fig, outdir / f"{prefix}_acc.png")

    return written


def _phase_markers(df, window=150, threshold=0.98):
    """Smoothed metrics plus the memorization / generalization steps of Def. 1."""
    smooth = {
        col: df[col].rolling(window=window, min_periods=1).mean()
        for col in ("train_acc", "val_acc", "train_loss", "val_loss")
        if col in df.columns
    }
    mem = df.loc[smooth["train_acc"] >= threshold, "step"]
    gen = df.loc[smooth["val_acc"] >= threshold, "step"]
    t_mem = None if mem.empty else int(mem.iloc[0])
    t_gen = None if gen.empty else int(gen.iloc[0])
    return smooth, t_mem, t_gen


def _draw_smoothed_accuracy(ax, df, smooth, t_mem, t_gen, title=None):
    steps = df["step"]
    ax.plot(steps, df["train_acc"], color=COLOR_TRAIN, alpha=0.15)
    ax.plot(steps, df["val_acc"], color=COLOR_VAL, alpha=0.15)
    ax.plot(steps, smooth["train_acc"], label="Train Acc (Smoothed)", color=COLOR_TRAIN, linewidth=2.5)
    ax.plot(steps, smooth["val_acc"], label="Val Acc (Smoothed)", color=COLOR_VAL, linewidth=3.0)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(loc="center right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(-0.05, 1.05)

    box = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
    if t_mem is not None:
        ax.axvline(x=t_mem, color=COLOR_TRAIN, linestyle=":", alpha=0.8, linewidth=2)
        ax.annotate(
            f"Memorization\n(Step {t_mem})", xy=(t_mem, 0.5), xytext=(t_mem + 500, 0.3),
            arrowprops=dict(facecolor=COLOR_TRAIN, shrink=0.05, width=1.5, headwidth=6),
            fontsize=11, color=COLOR_TRAIN, fontweight="bold", bbox=box,
        )
    if t_gen is not None:
        ax.axvline(x=t_gen, color=COLOR_VAL, linestyle=":", alpha=0.8, linewidth=2)
        ax.annotate(
            f"Grokking!\n(Step {t_gen})", xy=(t_gen, 0.5), xytext=(t_gen - 4000, 0.7),
            arrowprops=dict(facecolor=COLOR_VAL, shrink=0.05, width=1.5, headwidth=6),
            fontsize=11, color=COLOR_VAL, fontweight="bold", bbox=box,
        )


def plot_smoothed_accuracy(df, save_path, window=150, threshold=0.98, title=None):
    """Standalone smoothed-accuracy panel with the t_mem / t_gen markers."""
    smooth, t_mem, t_gen = _phase_markers(df, window=window, threshold=threshold)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw_smoothed_accuracy(ax, df, smooth, t_mem, t_gen, title=title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_smoothed_dynamics(df, save_path, window=150, threshold=0.98):
    """Two-panel overview: smoothed accuracy (left) and log-scale losses (right)."""
    smooth, t_mem, t_gen = _phase_markers(df, window=window, threshold=threshold)
    steps = df["step"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _draw_smoothed_accuracy(
        ax1, df, smooth, t_mem, t_gen,
        title="Accuracy: Memorization vs Generalization\n(Slingshot Effect visible in background)",
    )

    ax2.plot(steps, df["train_loss"], color=COLOR_TRAIN, alpha=0.15)
    ax2.plot(steps, df["val_loss"], color=COLOR_VAL, alpha=0.15)
    ax2.plot(steps, smooth["train_loss"], label="Train Loss (Smoothed)", color=COLOR_TRAIN, linewidth=2.5)
    ax2.plot(steps, smooth["val_loss"], label="Val Loss (Smoothed)", color=COLOR_VAL, linewidth=3.0)
    ax2.set_title("Loss Dynamics (Log Scale)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Optimization Steps", fontsize=12)
    ax2.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6, which="both")

    fig.tight_layout()
    return _save(fig, save_path)

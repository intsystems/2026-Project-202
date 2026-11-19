"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiment_runner.plotting import CODE, BAND, FAINT, GREY, POINTER, STOCHASTIC, TRANSIENT, bounds
from experiment_runner.plotting import draw


def fig_map():
    """Where the training logs of section 7 fall in the plane of the two diagnostics."""
    tr = pd.read_csv(CODE / 'real/grokking_and_stability/transformer_grokking/logs/real_logs_summary.csv')
    tr = tr[tr.column == "weight_norm"]
    gr = pd.concat([pd.read_csv(HERE / 'logs/dimension_probe_summary.csv'),
                    pd.read_csv(HERE / 'logs/dimension_probe_summary_poly.csv')])
    gr = gr[gr.column == "train_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 2.05))

    # one highlighted target zone: stable in E and recurrent enough to embed
    ax.add_patch(mpl.patches.Rectangle(
        (8.0, 0.95), 500 - 8.0, 0.15, facecolor=BAND, alpha=0.08,
        edgecolor=BAND, lw=0.7, ls=(0, (3.5, 3)), zorder=0))
    ax.axvline(8.0, color=FAINT, lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.axhline(1.10, color=FAINT, lw=0.7, ls=(0, (2, 3)), zorder=0)

    # Which transformer runs carry no weight decay is a property of the run, so it is
    # taken from the run name the registry gives them. Splitting on rho_ident, as this
    # did before, worked only while the two groups fell either side of 1.15: after the
    # rerun both zero-decay runs read 1.77 and were drawn, and counted in the legend, as
    # mini-batch runs.
    undecayed = tr.run.astype(str).str.startswith("wd0")
    lo, hi = tr[undecayed], tr[~undecayed]

    # Coincident points are separated by a small horizontal offset, and only then: the
    # offset exists to show a multiplicity, so a group that has spread out on its own
    # gets its true abscissa and no annotation. The archived campaign put ten perceptron
    # runs and both zero-decay runs on one mark at two crossings.
    OFF = 1.20
    piled = [g for g in (gr, lo) if len(g) > 1 and g.ident.round(2).nunique() == 1]
    shift = OFF if len(piled) > 1 else 1.0
    ax.plot(gr.osc / shift, gr.ident, "s", mfc="none", mec=TRANSIENT, mew=1.0,
            ms=6.5, ls="none", label=f"perceptron, full batch ({len(gr)})", zorder=3)
    ax.plot(hi.osc, hi.ident, "o", color=STOCHASTIC, ms=4.6, mec="white", mew=0.7,
            ls="none", label=f"transformer, mini-batch ({len(hi)})", zorder=4)
    ax.plot(lo.osc * shift, lo.ident, "D", color=TRANSIENT, ms=4.0, mec="white",
            mew=0.7, ls="none",
            label=f"transformer, no weight decay ({len(lo)})", zorder=5)
    if len(piled) > 1:
        y = float(pd.concat(piled).ident.median())
        ax.plot([2 / OFF, 2 * OFF], [y, y], "-", color=FAINT, lw=0.6, zorder=2)
        ax.plot([2, 2], [y - 0.018, y + 0.018], "-", color=GREY, lw=0.6, zorder=2)
        ax.text(2 / OFF / 1.16, y, rf"$\times {len(gr)}$", color=TRANSIENT,
                fontsize=7.0, ha="right", va="center")
        ax.text(2 * OFF * 1.13, y, rf"$\times {len(lo)}$", color=TRANSIENT,
                fontsize=7.0, ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.7, 500)
    # The admissible band is the reference every point is placed against, so it stays in
    # frame; the top follows the runs.
    low, high = bounds(gr.ident, tr.ident, pad=0.05, step=0.1, include=[0.95, 1.10])
    ax.set_ylim(low, high)
    ax.set_yticks(np.arange(1.0, high + 1e-9, 0.2))
    ax.set_xlabel("trend crossings per window")
    ax.set_ylabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    ax.text(430, 1.025, "accurate cases", color=BAND, fontsize=7.0, ha="right",
            va="center")

    fig.tight_layout(rect=[0, 0.095, 1, 1])
    # Three long entries at 9 pt fill the text width exactly, so the legend carries no
    # slack: the handles are short and the columns tight, and `figures.extent` measures
    # what is left.
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.3, handletextpad=0.28, columnspacing=0.6)
    return fig


def fig_pairs():
    """The label-matched pairs, and the training loss that explains them.

    All eight runs are full-batch perceptrons, which fig_map() classifies as
    transient, so the figure is monochrome by the palette rule: generalisation
    is carried by marker fill and line style, never by hue.
    """
    mg = pd.concat([pd.read_csv(HERE / 'logs/dimension_probe_summary.csv'),
                    pd.read_csv(HERE / 'logs/dimension_probe_summary_poly.csv')])
    mg = mg[mg.column == "train_loss"].set_index("run").MG
    # the seven sliding windows behind each of those run-level medians
    win = pd.concat([pd.read_csv(HERE / 'logs/dimension_probe.csv'), pd.read_csv(HERE / 'logs/dimension_probe_poly.csv')])
    win = win[win.column == "train_loss"].groupby("run").MG

    pairs = [("g_p1_p97", "g_p1x_p97", r"$(4n_1{+}n_2^2)^3$"),
             ("g_p2_p97", "g_p2x_p97", r"$(2n_1{+}3n_2)^4$"),
             ("g_p3_p97", "g_p3x_p97", r"$(5n_1^3{+}2n_2^4)^2$"),
             ("a_add", "x_no_grok", r"$n{+}m$")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.10),
                                 gridspec_kw={"width_ratios": [1.0, 1.05]})

    def spread(run, y):
        v = win.get_group(run)
        # thin line the full range of the seven windows, thick line the middle
        # half, marker the median that the run-level summary reports
        ax.plot([v.min(), v.max()], [y, y], "-", color=TRANSIENT, lw=0.6,
                alpha=0.50, zorder=2)
        ax.plot([v.quantile(0.25), v.quantile(0.75)], [y, y], "-",
                color=TRANSIENT, lw=2.4, alpha=0.25, solid_capstyle="butt",
                zorder=2)

    for i, (good, bad, name) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ax.plot([mg[good], mg[bad]], [y + 0.19, y - 0.19], color=FAINT, lw=0.7,
                zorder=1)
        spread(good, y + 0.19)
        spread(bad, y - 0.19)
        ax.plot(mg[good], y + 0.19, "o", color=TRANSIENT, ms=4.4, mec="white",
                mew=0.7, zorder=3)
        ax.plot(mg[bad], y - 0.19, "o", mfc="white", mec=TRANSIENT, mew=1.1,
                ms=4.4, zorder=3)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([n for _, _, n in pairs][::-1])
    ax.set_ylim(-0.72, len(pairs) - 0.28)
    drawn = [win.get_group(r) for pair in pairs for r in pair[:2]]
    low, high = bounds(*drawn, pad=0.05, step=1.0)
    ax.set_xlim(low, high)
    ax.set_xticks(np.arange(low, high + 1e-9, 1.0))
    ax.set_xlabel("estimate on the training loss")
    ax.set_title("(a) four label-matched pairs", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # panel (b) plots the g_p2 pair, the second row of panel (a): its two members
    # end three orders of magnitude apart, which the top pair does not (2.7)
    for key, ls in [("g_p2_p97", "-"), ("g_p2x_p97", (0, (3.5, 2)))]:
        t = pd.read_csv(HERE / "logs" / f"{key.removesuffix('_p97')}_train.csv")
        bx.plot(t.step, t.train_loss, linestyle=ls, color=TRANSIENT, lw=1.1)
    bx.set_yscale("log")
    bx.set_xlim(0, 100000)
    bx.set_xticks([0, 50000, 100000])
    bx.set_xticklabels(["0", "50k", "100k"])
    bx.set_xlabel("step")
    bx.set_ylabel("training loss")
    bx.set_title(r"(b) the $(2n_1{+}3n_2)^4$ pair", loc="left")

    handles = [
        Line2D([], [], color=TRANSIENT, ls="-", marker="o", ms=4.4,
               mec="white", mew=0.7, label="generalises"),
        Line2D([], [], color=TRANSIENT, ls=(0, (3.5, 2)), marker="o",
               mfc="white", mec=TRANSIENT, mew=1.1, ms=4.4, label="does not"),
    ]
    fig.tight_layout(rect=[0, 0.085, 1, 1], w_pad=1.6)
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=2.0)
    return fig


def fig_prwindow():
    """The full-batch participation ratio against window length.

    Both runs are full-batch perceptrons, which fig_map() classifies as
    transient, so the figure is monochrome: the generalising run is filled and
    solid, its label-matched control open and dashed.
    """
    d = pd.read_csv(HERE / 'logs/pr_vs_window.csv')
    # the paper's ladder: sixty samples spread further apart, so the bound on the
    # statistic and its noise floor stay where the published measurement put
    # them and only the span changes. The other ladder agrees; see the caption.
    d = d[d.ladder == "fixed_n"].sort_values("window_steps")

    PUB = 600.0  # the published window of section 7, not in this file
    runs = [("a_add", "generalises", dict(marker="o", ls="-", color=TRANSIENT,
                                          ms=3.4, mec="white", mew=0.6)),
            ("x_no_grok", "label-matched control",
             dict(marker="o", ls=(0, (3.5, 2)), color=TRANSIENT, ms=3.4,
                  mfc="white", mec=TRANSIENT, mew=1.0))]
    panels = [("PR_pos_det_med", "(a) median over windows"),
              ("PR_pos_det_max", "(b) maximum over windows")]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        ax.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
        ax.axvline(PUB, color=GREY, lw=0.7, ls=(0, (2, 2.5)), zorder=1)
        for run, label, kw in runs:
            g = d[d.run == run]
            ax.plot(g.window_steps, g[col], lw=1.1, zorder=3, **kw)
        ax.set_xscale("log")
        ax.set_xlim(380, 260000)
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["$10^3$", "$10^4$", "$10^5$"])
        ax.set_xlabel("window length (optimiser steps)")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0.94, 2.48)
    axes[0].set_yticks([1.0, 1.5, 2.0])
    axes[0].set_ylabel("participation ratio")
    axes[0].text(PUB * 1.25, 2.42, "published", ha="left", va="top", **POINTER)

    handles = [Line2D([], [], lw=1.1, label=lbl, **kw) for _, lbl, kw in runs]
    fig.tight_layout(rect=[0, 0.085, 1, 1], w_pad=1.6)
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=2.0)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_map, fig_pairs, fig_prwindow,))

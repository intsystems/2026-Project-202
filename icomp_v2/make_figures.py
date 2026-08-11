"""Regenerate the article's figures from the committed result files.

    python make_figures.py

Writes PDF (for LaTeX) and PNG (for preview) into figures/.

Design notes, so that later edits do not undo them:

* Every figure is drawn at 5.5 in, the exact \\textwidth of the ICOMP style, with
  8 pt type, so LaTeX never rescales it and the type stays at 8 pt on the page.
* Colour encodes the paper's three dynamical regimes and nothing else:
  recurrent, stochastic, transient. Line style distinguishes variants inside a
  regime and marker shape distinguishes experimental settings, so identity is
  never carried by colour alone.
* The palette passes the categorical checks (OKLCH lightness band, chroma floor,
  Machado severity-1.0 protan and deutan separation on all pairs, normal-vision
  floor, WCAG contrast against the page) with no warnings.
* Where a point is an aggregate the spread is drawn with it: a low-alpha
  interquartile fill in the regime colour in fig_regimes(a), range and
  interquartile bars in fig_pairs(a), the two-hash-family disagreement as a
  band on the mean of fig_dip(a) and (b), and in fig_window(b) the 39,990-step
  window each point summarises, drawn as a horizontal bar. Do not remove them
  for tidiness. The fig_dip band is about 1 % of the value and is therefore
  nearly invisible: that is the finding, and the legend names it so that the
  reader knows it was drawn rather than omitted.
* Where points coincide, the multiplicity is made visible rather than hidden:
  fig_regimes(b) plots every raw value on a per-observer row, fig_map offsets
  the twelve runs that share (2 crossings, rho_ident = 1.00) and labels the
  offset, fig_window(c) gives each run its own sub-row so that the eight
  strides of a flat run stay countable. Any figure edit that reintroduces a
  single mark for many runs must also fix the caption.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical palette, in fixed order: one hue per dynamical regime.
RECURRENT = "#0072B2"
STOCHASTIC = "#D55E00"
TRANSIENT = "#5D3A9B"
GREY = "#666666"
BAND = "#0072B2"

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
})


def save(fig, stem):
    # No bbox="tight": it expands the canvas past 5.5 in when a legend is wider
    # than the axes, and LaTeX then scales the figure down and the type with it.
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print("wrote", stem)


# ------------------------------------------------------------------ figure 1
def fig_regimes():
    """Recovery and admissibility on the image-data system of section 5.4."""
    d = pd.read_csv(CODE / "active_dimension/results/e2_rank_sweep/sweep_raw.csv")
    d = d[(~d.eta_zero) & (~d.observer.isin(["acc_probe", "loss_step"]))]

    series = [
        ("qp", "recurrent, fast drive", RECURRENT, "-", "o"),
        ("qp_slow", "recurrent, slow drive", RECURRENT, "--", "s"),
        ("mixed", "recurrent $+$ weak noise", STOCHASTIC, "--", "^"),
        ("noise", "stochastic", STOCHASTIC, "-", "v"),
        ("gd", "transient", TRANSIENT, ":", "D"),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.72),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax.axhline(20, color=GREY, lw=0.8, ls=(0, (1, 2)), zorder=0)
    # a reference line, not a bound: no clamping is applied and a violated model
    # returns values above E. Placed away from x = 1, where the transient piles up.
    ax.text(2.3, 20.8, r"$E_{\max}$, not a bound", color=GREY, fontsize=6.5,
            ha="left", va="bottom")
    ax.plot([0.9, 8.3], [0.9, 8.3], color=GREY, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(8.6, 7.6, "truth", color=GREY, fontsize=6.5, ha="left", va="center")

    ident = {}
    for arm, label, c, ls, mk in series:
        g = d[d.arm == arm]
        if g.empty:
            continue
        t = g.groupby("r").traj_PR.median().values
        y = g.groupby("r").MG.median().values
        # interquartile band over the forty values behind each point (four seeds
        # x ten observers). Sorted in x only so that the transient, whose seven
        # points all sit at x ~ 1.05 in no particular order, fills a blob rather
        # than a self-intersecting polygon; the other arms are already monotone.
        q1 = g.groupby("r").MG.quantile(0.25).values
        q3 = g.groupby("r").MG.quantile(0.75).values
        o = np.argsort(t)
        ax.fill_between(t[o], q1[o], q3[o], color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(t, y, ls, color=c, marker=mk, ms=3.2, mec="white", mew=0.5,
                label=label, zorder=4, clip_on=False)
        # rho_ident is populated for seed 0, three observers and r in {2, 6} only,
        # so panel (b) plots the six raw values rather than an aggregate.
        ident[arm] = g.dropna(subset=["ident_ratio"])[["observer", "r",
                                                       "ident_ratio"]]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 9.5)
    ax.set_ylim(0.6, 42)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("measured effective rank")
    ax.set_ylabel("estimated dimension")
    ax.set_title("(a) what the estimator returns", loc="left", pad=6)

    bx.axvspan(0.95, 1.10, color=BAND, alpha=0.10, lw=0, zorder=0)
    # one row of markers per observer, so that coincident values stay countable
    OBS_DY = {"c_proj1": 0.17, "g_fro": 0.0, "w_fro": -0.17}
    for i, (arm, label, c, ls, mk) in enumerate(series):
        v = ident.get(arm)
        if v is None or v.empty:
            continue
        y = len(series) - 1 - i
        bx.plot([v.ident_ratio.min(), v.ident_ratio.max()], [y, y], "-",
                color=c, lw=0.7, alpha=0.45, zorder=2)
        bx.plot(v.ident_ratio.values, y + v.observer.map(OBS_DY).values, mk,
                color=c, ms=3.4, mec="white", mew=0.5, zorder=3, clip_on=False)
    bx.set_yticks(range(len(series)))
    bx.set_yticklabels([lbl for _, lbl, _, _, _ in series][::-1])
    bx.set_ylim(-0.60, len(series) + 0.05)
    bx.set_xlim(0.92, 1.70)
    bx.set_xticks([1.0, 1.2, 1.4, 1.6])
    bx.set_xlabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    bx.set_title("(b) is it a count?", loc="left", pad=6)
    bx.text(1.025, 4.42, "admissible", color=BAND, fontsize=6.5, ha="center")
    bx.text(1.70, 4.79, "6 values per row: seed 0,\n"
                        r"3 observers, $r \in \{2, 6\}$",
            color=GREY, fontsize=5.5, ha="right", va="center", linespacing=1.35)
    # inside the axes, on the empty right of the transient row: outside it the
    # text ran into the spine and the x label
    bx.annotate("passes, yet returns 29\nwhere the measurement is 1",
                xy=(1.06, 0.0), xytext=(1.135, 0.0), fontsize=6.0,
                color=TRANSIENT, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=TRANSIENT, lw=0.9,
                                shrinkA=2, shrinkB=2))
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.2, mec="white",
                      mew=0.5, label=lbl) for _, lbl, c, ls, mk in series]
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2, columnspacing=1.2)
    save(fig, "fig_regimes")


# ------------------------------------------------------------------ figure 2
def fig_dip():
    """The trajectory-rank collapse at generalisation, aligned on t_gen."""
    d = pd.read_csv(CODE / "active_rank/results_fine/rank_windows.csv")
    meta = {m["run"]: m for m in
            json.loads((CODE / "active_rank/results_fine/rank_milestones.json").read_text())}

    groks = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    ctrls = ["mod_wd0", "s5_wd0"]
    panels = [("fn_PR_pos_det", "(a) function space", None),
              ("PR_pos_det", "(b) parameter space", None),
              ("move", "(c) displacement", "log")]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.90), sharex=True)
    grid = np.arange(-5000, 5200, 100)

    for ax, (col, title, scale) in zip(axes, panels):
        stack, sdstack = [], []
        # the two CountSketch hash families disagree by <col>_sketchsd per window;
        # 'move' is computed on the raw parameter vector and has no such column,
        # which is why panel (c) carries no band. Said in the caption.
        sdcol = col + "_sketchsd" if col + "_sketchsd" in d.columns else None
        for r in groks:
            g = d[d.run == r].sort_values("right_step")
            x = 0.5 * (g.right_step + g.left_step) - meta[r]["t_gen"]
            ax.plot(x, g[col], color=RECURRENT, lw=0.7, alpha=0.30)
            stack.append(np.interp(grid, x, g[col], left=np.nan, right=np.nan))
            if sdcol:
                sdstack.append(np.interp(grid, x, g[sdcol], left=np.nan,
                                         right=np.nan))
        m = np.nanmean(np.vstack(stack), axis=0)
        if sdcol:
            # the sketch uncertainty of the plotted mean, propagated as the mean
            # of the four per-run disagreements. It is about 1 % of the value, so
            # it is meant to be invisible at the scale of the collapse.
            s = np.nanmean(np.vstack(sdstack), axis=0)
            ax.fill_between(grid, m - s, m + s, facecolor=RECURRENT, alpha=0.55,
                            edgecolor=RECURRENT, lw=0.4, zorder=4)
        ax.plot(grid, m, color=RECURRENT,
                lw=1.7, label="generalises (4 runs)", zorder=5)

        for r, ls in zip(ctrls, ["--", (0, (1, 1.6))]):
            g = d[d.run == r].sort_values("right_step")
            ref = meta["mod_wd1" if r.startswith("mod") else "s5_wd1"]["t_gen"]
            ax.plot(0.5 * (g.right_step + g.left_step) - ref, g[col], linestyle=ls,
                    color=STOCHASTIC, lw=1.2,
                    label="no weight decay (2 runs)" if r == "mod_wd0" else None)

        ax.axvline(0, color=GREY, lw=0.9, ls=(0, (2, 2)))
        if scale:
            ax.set_yscale(scale)
        ax.set_xlim(-5000, 5000)
        ax.set_xticks([-5000, 0, 5000])
        ax.set_xticklabels(["-5k", "$t_{gen}$", "5k"])
        ax.set_title(title, loc="left", pad=4)

    axes[0].set_ylabel("participation ratio")
    axes[2].set_ylabel("displacement")
    axes[1].set_xlabel("steps since generalisation")
    h, l = axes[0].get_legend_handles_labels()
    # the band is drawn to scale and is therefore almost invisible, which is the
    # point; naming it in the legend costs no plot area and tells the reader it
    # is there. Panel (c) has none because 'move' is not a sketched statistic.
    h.append(mpl.patches.Patch(facecolor=RECURRENT, alpha=0.55, lw=0,
                               label="hash-family spread, (a) and (b)"))
    l.append("hash-family spread, (a) and (b)")
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=1.2, handletextpad=0.5)
    save(fig, "fig_dip")


# ------------------------------------------------------------------ figure 3
def fig_map():
    """Where the training logs of section 7 fall in the admissibility plane."""
    tr = pd.read_csv(CODE / "active_dimension/results/e5_real_logs/real_logs_summary.csv")
    tr = tr[tr.column == "weight_norm"]
    gr = pd.concat([
        pd.read_csv(CODE / "gromov_arithmetic/results/arith/dimension_probe_summary.csv"),
        pd.read_csv(CODE / "gromov_polynomials/results/dimension_probe_summary.csv")])
    gr = gr[gr.column == "train_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 2.35))

    # one highlighted target zone: stable in E and recurrent enough to embed
    ax.add_patch(mpl.patches.Rectangle(
        (8.0, 0.95), 500 - 8.0, 0.15, facecolor=BAND, alpha=0.10,
        edgecolor=BAND, lw=1.0, ls=(0, (4, 3)), zorder=0))
    ax.axvline(8.0, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)
    ax.axhline(1.10, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)

    hi = tr[tr.ident > 1.15]
    lo = tr[tr.ident <= 1.15]

    # All ten perceptron runs and both zero-weight-decay transformer runs sit at
    # exactly two crossings and rho_ident = 1.00 to three decimals, so without an
    # offset the twelve of them render as one mark. The offset is cosmetic and is
    # declared in the caption; the true abscissa of every point is the grey tick.
    OFF = 1.18
    ax.plot([2 / OFF, 2 * OFF], [0.999, 0.999], "-", color=GREY, lw=0.7,
            alpha=0.6, zorder=2)
    ax.plot([2, 2], [0.978, 1.020], "-", color=GREY, lw=0.7, alpha=0.6, zorder=2)
    ax.plot(gr.osc / OFF, gr.ident, "s", mfc="none", mec=TRANSIENT, mew=1.2, ms=9,
            ls="none", label="perceptron, full batch (10)", zorder=3)
    ax.plot(hi.osc, hi.ident, "o", color=STOCHASTIC, ms=5.5, mec="white", mew=0.8,
            ls="none", label="transformer, mini-batch (5)", zorder=4)
    ax.plot(lo.osc * OFF, lo.ident, "o", color=TRANSIENT, ms=5.5, mec="white",
            mew=0.8, ls="none", label="transformer, no weight decay (2)", zorder=5)
    ax.text(2 / OFF / 1.13, 0.999, r"$\times 10$", color=TRANSIENT, fontsize=6.5,
            ha="right", va="center")
    ax.text(2 * OFF * 1.11, 0.999, r"$\times 2$", color=TRANSIENT, fontsize=6.5,
            ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.7, 500)
    ax.set_ylim(0.85, 1.74)
    ax.set_yticks([0.9, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("trend crossings per window")
    ax.set_ylabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")

    ax.text(90, 1.025, "admissible: no run is here", color=BAND,
            fontsize=7, ha="center", va="center")
    ax.text(12, 1.63, "stochastic", color=STOCHASTIC, fontsize=7,
            ha="left", va="center")
    ax.text(12, 1.555, "no invariant set exists", color=STOCHASTIC,
            fontsize=6.5, ha="left", va="center")
    ax.text(0.78, 1.34, "transient", color=TRANSIENT, fontsize=7,
            ha="left", va="center")
    ax.text(0.78, 1.245, "monotone, so stable\nfor the wrong reason",
            color=TRANSIENT, fontsize=6.5, ha="left", va="center")
    ax.text(0.75, 0.908, "12 runs at exactly 2 crossings,\noffset to separate the groups",
            color=GREY, fontsize=6.0, ha="left", va="center")
    ax.text(7.0, 1.70, "too few crossings", color=GREY, fontsize=6.5,
            ha="right", va="top")
    ax.text(440, 1.135, "unstable in $E$", color=GREY, fontsize=6.5,
            ha="right", va="bottom")

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.44), handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig_map")


# ------------------------------------------------------------------ figure 4
def fig_pairs():
    """The label-matched pairs, and the training loss that explains them."""
    mg = pd.concat([
        pd.read_csv(CODE / "gromov_arithmetic/results/arith/dimension_probe_summary.csv"),
        pd.read_csv(CODE / "gromov_polynomials/results/dimension_probe_summary.csv")])
    mg = mg[mg.column == "train_loss"].set_index("run").MG
    # the seven sliding windows behind each of those run-level medians
    win = pd.concat([
        pd.read_csv(CODE / "gromov_arithmetic/results/arith/dimension_probe.csv"),
        pd.read_csv(CODE / "gromov_polynomials/results/dimension_probe.csv")])
    win = win[win.column == "train_loss"].groupby("run").MG

    pairs = [("g_p1_p97", "g_p1x_p97", r"$(4n_1{+}n_2^2)^3$"),
             ("g_p2_p97", "g_p2x_p97", r"$(2n_1{+}3n_2)^4$"),
             ("g_p3_p97", "g_p3x_p97", r"$(5n_1^3{+}2n_2^4)^2$"),
             ("a_add", "x_no_grok", r"$n{+}m$")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.35),
                                 gridspec_kw={"width_ratios": [1.0, 1.05]})

    def spread(run, y, colour):
        v = win.get_group(run)
        # thin line the full range of the seven windows, thick line the middle
        # half, marker the median that the run-level summary reports
        ax.plot([v.min(), v.max()], [y, y], "-", color=colour, lw=0.7,
                alpha=0.55, zorder=2)
        ax.plot([v.quantile(0.25), v.quantile(0.75)], [y, y], "-", color=colour,
                lw=2.6, alpha=0.30, solid_capstyle="butt", zorder=2)

    for i, (good, bad, name) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ax.plot([mg[good], mg[bad]], [y + 0.19, y - 0.19], color=GREY, lw=0.8,
                zorder=1)
        spread(good, y + 0.19, RECURRENT)
        spread(bad, y - 0.19, STOCHASTIC)
        ax.plot(mg[good], y + 0.19, "o", color=RECURRENT, ms=5.0, mec="white",
                mew=0.8, zorder=3, label="generalises" if i == 0 else None)
        ax.plot(mg[bad], y - 0.19, "o", color=STOCHASTIC, ms=5.0, mec="white",
                mew=0.8, zorder=3, label="does not" if i == 0 else None)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([n for _, _, n in pairs][::-1])
    ax.set_ylim(-0.75, len(pairs) - 0.25)
    ax.set_xlim(17.4, 28.0)
    ax.set_xticks([18, 20, 22, 24, 26, 28])
    ax.set_xlabel("estimate on the training loss")
    ax.set_title("(a) four label-matched pairs", loc="left", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # panel (b) plots the g_p2 pair, the second row of panel (a): its two members
    # end three orders of magnitude apart, which the top pair does not (2.7)
    for key, colour, ls, lbl in [
            ("g_p2_p97", RECURRENT, "-", "generalises"),
            ("g_p2x_p97", STOCHASTIC, "--", "does not")]:
        t = pd.read_csv(CODE / f"gromov_polynomials/results/{key}_train.csv")
        bx.plot(t.step, t.train_loss, ls, color=colour, lw=1.3, label=lbl)
    bx.set_yscale("log")
    bx.set_xlim(0, 100000)
    bx.set_xticks([0, 50000, 100000])
    bx.set_xticklabels(["0", "50k", "100k"])
    bx.set_xlabel("step")
    bx.set_ylabel("training loss")
    bx.set_title(r"(b) the $(2n_1{+}3n_2)^4$ pair", loc="left", pad=6)
    h, l = bx.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2)
    save(fig, "fig_pairs")


# ------------------------------------------------------------------ figure 5
def fig_window():
    """The windowed log estimate against t_gen: the change test of section 7.1.

    Colour is the regime as everywhere else. None of these seven runs is
    recurrent, so RECURRENT never appears here: the five regularised runs are
    stochastic and the two weight-decay-zero runs transient, exactly as
    fig_map() classifies them. Generalisation is carried by line style and
    marker, never by hue.
    """
    d = pd.read_csv(CODE / "active_dimension/results/e5_real_logs/real_logs_windows.csv")
    d = d[d.column == "weight_norm"].copy()
    out = pd.read_csv(CODE / "dimension_recovery/results/exp8_outcomes.csv")
    tgen = out.set_index("run").t_gen.to_dict()

    # The frozen configuration of section 7.1: window a third of the 12,000-sample
    # record, stride 1,000 samples, logging stride 10 optimiser steps. So a window
    # spans 39,990 steps and the centres are 10,000 steps apart: nine per run, and
    # a feature is located only to within half a span.
    SPAN = 39990.0
    HALF = SPAN / 2
    FLOOR = 1.16  # the ramped-gain nuisance bound of section 6.3
    d["centre"] = d.right_step - HALF

    # (regime colour, generalises, linestyle, marker)
    runs = [
        ("grokpos_s0",   STOCHASTIC, True,  "-",            "o"),
        ("lowdata20_s0", STOCHASTIC, True,  "-",            "^"),
        ("lowdata15_s0", STOCHASTIC, True,  "-",            "s"),
        ("lowdata15_s1", STOCHASTIC, False, "--",           "v"),
        ("lowdata15_s2", STOCHASTIC, False, "--",           "D"),
        ("wd0_s0",       TRANSIENT,  False, (0, (1, 1.6)),  "o"),
        ("wd0_s1",       TRANSIENT,  False, (0, (1, 1.6)),  "s"),
    ]

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(5.5, 2.2),
                                     gridspec_kw={"width_ratios": [1.0, 1.06, 0.82]})

    # ---- (a) the raw windowed estimate over training, every run --------------
    for r, c, gen, ls, mk in runs:
        g = d[d.run == r].sort_values("centre")
        ax.plot(g.centre / 1000, g.MG, linestyle=ls, color=c, marker=mk,
                ms=2.4, mec="white", mew=0.35, lw=1.0, alpha=0.95, zorder=3)
    # the three generalisation steps, marked on the axis rather than by hue
    for r, c, gen, ls, mk in runs:
        if gen:
            ax.plot([tgen[r] / 1000], [2.05], marker="^", color=GREY, ms=3.4,
                    mew=0, zorder=5, clip_on=False)
    # a window is an aggregate over 39,990 steps: draw that span once, to scale
    ax.plot([4, 4 + SPAN / 1000], [37.0, 37.0], "-", color=GREY, lw=2.0,
            solid_capstyle="butt", zorder=4)
    ax.text(4 + SPAN / 2000, 39.0, "one window", color=GREY, fontsize=5.8,
            ha="center", va="bottom")
    ax.text(8.0, 2.05, r"$t_{\mathrm{gen}}$", color=GREY, fontsize=6.2,
            ha="left", va="center")
    ax.set_yscale("log")
    ax.set_xlim(-2, 122)
    ax.set_ylim(1.9, 46)
    ax.set_yticks([2, 4, 8, 16, 32])
    ax.set_yticklabels(["2", "4", "8", "16", "32"])
    ax.set_xticks([0, 60, 120])
    ax.set_xticklabels(["0", "60k", "120k"])
    ax.set_xlabel("window centre (steps)")
    ax.set_ylabel("estimate on the norm")
    ax.set_title("(a) the whole record", loc="left", pad=4)

    # ---- (b) aligned on t_gen, in components, against the nuisance floor -----
    bx.axvspan(-5, 5, color=STOCHASTIC, alpha=0.11, lw=0, zorder=0)
    bx.axhspan(-FLOOR, FLOOR, color=GREY, alpha=0.15, lw=0, zorder=0)
    bx.axvline(0, color=GREY, lw=0.9, ls=(0, (2, 2)), zorder=1)
    for r, c, gen, ls, mk in runs:
        if not gen:
            continue
        g = d[d.run == r].sort_values("centre")
        off = (g.centre - tgen[r]).values
        ref = g.MG.values[np.abs(off).argmin()]
        y = g.MG.values - ref
        # every marker carries the span of the window it summarises, so that the
        # reader can see that consecutive windows overlap by three quarters and
        # that the record localises nothing to better than +-20,000 steps
        for xi, yi in zip(off, y):
            bx.plot([(xi - HALF) / 1000, (xi + HALF) / 1000], [yi, yi], "-",
                    color=c, lw=0.7, alpha=0.30, zorder=2)
        bx.plot(off / 1000, y, linestyle=ls, color=c, marker=mk, ms=3.0,
                mec="white", mew=0.4, lw=1.0, zorder=3)
    bx.set_xlim(-108, 108)
    bx.set_ylim(-11.6, 3.4)
    bx.set_xticks([-100, -50, 0, 50, 100])
    bx.set_xticklabels(["-100k", "", "$t_{gen}$", "", "100k"])
    bx.set_yticks([-10, -5, 0])
    bx.set_xlabel("steps since generalisation")
    bx.set_ylabel("change (components)")
    bx.set_title("(b) aligned on $t_{\\mathrm{gen}}$", loc="left", pad=4)
    bx.text(-105, 2.5, "$\\pm 1.16$: the nuisance floor", color=GREY,
            fontsize=5.8, ha="left", va="center")
    # the rest of the reading -- what the bars are, and that the shaded column
    # holds one of the twenty-seven centres -- is in the caption: written into
    # the panel it collides with the two runs that descend through it.
    bx.text(0, -11.2, "$\\pm 5$k: 1 of 27 centres", color=STOCHASTIC,
            fontsize=5.6, ha="center", va="center")

    # ---- (c) every stride-to-stride change, by outcome -----------------------
    cx.axvline(FLOOR, color=GREY, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ROWS = {True: 1.0, False: 0.0}
    seen = {True: 0, False: 0}
    nrun = {True: 3, False: 4}
    for r, c, gen, ls, mk in runs:
        g = d[d.run == r].sort_values("centre")
        v = np.abs(np.diff(g.MG.values))
        # one sub-row per run, so that the eight near-coincident points of a flat
        # run stay countable instead of piling into a single smear
        k = seen[gen]
        seen[gen] += 1
        dy = 0.34 * (2 * k / (nrun[gen] - 1) - 1)
        cx.plot(v, np.full_like(v, ROWS[gen] + dy), marker=mk, color=c, ms=2.8,
                mec="white", mew=0.35, ls="none", zorder=3, clip_on=False)
    cx.set_yticks([1.0, 0.0])
    cx.set_yticklabels(["generalises", "does not"])
    cx.set_ylim(-1.30, 1.60)
    cx.set_xlim(-0.2, 7.0)
    cx.set_xticks([0, 2, 4, 6])
    cx.set_xlabel("$|\\Delta|$ per stride")
    cx.set_title("(c) specificity", loc="left", pad=4)
    cx.spines["left"].set_visible(False)
    cx.tick_params(axis="y", length=0)
    cx.text(1.35, 1.58, "floor", color=GREY, fontsize=5.8, ha="left", va="top")
    # the 6.36 is grokpos_s0, whose sub-row is the lowest of the three
    cx.annotate("$t_{\\mathrm{gen}}\\!+\\!95$k", xy=(6.36, 0.66),
                xytext=(6.95, 1.30), fontsize=5.4, color=STOCHASTIC,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=STOCHASTIC, lw=0.7,
                                shrinkA=1, shrinkB=2))
    cx.annotate("largest of all, in a run\nthat never generalises",
                xy=(5.20, 0.34), xytext=(-0.15, -0.86), fontsize=5.4,
                color=TRANSIENT, va="center", ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=TRANSIENT, lw=0.8,
                                shrinkA=3, shrinkB=2))

    handles = [
        Line2D([], [], color=STOCHASTIC, ls="-", marker="o", ms=3.0,
               mec="white", mew=0.4, label="stochastic, generalises (3)"),
        Line2D([], [], color=STOCHASTIC, ls="--", marker="v", ms=3.0,
               mec="white", mew=0.4, label="stochastic, does not (2)"),
        Line2D([], [], color=TRANSIENT, ls=(0, (1, 1.6)), marker="s", ms=3.0,
               mec="white", mew=0.4, label="transient, does not (2)"),
    ]
    fig.tight_layout(rect=[0, 0.15, 1, 1], w_pad=0.9)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=1.1, handletextpad=0.4)
    save(fig, "fig_window")


if __name__ == "__main__":
    fig_regimes()
    fig_dip()
    fig_map()
    fig_pairs()
    fig_window()

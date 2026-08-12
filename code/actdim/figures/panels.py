"""The article's twelve figures.

The drawing is the archived generator's and is meant to stay that way: a figure rebuilt
here should be indistinguishable from the committed one wherever its data has not
changed. What moved is the input side. The archived generator resolved every file
against one constant pointing into a tree of per-cluster ``results/`` directories; here a
figure asks :mod:`actdim.figures.sources` for a logical name and never sees a path.
Nothing in this module chooses where its output goes either: :func:`build` draws into the
directory it is handed, and the experiment module decides what that is.

Design notes, so that later edits do not undo them:

* Every figure is drawn at 5.5 in, the exact \\textwidth of the ICOMP style, with
  8 pt type, so LaTeX never rescales it and the type stays at 8 pt on the page.
  Figures in the appendices may be up to 5.5 in wide and about 2.2 in tall.
* Colour encodes the paper's three dynamical regimes and nothing else:
  recurrent, stochastic, transient. Line style distinguishes variants inside a
  regime and marker shape distinguishes experimental settings, so identity is
  never carried by colour alone. When every run in a figure belongs to one
  regime, that figure is monochrome by construction: fig_pairs and fig_prwindow
  are all-transient, fig_aniso, fig_tau and fig_observers all-recurrent. Do not
  reintroduce a second hue there to separate two groups; use fill and dash.
* The palette is Paul Tol's high-contrast qualitative scheme:

      recurrent   #004488   deep blue
      stochastic  #BB5566   brick rose
      transient   #997700   dark gold
      neutral     #666666   grey, for reference lines and pointers

  Measured: worst-pair separation 45.3 dE under simulated protanopia and
  50.7 dE under simulated deuteranopia, minimum WCAG contrast 4.21 against
  white. This replaces an earlier Okabe-Ito-style blue/orange/purple set whose
  worst pair separated by only 8.8 dE under deuteranopia.
* Inside the axes: axis labels, tick labels, a legend, and at most one very
  short pointer per panel where a mark would otherwise be unreadable. Every
  explanatory sentence lives in the LaTeX caption. If a panel needs a sentence
  to be understood, move the sentence, not the ink.
* The style is deliberately light: 0.5 pt spines, short 0.5 pt ticks, no black,
  markers at 2.5-4 pt, no gridlines except the row guides of fig_observers,
  which is a dot plot and needs them.
* Where a point is an aggregate the spread is drawn with it: a low-alpha
  interquartile fill in the regime colour in fig_regimes(a), range and
  interquartile bars in fig_pairs(a), the two-hash-family disagreement as a
  band on the mean of fig_dip(a) and (b), the seed spread as a bar in
  fig_aniso, fig_tau and fig_observers, and in fig_window(b) the 39,990-step
  window each point summarises, drawn as a horizontal bar. Do not remove them
  for tidiness. The fig_dip band is about 1 % of the value and is therefore
  nearly invisible: that is the finding, and the legend names it so that the
  reader knows it was drawn rather than omitted.
* Where points coincide, the multiplicity is made visible rather than hidden:
  fig_regimes(b) plots every raw value on a per-observer row, fig_map offsets
  the twelve runs that share (2 crossings, rho_ident = 1.00) and labels the
  multiplicity, fig_window(c) gives each run its own sub-row so that the eight
  strides of a flat run stay countable. Any figure edit that reintroduces a
  single mark for many runs must also fix the caption.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .sources import Reader
from .style import (BAND, FAINT, GREY, POINTER, RECURRENT, STOCHASTIC, TRANSIENT,
                    context, save)


# ------------------------------------------------------------------ figure 1
def fig_regimes(read: Reader):
    """Recovery and admissibility on the image-data system of section 5.4."""
    d = read.table("sweep_raw")
    d = d[(~d.eta_zero) & (~d.observer.isin(["acc_probe", "loss_step"]))]

    series = [
        ("qp", "recurrent, fast", RECURRENT, "-", "o"),
        ("qp_slow", "recurrent, slow", RECURRENT, "--", "s"),
        ("mixed", "recurrent + noise", STOCHASTIC, "--", "^"),
        ("noise", "stochastic", STOCHASTIC, "-", "v"),
        ("gd", "transient", TRANSIENT, ":", "D"),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.50),
                                 gridspec_kw={"width_ratios": [1.18, 1.0]})

    # two reference lines, each named by a six-character label at its own end:
    # E_max is not a bound (no clamping is applied) and the diagonal is the truth
    ax.axhline(20, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    ax.text(9.6, 20, r"$E_{\max}$", va="center", ha="left", **POINTER)
    ax.plot([0.9, 8.6], [0.9, 8.6], color=FAINT, lw=0.9, ls=(0, (3.5, 2.5)),
            zorder=1)
    ax.text(9.6, 8.6, "truth", va="center", ha="left", **POINTER)

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
        ax.fill_between(t[o], q1[o], q3[o], color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(t, y, ls, color=c, marker=mk, ms=2.8, mec="white", mew=0.5,
                lw=1.1, label=label, zorder=4, clip_on=False)
        # rho_ident is populated for seed 0, three observers and r in {2, 6} only,
        # so panel (b) plots the six raw values rather than an aggregate.
        ident[arm] = g.dropna(subset=["ident_ratio"])[["observer", "r",
                                                       "ident_ratio"]]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 9.2)
    ax.set_ylim(0.6, 42)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel("measured effective rank")
    ax.set_ylabel("estimated dimension")
    ax.set_title("(a) what the estimator returns", loc="left")

    bx.axvspan(0.95, 1.10, color=BAND, alpha=0.08, lw=0, zorder=0)
    # one row of markers per observer, so that coincident values stay countable
    OBS_DY = {"c_proj1": 0.17, "g_fro": 0.0, "w_fro": -0.17}
    for i, (arm, label, c, ls, mk) in enumerate(series):
        v = ident.get(arm)
        if v is None or v.empty:
            continue
        y = len(series) - 1 - i
        bx.plot([v.ident_ratio.min(), v.ident_ratio.max()], [y, y], "-",
                color=c, lw=0.6, alpha=0.40, zorder=2)
        bx.plot(v.ident_ratio.values, y + v.observer.map(OBS_DY).values, mk,
                color=c, ms=3.0, mec="white", mew=0.5, zorder=3, clip_on=False)
    bx.set_yticks(range(len(series)))
    bx.set_yticklabels([lbl for _, lbl, _, _, _ in series][::-1])
    bx.set_ylim(-0.65, len(series) - 0.20)
    bx.set_xlim(0.92, 1.66)
    bx.set_xticks([1.0, 1.2, 1.4, 1.6])
    bx.set_xlabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    bx.set_title("(b) admissibility", loc="left")
    bx.text(1.025, 4.30, "admissible", color=BAND, fontsize=6.2, ha="center",
            va="bottom")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=2.8, mec="white",
                      mew=0.5, label=lbl) for _, lbl, c, ls, mk in series]
    fig.tight_layout(rect=[0, 0.095, 1, 1], w_pad=1.4)
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.1)
    return fig


# ------------------------------------------------------------------ figure 2
def fig_dip(read: Reader):
    """The trajectory-rank collapse at generalisation, aligned on t_gen.

    Every run here is a mini-batch transformer. The four that generalise carry
    weight decay and are stochastic; the two controls have none and are
    transient, exactly as fig_map() classifies the same two configurations. No
    run is recurrent, so RECURRENT does not appear.
    """
    d = read.table("rank_windows")
    meta = {m["run"]: m for m in read.object("rank_milestones")}

    # panel (d) is a different statistic on the same windows: the delay estimate on the
    # parameter-norm log, at the window matched to the transition (grok.matched.window).
    # It is joined on mid_step, which that experiment takes from this very file, so the
    # two curves are sampled at identical instants and the alignment below is the same
    # alignment.
    e9 = read.table("headline_trace")
    e9 = e9[e9.column == "weight_norm"].rename(columns={"mid_step": "mid"})
    d["mid"] = 0.5 * (d.right_step + d.left_step)
    d = d.merge(e9[["run", "mid", "MG"]], on=["run", "mid"], how="left")
    assert d.MG.notna().sum() > 1000, "matched-window trace did not join onto the direct grid"

    groks = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    ctrls = ["mod_wd0", "s5_wd0"]
    panels = [("fn_PR_pos_det", "(a) function space", None),
              ("PR_pos_det", "(b) parameter space", None),
              ("move", "(c) displacement", "log"),
              ("MG", "(d) scalar log", None)]

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.40), sharex=True)
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
            ax.plot(x, g[col], color=STOCHASTIC, lw=0.6, alpha=0.28)
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
            ax.fill_between(grid, m - s, m + s, facecolor=STOCHASTIC, alpha=0.55,
                            edgecolor=STOCHASTIC, lw=0.4, zorder=4)
        ax.plot(grid, m, color=STOCHASTIC, lw=1.5, zorder=5)

        for r, ls in zip(ctrls, ["--", (0, (1, 1.6))]):
            g = d[d.run == r].sort_values("right_step")
            ref = meta["mod_wd1" if r.startswith("mod") else "s5_wd1"]["t_gen"]
            ax.plot(0.5 * (g.right_step + g.left_step) - ref, g[col],
                    linestyle=ls, color=TRANSIENT, lw=1.0)

        ax.axvline(0, color=FAINT, lw=0.7, ls=(0, (2, 2.5)), zorder=0)
        if scale:
            ax.set_yscale(scale)
        ax.set_xlim(-5000, 5000)
        ax.set_xticks([-5000, 0, 5000])
        ax.set_xticklabels(["$-5$k", "$t_{\\mathrm{gen}}$", "5k"])
        ax.set_title(title, loc="left")

    axes[0].set_ylabel("participation ratio")
    axes[2].set_ylabel("displacement")
    axes[3].set_ylabel("components")

    handles = [
        Line2D([], [], color=STOCHASTIC, lw=1.5, label="generalises (4)"),
        Line2D([], [], color=TRANSIENT, ls="--", lw=1.0,
               label="no weight decay (2)"),
        mpl.patches.Patch(facecolor=STOCHASTIC, alpha=0.55, lw=0,
                          label="hash-family spread"),
    ]
    fig.tight_layout(rect=[0, 0.150, 1, 1], w_pad=0.9)
    # one x label under four panels: placed against the measured axes box, because
    # supxlabel(y=...) and tight_layout(rect=...) do not know about each other and
    # leave a band of white between the ticks and the label
    fig.text(0.5, min(a.get_position().y0 for a in axes) - 0.105,
             "steps since generalisation", ha="center", va="top")
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.4)
    return fig


# ------------------------------------------------------------------ figure 3
def fig_map(read: Reader):
    """Where the training logs of section 7 fall in the admissibility plane."""
    tr = read.table("real_logs_summary")
    tr = tr[tr.column == "weight_norm"]
    gr = pd.concat([read.table("probe_arith_summary"),
                    read.table("probe_poly_summary")])
    gr = gr[gr.column == "train_loss"]

    fig, ax = plt.subplots(figsize=(5.5, 2.05))

    # one highlighted target zone: stable in E and recurrent enough to embed
    ax.add_patch(mpl.patches.Rectangle(
        (8.0, 0.95), 500 - 8.0, 0.15, facecolor=BAND, alpha=0.08,
        edgecolor=BAND, lw=0.7, ls=(0, (3.5, 3)), zorder=0))
    ax.axvline(8.0, color=FAINT, lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.axhline(1.10, color=FAINT, lw=0.7, ls=(0, (2, 3)), zorder=0)

    hi = tr[tr.ident > 1.15]
    lo = tr[tr.ident <= 1.15]

    # All ten perceptron runs and both zero-weight-decay transformer runs sit at
    # exactly two crossings and rho_ident = 1.00 to three decimals, so without an
    # offset the twelve of them render as one mark. The offset is cosmetic and is
    # declared in the caption; the true abscissa of every point is the grey tick.
    OFF = 1.20
    ax.plot([2 / OFF, 2 * OFF], [0.999, 0.999], "-", color=FAINT, lw=0.6,
            zorder=2)
    ax.plot([2, 2], [0.981, 1.017], "-", color=GREY, lw=0.6, zorder=2)
    ax.plot(gr.osc / OFF, gr.ident, "s", mfc="none", mec=TRANSIENT, mew=1.0,
            ms=6.5, ls="none", label="perceptron, full batch (10)", zorder=3)
    ax.plot(hi.osc, hi.ident, "o", color=STOCHASTIC, ms=4.6, mec="white",
            mew=0.7, ls="none", label="transformer, mini-batch (5)", zorder=4)
    ax.plot(lo.osc * OFF, lo.ident, "D", color=TRANSIENT, ms=4.0, mec="white",
            mew=0.7, ls="none", label="transformer, no weight decay (2)",
            zorder=5)
    ax.text(2 / OFF / 1.16, 0.999, r"$\times 10$", color=TRANSIENT,
            fontsize=6.2, ha="right", va="center")
    ax.text(2 * OFF * 1.13, 0.999, r"$\times 2$", color=TRANSIENT,
            fontsize=6.2, ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.7, 500)
    ax.set_ylim(0.90, 1.68)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("trend crossings per window")
    ax.set_ylabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    ax.text(430, 1.025, "admissible", color=BAND, fontsize=6.5, ha="right",
            va="center")

    fig.tight_layout(rect=[0, 0.085, 1, 1])
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.3, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ figure 4
def fig_pairs(read: Reader):
    """The label-matched pairs, and the training loss that explains them.

    All eight runs are full-batch perceptrons, which fig_map() classifies as
    transient, so the figure is monochrome by the palette rule: generalisation
    is carried by marker fill and line style, never by hue.
    """
    mg = pd.concat([read.table("probe_arith_summary"),
                    read.table("probe_poly_summary")])
    mg = mg[mg.column == "train_loss"].set_index("run").MG
    # the seven sliding windows behind each of those run-level medians
    win = pd.concat([read.table("probe_arith"), read.table("probe_poly")])
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
    ax.set_xlim(17.7, 27.7)
    ax.set_xticks([18, 20, 22, 24, 26])
    ax.set_xlabel("estimate on the training loss")
    ax.set_title("(a) four label-matched pairs", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # panel (b) plots the g_p2 pair, the second row of panel (a): its two members
    # end three orders of magnitude apart, which the top pair does not (2.7)
    for key, ls in [("g_p2_p97", "-"), ("g_p2x_p97", (0, (3.5, 2)))]:
        t = read.run_table("poly_train", key)
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


# ------------------------------------------------------------------ figure 5
def fig_window(read: Reader):
    """The windowed log estimate against t_gen: the change test of section 7.1.

    Colour is the regime as everywhere else. None of these seven runs is
    recurrent, so RECURRENT never appears here: the five regularised runs are
    stochastic and the two weight-decay-zero runs transient, exactly as
    fig_map() classifies them. Generalisation is carried by line style and
    marker, never by hue.
    """
    d = read.table("real_logs_windows")
    d = d[d.column == "weight_norm"].copy()
    out = read.table("exp8_outcomes")
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
                                     gridspec_kw={"width_ratios": [1.0, 1.06, 0.84]})

    # ---- (a) the raw windowed estimate over training, every run --------------
    for r, c, gen, ls, mk in runs:
        g = d[d.run == r].sort_values("centre")
        ax.plot(g.centre / 1000, g.MG, linestyle=ls, color=c, marker=mk,
                ms=2.2, mec="white", mew=0.3, lw=0.9, zorder=3)
    # the three generalisation steps, marked on the axis rather than by hue
    for r, c, gen, ls, mk in runs:
        if gen:
            ax.plot([tgen[r] / 1000], [2.05], marker="^", color=GREY, ms=3.0,
                    mew=0, zorder=5, clip_on=False)
    ax.text(9.0, 2.05, r"$t_{\mathrm{gen}}$", ha="left", va="center", **POINTER)
    ax.set_yscale("log")
    ax.set_xlim(-2, 124)
    ax.set_ylim(1.9, 44)
    ax.set_yticks([2, 4, 8, 16, 32])
    ax.set_yticklabels(["2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xticks([0, 60, 120])
    ax.set_xticklabels(["0", "60k", "120k"])
    ax.set_xlabel("window centre (steps)")
    ax.set_ylabel("estimate on the norm")
    ax.set_title("(a) the whole record", loc="left")

    # ---- (b) aligned on t_gen, in components, against the nuisance floor -----
    bx.axvspan(-5, 5, color=STOCHASTIC, alpha=0.10, lw=0, zorder=0)
    bx.axhspan(-FLOOR, FLOOR, color=GREY, alpha=0.12, lw=0, zorder=0)
    bx.axvline(0, color=FAINT, lw=0.7, ls=(0, (2, 2.5)), zorder=1)
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
                    color=c, lw=0.6, alpha=0.28, zorder=2)
        bx.plot(off / 1000, y, linestyle=ls, color=c, marker=mk, ms=2.6,
                mec="white", mew=0.35, lw=0.9, zorder=3)
    bx.set_xlim(-108, 108)
    bx.set_ylim(-11.6, 3.4)
    bx.set_xticks([-100, 0, 100])
    bx.set_xticklabels(["$-100$k", "$t_{\\mathrm{gen}}$", "100k"])
    bx.set_yticks([-10, -5, 0])
    bx.set_xlabel("steps since generalisation")
    bx.set_ylabel("change (components)")
    bx.set_title("(b) aligned on $t_{\\mathrm{gen}}$", loc="left")
    bx.text(105, 1.45, "floor", ha="right", va="bottom", **POINTER)

    # ---- (c) every stride-to-stride change, by outcome -----------------------
    cx.axvline(FLOOR, color=FAINT, lw=0.7, ls=(0, (2, 2.5)), zorder=1)
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
        dy = 0.30 * (2 * k / (nrun[gen] - 1) - 1)
        cx.plot(v, np.full_like(v, ROWS[gen] + dy), marker=mk, color=c, ms=2.4,
                mec="white", mew=0.3, ls="none", zorder=3, clip_on=False)
    cx.set_yticks([1.0, 0.0])
    cx.set_yticklabels(["generalises", "does not"])
    cx.set_ylim(-0.62, 1.62)
    cx.set_xlim(-0.2, 7.0)
    cx.set_xticks([0, 2, 4, 6])
    cx.set_xlabel("$|\\Delta|$ per stride")
    cx.set_title("(c) specificity", loc="left")
    cx.spines["left"].set_visible(False)
    cx.tick_params(axis="y", length=0)
    cx.text(1.45, 1.52, "floor", ha="left", va="top", **POINTER)

    handles = [
        Line2D([], [], color=STOCHASTIC, ls="-", marker="o", ms=2.6,
               mec="white", mew=0.35, label="stochastic, generalises (3)"),
        Line2D([], [], color=STOCHASTIC, ls="--", marker="v", ms=2.6,
               mec="white", mew=0.35, label="stochastic, does not (2)"),
        Line2D([], [], color=TRANSIENT, ls=(0, (1, 1.6)), marker="s", ms=2.6,
               mec="white", mew=0.35, label="transient, does not (2)"),
    ]
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=1.5)
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.3)
    return fig


# ------------------------------------------------------------------ figure 6
def fig_aniso(read: Reader):
    """Anisotropy: the estimate follows the count, not the participation ratio.

    One panel per r. The CSV column is `rho`, but the paper renamed the
    amplitude decay factor to q, rho being the Spearman correlation everywhere
    else, so the axis says q. Everything here is one r-torus, i.e. the
    recurrent regime, so the figure is monochrome: the estimate is in the
    regime colour and every reference quantity is grey.
    """
    d = read.table("aniso_summary")
    rs = sorted(d.r.unique())

    fig, axes = plt.subplots(1, len(rs), figsize=(5.5, 2.2), sharey=True)
    for ax, r in zip(axes, rs):
        g = d[d.r == r].sort_values("rho")
        # the closed form (sum a^2)^2 / sum a^4 and the measured covariance PR
        # agree to two decimals: draw the prediction as a line and the
        # measurement as open markers on top of it, so the reader sees both
        ax.plot(g.rho, g.pr_pred, "-", color=GREY, lw=0.9, alpha=0.75, zorder=2)
        ax.plot(g.rho, g.pr_pos, "s", mfc="white", mec=GREY, mew=0.9, ms=3.2,
                ls="none", zorder=3)
        ax.axhline(r, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
        # MG is the median of three seeds and MG_sd their standard deviation
        ax.errorbar(g.rho, g.MG, yerr=g.MG_sd, color=RECURRENT, lw=1.2,
                    marker="o", ms=3.0, mec="white", mew=0.5, elinewidth=0.8,
                    capsize=1.6, capthick=0.8, zorder=4)
        ax.set_title(f"$r = {r}$", loc="left")
        ax.set_xlim(0.44, 1.06)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel("$q$")
    axes[0].set_ylim(0, 9.6)
    axes[0].set_yticks([0, 2, 4, 6, 8])
    axes[0].set_ylabel("components")

    handles = [
        Line2D([], [], color=RECURRENT, ls="-", marker="o", ms=3.0,
               mec="white", mew=0.5, label="estimate (3 seeds, $\\pm$ s.d.)"),
        Line2D([], [], color=GREY, ls="-", marker="s", mfc="white", mec=GREY,
               mew=0.9, ms=3.2, label="participation ratio"),
        Line2D([], [], color=FAINT, ls=(0, (1, 2.5)), lw=0.7,
               label="active dimension"),
    ]
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=0.9)
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.4)
    return fig


# ------------------------------------------------------------------ figure 7
def fig_tau(read: Reader):
    """The delay-lag sweep: the usable span is about four tenths of a period.

    One torus of period 400 at E_max = 20, the configuration the paper's table
    reports. All of it is the recurrent regime, so the figure is monochrome and
    the six ranks are separated by marker shape.
    """
    d = read.table("tau_sensitivity")
    d = d[(d.period == 400) & (d.max_E == 20)]
    # tau='acorr' picks the lag from the autocorrelation, so its span differs
    # between the two seeds and it is not a point on a common x grid; the six
    # fixed lags 1..32 are. Reported in the caption.
    d = d[d.tau != "acorr"].copy()
    d["span_periods"] = d.span_periods.astype(float)

    rs = sorted(d.r.unique())
    marks = ["o", "s", "^", "v", "D", "X"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.2))

    for r, mk in zip(rs, marks):
        g = d[d.r == r]
        a = g.groupby("span_periods").MG.agg(["median", "min", "max"]).sort_index()
        x = a.index.values
        for panel, scale in ((ax, 1.0), (bx, float(r))):
            panel.vlines(x, a["min"] / scale, a["max"] / scale, color=RECURRENT,
                         lw=0.7, alpha=0.55, zorder=2)
            panel.plot(x, a["median"] / scale, "-", color=RECURRENT, lw=1.0,
                       marker=mk, ms=3.0, mec="white", mew=0.5, zorder=3,
                       label=f"$r = {r}$" if panel is ax else None)

    for panel in (ax, bx):
        panel.axvspan(0.30, 0.50, color=GREY, alpha=0.10, lw=0, zorder=0)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(0.038, 1.95)
        panel.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        panel.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
        panel.set_xticks([], minor=True)
        panel.set_xlabel("delay span (periods)")

    ax.set_ylim(1.15, 90)
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_yticks([], minor=True)
    ax.set_ylabel("estimated dimension")
    ax.set_title("(a) the estimate", loc="left")

    bx.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
    bx.set_ylim(0.26, 11)
    bx.set_yticks([0.3, 0.5, 1, 2, 5, 10])
    bx.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    bx.set_yticks([], minor=True)
    bx.set_ylabel("estimate / true rank")
    bx.set_title("(b) relative to the truth", loc="left")
    bx.text(0.40, 9.6, "usable", ha="center", va="top", **POINTER)

    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=1.8)
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.8, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ figure 8
def fig_observers(read: Reader):
    """Which scalar log should you keep? Per-observer error on the recurrent arm.

    One arm (qp), one regime, so the figure is monochrome and the five observer
    families are separated by marker shape.
    """
    d = read.table("observer_scores")
    d = d[d.arm == "qp"]
    # acc_probe is degenerate in every window and has no score; loss_step fails
    # the zero-learning-rate control and is excluded from every aggregate in the
    # paper. Both exclusions are named in the caption.
    d = d[~d.observer.isin(["acc_probe", "loss_step"])].sort_values("mae_raw")

    NAMES = {"loss_probe": "probe loss", "loss_full": "full-batch loss",
             "w_fro": "parameter norm", "c_norm": "subspace norm",
             "fn_fro": "function-space norm", "g_fro": "gradient norm",
             "g_proj": "gradient projection", "c_proj1": "parameter projection",
             "fn_proj1": "function-space projection", "margin": "margin"}
    MARKS = {"loss": "o", "norm": "s", "gradient": "^", "projection": "D",
             "function": "v"}
    LOGGED = {"w_fro", "c_norm"}  # the two parameter norms

    fig, ax = plt.subplots(figsize=(5.5, 2.35))

    y = np.arange(len(d))[::-1]
    # the two parameter norms are adjacent in the ordering, so one band marks them
    band = [yy for yy, o in zip(y, d.observer) if o in LOGGED]
    ax.axhspan(min(band) - 0.5, max(band) + 0.5, color=GREY, alpha=0.09, lw=0,
               zorder=0)
    ax.axvline(0.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
    for yy in y:
        ax.plot([-0.35, 2.75], [yy, yy], "-", color="#EEEEEE", lw=0.5, zorder=0)

    for yy, (_, row) in zip(y, d.iterrows()):
        # bar: the mean across-seed standard deviation of the estimate, drawn at
        # the point as +- that half-width. It is a spread of the estimate, not of
        # the error, so it may reach below zero; the caption says so.
        ax.plot([row.mae_raw - row.seed_sd, row.mae_raw + row.seed_sd],
                [yy, yy], "-", color=RECURRENT, lw=0.8, alpha=0.45, zorder=2)
        ax.plot([row.mae_raw], [yy], MARKS[row.family], color=RECURRENT,
                ms=3.6, mec="white", mew=0.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(o, o) for o in d.observer])
    for tick, o in zip(ax.get_yticklabels(), d.observer):
        if o in LOGGED:
            tick.set_fontweight("bold")
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(-0.35, 2.75)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("mean absolute error (components)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=RECURRENT, ls="none", marker=m, ms=3.6,
                      mec="white", mew=0.6, label=f) for f, m in MARKS.items()]
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=1.2,
               columnspacing=1.6)
    return fig


# ------------------------------------------------------------------ figure 9
def fig_prwindow(read: Reader):
    """The full-batch participation ratio against window length.

    Both runs are full-batch perceptrons, which fig_map() classifies as
    transient, so the figure is monochrome: the generalising run is filled and
    solid, its label-matched control open and dashed.
    """
    d = read.table("pr_vs_window")
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


# ----------------------------------------------------------------- figure 10
def fig_eos(read: Reader):
    """Full-batch descent at the edge of stability, and what the logging stride hides."""
    runs = read.table("eos_runs").sort_values(["lr", "seed"])

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.78),
                             gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]})
    ax, bx, cx = axes

    # (a) the stability ratio along training, one line per rate, seed 1 only so the
    # panel stays readable; the seeds agree to within the linewidth (see eos_runs.csv).
    ax.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    ax.text(30500, 1.0, "$2/\\eta$", va="center", ha="left", **POINTER)
    for _, m in runs[runs.seed == 1].iterrows():
        sh = read.run_table("eos_sharp", m["key"])
        # Colour is the regime the run turns out to be in, as everywhere else: the
        # sub-threshold rates descend monotonically and are transient; the rates that
        # pin at 2/eta are the candidate recurrent regime this appendix is about.
        eos = pd.isna(m["diverged_at"]) and m["eta_lam_over_2_median_tail"] > 0.9
        c = RECURRENT if eos else (GREY if pd.notna(m["diverged_at"]) else TRANSIENT)
        ls = "-" if eos else ((0, (1, 2)) if pd.notna(m["diverged_at"]) else (0, (4, 2)))
        ax.plot(sh["step"], sh["eta_lam_over_2"], ls=ls, color=c, lw=1.0,
                clip_on=True, zorder=3 if eos else 2)
    ax.set_xlim(0, 30000)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("optimiser step")
    ax.set_ylabel(r"$\eta\lambda_{\max}/2$")
    ax.set_xticks([0, 15000, 30000])
    ax.set_xticklabels(["0", "15k", "30k"])

    # (b) sixty consecutive steps deep in the edge-of-stability phase, against the same
    # series read at the stride every full-batch log in this repository uses.
    #
    # Detrended, and that is not cosmetic. Over sixty steps the descent is much larger
    # than the two-cycle riding on it, so the raw loss is a straight line to the eye at
    # every window and the panel would show nothing -- an earlier draft of this figure
    # did exactly that. Removing the linear fit over the window shown puts the two
    # series on the scale of the thing being compared. The window is a typical one, not
    # a chosen one: its rise fraction, 0.492, is the run's own median.
    tr = read.run_table("eos_train", "eos_lr2e+06_s1")
    seg = tr.iloc[20000:20060]
    step = seg["step"].to_numpy(float)
    loss = seg["train_loss"].to_numpy(float)
    resid = (loss - np.polyval(np.polyfit(np.arange(60.0), loss, 1),
                               np.arange(60.0))) * 1e9
    bx.axhline(0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    bx.plot(step, resid, "-", color=RECURRENT, lw=0.9, marker="o", ms=1.8,
            mec="none", label="every step", zorder=3)
    bx.plot(step[::10], resid[::10], ls=(0, (4, 2)), color=TRANSIENT, lw=1.0,
            marker="s", ms=3.0, mec="white", mew=0.5, label="stride 10", zorder=4)
    bx.set_xlabel("optimiser step")
    bx.set_ylabel("detrended loss ($10^{-9}$)")
    bx.set_xticks([20000, 20030, 20060])
    bx.set_xticklabels(["20k", "+30", "+60"])
    # headroom for the legend: the two-cycle fills the axes, and the stride-10 series
    # rides its upper branch, so an unexpanded top puts the legend on top of the data.
    bx.set_ylim(-3.0, 4.6)
    bx.set_yticks([-2, 0, 2])
    bx.legend(loc="upper right", handlelength=1.8)

    # (c) what the two non-monotonicity statistics do as the log is decimated.
    diag = read.table("eos_diagnostics")
    g = diag[(diag.column == "train_loss") & (diag.segment == "post") & (diag.tau == 1)]
    g = g.groupby(["lr", "subsample"]).agg(
        rises=("rises", "median"), crossings=("crossings", "median")).reset_index()
    eos_lrs = sorted(runs[(runs.eta_lam_over_2_median_tail > 0.9)
                          & (runs.diverged_at.isna())].lr.unique())
    for lr in eos_lrs:
        h = g[g.lr == lr].sort_values("subsample")
        if h.empty:
            continue
        cx.plot(h["subsample"], h["rises"], "-", color=RECURRENT, lw=0.9,
                marker="o", ms=2.6, mec="white", mew=0.4, alpha=0.85)
    sub = sorted(g["subsample"].unique())
    cx.axhline(0.0, color=FAINT, lw=0.7, ls=(0, (1, 2.5)), zorder=0)
    cx.set_xscale("log")
    cx.set_xticks(sub)
    cx.set_xticklabels([str(s) for s in sub])
    cx.minorticks_off()
    cx.set_ylim(-0.04, 0.56)
    cx.set_xlabel("logging stride")
    cx.set_ylabel("fraction of steps rising")
    cx.text(sub[-1], 0.02, "monotone", ha="right", va="bottom", **POINTER)

    fig.tight_layout(w_pad=1.5)
    return fig


# ----------------------------------------------------------------- figure 11
def fig_ceiling(read: Reader):
    """The tracking ceiling against each hypothesis's knob, and against its prediction."""
    d = read.table("ceiling_summary")
    d = d[d.arm == "frozen"]
    e = d[d.sweep == "E"].sort_values("max_E")
    n = d[d.sweep == "N"].sort_values("N")
    n56 = d[d.sweep == "N_E56"].sort_values("N")

    # Every arm here is the constructed recurrent system, so the figure is monochrome by
    # the rule in the module docstring; the two predictions are reference lines.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.95))

    ax.plot(e.max_E, e.takens, ls=(0, (3.5, 2.5)), color=FAINT, lw=0.9, zorder=1)
    ax.axhline(e.eckmann_ruelle.iloc[0], color=FAINT, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
    ax.plot(e.max_E, e["r_track_1"], "-", color=RECURRENT, lw=1.1, marker="o", ms=3.0,
            mec="white", mew=0.5, zorder=4, clip_on=False)
    ax.set_xscale("log")
    ax.set_xticks(e.max_E)
    ax.set_xticklabels([str(int(v)) for v in e.max_E])
    ax.minorticks_off()
    ax.set_ylim(0, 29)
    ax.set_xlabel("$E_{\\max}$   (record fixed at 8000)")
    ax.set_ylabel("rank tracked")
    ax.text(56, 27.5, "$E_{\\max}/2$", ha="right", va="top", **POINTER)
    ax.text(10.4, 8.6, "$2\\log_{10}N$", ha="left", va="bottom", **POINTER)

    bx.plot(n.N, n.eckmann_ruelle, ls=(0, (1, 2.5)), color=FAINT, lw=0.9, zorder=1)
    bx.axhline(10.0, color=FAINT, lw=0.9, ls=(0, (3.5, 2.5)), zorder=1)
    bx.plot(n.N, n["r_track_1"], "-", color=RECURRENT, lw=1.1, marker="o", ms=3.0,
            mec="white", mew=0.5, label="$E_{\\max}=20$", zorder=4)
    bx.plot(n56.N, n56["r_track_1"], ls=(0, (4, 2)), color=RECURRENT, lw=1.1, marker="s",
            ms=3.2, mfc="white", mec=RECURRENT, mew=0.9, label="$E_{\\max}=56$", zorder=5)
    bx.set_xscale("log")
    bx.set_xticks([1e3, 1e4, 1e5])
    bx.minorticks_off()
    bx.set_ylim(0, 14)
    bx.set_xlabel("record length   ($E_{\\max}$ fixed)")
    bx.set_ylabel("rank tracked")
    # The dotted line is the hypothesis this panel refutes, so it is the one labelled;
    # the dashed line is Takens for the E_max = 20 arm only and would be 28 for the
    # other, which is off scale, so the caption names it rather than a pointer here.
    bx.text(1.05e3, 5.7, "$2\\log_{10}N$", ha="left", va="top", **POINTER)
    bx.legend(loc="lower right", handlelength=2.0)

    fig.tight_layout(w_pad=1.8)
    return fig


# ----------------------------------------------------------------- figure 12
def fig_traces(read: Reader):
    """One raw scalar log per regime -- what the estimator is actually reading."""
    syn = read.table("example_traces")
    real = read.table("mod_wd1_train")

    def z(a):
        a = np.asarray(a, float)
        return (a - a.mean()) / a.std()

    # Each panel spans what that regime needs to be legible: the recurrent arm's drive
    # period is 16 samples, so 400 samples is 25 cycles, while the transient and the
    # real run are shown whole. Standardised, because the estimator standardises.
    panels = [
        ("recurrent", z(syn["recurrent"].to_numpy()[:400]), np.arange(400),
         RECURRENT, "deterministic, recurrent", "sample"),
        ("transient", z(syn["transient"].to_numpy()), np.arange(len(syn)),
         TRANSIENT, "deterministic, transient", "sample"),
        ("stochastic", z(real["weight_norm"].to_numpy()),
         real["step"].to_numpy() / 1000.0, STOCHASTIC, "stochastically driven",
         "step ($10^3$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.55))
    for ax, (_, y, x, c, title, xlab) in zip(axes, panels):
        ax.plot(x, y, "-", color=c, lw=0.6, zorder=3)
        ax.set_title(title, pad=3.0)
        ax.set_xlabel(xlab)
        ax.set_ylim(-3.4, 3.4)
        ax.set_yticks([-2, 0, 2])
    axes[0].set_ylabel("standardised")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.tight_layout(w_pad=1.2)
    return fig


# ------------------------------------------------------------------ the build
#
# In the order the archived generator drew them, which is the order they appear in the
# article.

PANELS: Dict[str, Callable[[Reader], Any]] = {
    "fig_regimes": fig_regimes,
    "fig_dip": fig_dip,
    "fig_map": fig_map,
    "fig_pairs": fig_pairs,
    "fig_window": fig_window,
    "fig_aniso": fig_aniso,
    "fig_tau": fig_tau,
    "fig_observers": fig_observers,
    "fig_prwindow": fig_prwindow,
    "fig_eos": fig_eos,
    "fig_ceiling": fig_ceiling,
    "fig_traces": fig_traces,
}

NAMES = tuple(PANELS)


def draw(name: str, read: Reader):
    """Draw one figure at the article's rcParams and return it, unsaved.

    Separate from :func:`build` so that a test can hold the figure and check its width,
    which is the property the no-``bbox_inches`` rule exists to protect.
    """
    if name not in PANELS:
        raise KeyError(f"no such figure: {name!r}. Known: {', '.join(NAMES)}")
    with context():
        return PANELS[name](read)


def build(outdir: Path, names: Sequence[str] = (), allow_archive: bool = False) -> Dict[str, Any]:
    """Draw the requested figures into ``outdir``, all twelve by default.

    Returns what was drawn and where each input came from, so that the caller can record
    it and can see at once whether anything was built from the archived tree.
    """
    outdir = Path(outdir)
    wanted = list(names) if names else list(NAMES)
    unknown = [n for n in wanted if n not in PANELS]
    if unknown:
        raise KeyError(f"no such figure: {', '.join(unknown)}. Known: {', '.join(NAMES)}")

    record: Dict[str, Any] = {
        "outdir": outdir.as_posix(),
        "allow_archive": bool(allow_archive),
        "figures": {},
        "archived_figures": [],
        "archived_sources": [],
    }

    for name in wanted:
        read = Reader(allow_archive=allow_archive)
        fig = draw(name, read)
        written = save(fig, name, outdir)
        entry = read.record()
        entry["files"] = [p.as_posix() for p in written]
        record["figures"][name] = entry
        if entry["archived"]:
            record["archived_figures"].append(name)
            for source in entry["archived"]:
                if source not in record["archived_sources"]:
                    record["archived_sources"].append(source)

    return record


def summary(record: Dict[str, Any]) -> str:
    """One printable paragraph saying what was drawn, and what it was drawn from."""
    drawn = len(record["figures"])
    lines: List[str] = [f"{drawn} figure(s) written to {record['outdir']}"]
    stale = record["archived_figures"]
    if not stale:
        lines.append("every input came from data/")
        return "\n".join(lines)

    lines.append(f"WARNING: {len(stale)} of them were built from the archived tree, "
                 f"which no experiment in this package produced:")
    for name in stale:
        used = ", ".join(record["figures"][name]["archived"])
        lines.append(f"  {name}: {used}")
    lines.append("These figures are not reproducible from data/ and must not be "
                 "published. Re-run and promote the experiments above, then rebuild "
                 "without allow_archive.")
    return "\n".join(lines)

"""The article's nineteen figures.

The drawing is the archived generator's and is meant to stay that way: a figure rebuilt
here should be indistinguishable from the committed one wherever its data has not
changed. What moved is the input side. The archived generator resolved every file
against one constant pointing into a tree of per-cluster ``results/`` directories; here a
figure asks :mod:`actdim.figures.sources` for a logical name and never sees a path.
Nothing in this module chooses where its output goes either: :func:`build` draws into the
directory it is handed, and the experiment module decides what that is.

Design notes, so that later edits do not undo them:

* Every figure is drawn at 5.5 in, the exact \\textwidth of the ICOMP style, with
  9 pt type, so LaTeX never rescales it and the type stays at 9 pt on the page.
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
  fig_regimes(b) plots every raw value on a per-rank, per-observer row, fig_map offsets
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
from matplotlib.patches import Circle, Patch

from ..estimator.embedding import resolve_theiler
from ..frozen import eight_direction
from .sources import Reader
from .style import (BAND, FAINT, GREY, INK, POINTER, RECURRENT, STOCHASTIC,
                    TRANSIENT, WIDTH, bounds, context, save)


# ----------------------------------------------------------------- figure 1
def fig_method(read: Reader):
    """The three steps of the estimator on one recurrent log of known active dimension."""
    # Read off the estimator rather than written down here, so that a picture of the method
    # cannot drift from the method. Under theiler="embedding" the rule uses the
    # configuration alone, so the empty series it is handed is never looked at.
    cfg = eight_direction()
    tau = int(cfg.tau)
    exclusion = resolve_theiler(cfg, np.empty(0, dtype=float), tau)

    series = read.table("curve_series")
    log = series[(series.r == 1) & (series["sample"] <= 400)].sort_values("sample")

    # The estimator's own reconstruction of the same record, written out by the experiment
    # at the lag every number in the paper was computed at. Rebuilding it here from the
    # decimated series would draw a delay plane of some other lag.
    cloud = read.table("curve_shapes")
    cloud = cloud[(cloud.arm == "qp") & (cloud.r == 1)]
    points = np.column_stack([cloud.x.to_numpy(), cloud.y.to_numpy()])
    # One window's reconstruction, thinned to about 1400 rows in time order at a constant
    # stride, so the exclusion converts into a number of rows.
    thin = max(1, round((cfg.window - (cfg.max_E - 1) * tau) / len(points)))

    centre = int(0.37 * len(points))
    distance = np.linalg.norm(points - points[centre], axis=1)
    excluded = np.abs(np.arange(len(points)) - centre) <= exclusion // thin
    nearest = np.argsort(np.where(excluded, np.inf, distance))[:cfg.k_neighbors]
    radius = float(distance[nearest].max())

    # Placed by hand rather than by tight_layout: two of the three panels must be square,
    # or a closed loop drawn on a stretched axis becomes an ellipse and reads as
    # anisotropy, and tight_layout sets a title against an equal-aspect axes box rather
    # than against the slot, which puts the three titles on three different lines.
    height, foot, head = 1.68, 0.58, 0.18
    y0, h = foot / height, (height - foot - head) / height
    side = (height - foot - head) / WIDTH        # a square, in fractions of the width

    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0.072, y0, 0.312, h])
    bx = fig.add_axes([0.497, y0, side, h])
    cx = fig.add_axes([0.769, y0, side, h])
    for slot, name in ((ax, "(a) the log, first 400 steps"),
                       (bx, "(b) the delay plane"), (cx, "(c) one neighbourhood")):
        box = slot.get_position()
        fig.text(0.5 * (box.x0 + box.x1), (height - head + 0.025) / height, name,
                 ha="center", va="bottom", fontsize=9, color=INK)

    # A zoom, and the title of the panel it is drawn in says the whole record is longer.
    # At rank one the log turns over every few samples, so ten thousand of them across an
    # inch and a half resolve nothing at all.
    ax.plot(log["sample"].to_numpy(), log.z.to_numpy(), "-", color=RECURRENT, lw=0.8)
    ax.set_xlabel("optimiser step $t$", labelpad=1.5)
    ax.set_ylabel("$x_t$")
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 200, 400])
    ax.set_ylim(-2.8, 2.8)
    ax.set_yticks([-2, 0, 2])

    # The cloud is the data and carries the regime's colour; the exclusion is a mark on
    # it and carries the neutral one. Drawn the other way round, as it was at first, the
    # loop reads as though every sample on it had been excluded.
    bx.plot(points[:, 0], points[:, 1], ".", color=RECURRENT, ms=1.1, mew=0, alpha=0.45,
            zorder=2)
    # The excluded samples belong in (b) and not in (c): the record wraps the loop many
    # times over, so the seventy-six steps either side of the reference land all over it --
    # thirty-one points whose largest angular gap around the loop is 26 degrees -- and only
    # one of them falls inside (c)'s zoom. Their being spread out is the recurrent regime's
    # whole property. How many steps one turn takes is deliberately not claimed: both files
    # this figure reads are decimated above the rate the log turns over at, so a period
    # measured from either is an alias.
    bx.plot(points[excluded, 0], points[excluded, 1], ".", color=GREY, ms=3.6, mew=0,
            ls="none", zorder=4)
    bx.plot([points[centre, 0]], [points[centre, 1]], "o", mfc="white", mec=RECURRENT,
            mew=1.3, ms=5.4, ls="none", zorder=5)
    bx.set_xlabel("$x_t$", labelpad=1.5)
    bx.set_ylabel(r"$x_{t-\tau}$")
    span = 1.12 * np.abs(points).max()
    bx.set_xlim(-span, span)
    bx.set_ylim(-span, span)
    bx.set_xticks([-2, 0, 2])
    bx.set_yticks([-2, 0, 2])

    cx.plot(points[~excluded, 0], points[~excluded, 1], ".", color=FAINT, ms=1.3, mew=0,
            zorder=2)
    cx.add_patch(Circle(tuple(points[centre]), radius, fill=False, lw=0.8, ec=RECURRENT,
                        ls=(0, (3, 2)), zorder=4))
    for j in nearest:
        cx.plot([points[centre, 0], points[j, 0]], [points[centre, 1], points[j, 1]], "-",
                color=RECURRENT, lw=0.5, alpha=0.8, zorder=4)
    cx.plot(points[nearest, 0], points[nearest, 1], ".", color=RECURRENT, ms=3.4, mew=0,
            ls="none", zorder=5)
    cx.plot([points[centre, 0]], [points[centre, 1]], "o", mfc="white", mec=RECURRENT,
            mew=1.3, ms=5.4, ls="none", zorder=6)
    pad = 1.5 * radius
    cx.set_xlim(points[centre, 0] - pad, points[centre, 0] + pad)
    cx.set_ylim(points[centre, 1] - pad, points[centre, 1] + pad)
    cx.set_xticks([])
    cx.set_yticks([])
    # The power law as the panel's axis name: it is what the panel is for and it costs no
    # row of its own. The radius has to be named somewhere, and this is the one pointer.
    cx.set_xlabel(r"$m(r) \propto r^{\,d}$", labelpad=1.5)
    cx.annotate(r"$r_m$", xy=(points[centre, 0] - 0.70 * radius,
                              points[centre, 1] + 0.70 * radius),
                xytext=(points[centre, 0] - 1.34 * radius,
                        points[centre, 1] + 1.16 * radius),
                ha="left", va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=GREY, shrinkA=1, shrinkB=1),
                **POINTER)

    # Built by hand rather than collected from the axes: a key drawn at the marker size
    # the panel uses is a dot two pixels across, which names nothing.
    handles = [Line2D([], [], color=RECURRENT, marker=".", ms=6, mew=0, alpha=0.45,
                      ls="none", label="the reconstruction"),
               Line2D([], [], color=GREY, marker=".", ms=8, mew=0, ls="none",
                      label=r"excluded: $|\Delta t| \leq W_T$"),
               Line2D([], [], mfc="white", mec=RECURRENT, mew=1.3, marker="o", ms=5.4,
                      ls="none", label="reference point"),
               Line2D([], [], color=RECURRENT, marker=".", ms=8, mew=0, ls="none",
                      label=r"its $m$ nearest")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=0.8, handletextpad=0.35, columnspacing=1.1)
    return fig


# ----------------------------------------------------------------- figure 2
def fig_regimes(read: Reader):
    """Recovery, and whether it can be told without ground truth, on the image-data system."""
    d = read.table("sweep_raw")
    d = d[(~d.eta_zero) & (~d.observer.isin(["acc_probe", "loss_step"]))]

    series = [
        ("qp", "recurrent, fast", RECURRENT, "-", "o"),
        ("qp_slow", "recurrent, slow", RECURRENT, "--", "s"),
        ("mixed", "recurrent + noise", STOCHASTIC, "--", "^"),
        ("noise", "stochastic", STOCHASTIC, "-", "v"),
        ("gd", "transient", TRANSIENT, ":", "D"),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.66),
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
    # The paper's own symbol, not "estimated dimension". Two reasons: the phrase runs
    # off the left edge at 1.66 in of panel height, and this panel plots the transient
    # and mini-batch arms, where section 3.3 says the value is a statistic of the window
    # and not a dimension at all. fig_tau keeps the phrase because every case in it is
    # recurrent.
    ax.set_ylabel(r"$\hat{d}_{\mathrm{MG}}$")
    ax.set_title("(a) the estimate against truth", loc="left")

    bx.axvspan(0.95, 1.10, color=BAND, alpha=0.08, lw=0, zorder=0)
    # Two sub-rows per arm, one per rank, because the rank is what connects this
    # panel to (a): the slow recurrent arm tracks the truth at r = 2 and saturates
    # by r = 6, and the diagnostic has to be seen changing across the same step or
    # the panel only repeats what the regime labels already say. Fill encodes the
    # rank (open r = 2, solid r = 6); the observers stay separated within a sub-row
    # so that coincident values remain countable.
    RANK_DY = {2: 0.18, 6: -0.18}
    OBS_DY = {"c_proj1": 0.07, "g_fro": 0.0, "w_fro": -0.07}
    for i, (arm, label, c, ls, mk) in enumerate(series):
        v = ident.get(arm)
        if v is None or v.empty:
            continue
        y = len(series) - 1 - i
        for rank, dy in RANK_DY.items():
            w = v[v.r == rank]
            if w.empty:
                continue
            bx.plot([w.ident_ratio.min(), w.ident_ratio.max()], [y + dy] * 2, "-",
                    color=c, lw=0.6, alpha=0.40, zorder=2)
            bx.plot(w.ident_ratio.values, y + dy + w.observer.map(OBS_DY).values,
                    mk, color=c, ms=3.0, mec=c if rank == 2 else "white",
                    mfc="white" if rank == 2 else c, mew=0.6, zorder=3,
                    clip_on=False)
    bx.set_yticks(range(len(series)))
    bx.set_yticklabels([lbl for _, lbl, _, _, _ in series][::-1])
    bx.set_ylim(-0.65, len(series) - 0.02)
    # The band has to stay in frame whatever the ratios do: the panel is about which
    # regimes fall inside it, and a frame that cut it off would answer the question by
    # cropping. Everything else follows the data.
    ratios = pd.concat(ident.values()).ident_ratio if ident else pd.Series([1.0])
    low, high = bounds(ratios, pad=0.05, step=0.1, include=[0.95, 1.10])
    bx.set_xlim(low, high)
    bx.set_xticks(np.arange(low, high + 1e-9, 0.2))
    bx.set_xlabel(r"identifiability ratio $\rho_{\mathrm{ident}}$")
    bx.set_title("(b) the same cases", loc="left")
    bx.text(1.025, 4.42, "accurate cases", color=BAND, fontsize=7.0, ha="center",
            va="bottom")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=2.8, mec="white",
                      mew=0.5, label=lbl) for _, lbl, c, ls, mk in series]
    fig.tight_layout(rect=[0, 0.115, 1, 1], w_pad=1.4)
    # Five entries across 5.5 in is the widest legend in the article, and at 9 pt type it
    # is the one that runs out of room first. The handles and the gaps carry the reduction
    # rather than the type, which has to stay level with the other eleven figures.
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=1.5,
               handletextpad=0.35, columnspacing=0.75)
    return fig


# ----------------------------------------------------------------- figure 3
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

    panels = [("fn_PR_pos_det", "(c) functions", None),
              ("PR_pos_det", "(d) parameters", None),
              ("move", "(e) displacement", "log"),
              ("MG", "(f) scalar log", None)]

    groks = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    ctrls = ["mod_wd0", "s5_wd0"]

    fig, flat = plt.subplots(2, 3, figsize=(5.5, 2.72), sharex=True)
    curve_ax, log_ax = flat[0, 0], flat[0, 1]
    axes = [flat[0, 2], flat[1, 0], flat[1, 1], flat[1, 2]]
    grid = np.arange(-5000, 5200, 100)

    # Panel (a) is the run itself, what a reader of the grokking literature recognises.
    # Only mod_wd1 has its training curve in data/ -- the other five campaigns promoted
    # milestones alone -- so it is one run where the rest are all six. Said in the caption.
    curve = read.table("mod_wd1_train")
    shift = curve.step.to_numpy() - meta["mod_wd1"]["t_gen"]
    curve_ax.plot(shift, curve.train_acc.to_numpy(), "-", color=FAINT, lw=0.9)
    curve_ax.plot(shift, curve.val_acc.to_numpy(), "-", color=STOCHASTIC, lw=1.1)
    curve_ax.text(-4700, 0.91, "train", ha="left", va="center", **POINTER)
    curve_ax.text(-4700, 0.20, "validation", ha="left", va="center", **POINTER)
    curve_ax.set_ylim(-0.05, 1.05)
    curve_ax.set_yticks([0, 0.5, 1.0])
    curve_ax.set_title("(a) accuracy", loc="left")

    # The scalar the estimator is given, for every run rather than one. Weight decay
    # separates the two groups and moves the observer with them, which is the confound
    # the Limitations paragraph names and which nothing else in the article shows.
    for run in groks:
        g = d[d.run == run].sort_values("centre")
        log_ax.plot(g.centre.to_numpy() - meta[run]["t_gen"], g.pnorm.to_numpy(),
                    "-", color=STOCHASTIC, lw=0.7, alpha=0.8)
    for run, ls in zip(ctrls, ["--", (0, (1, 1.6))]):
        g = d[d.run == run].sort_values("centre")
        ref = meta["mod_wd1" if run.startswith("mod") else "s5_wd1"]["t_gen"]
        log_ax.plot(g.centre.to_numpy() - ref, g.pnorm.to_numpy(),
                    linestyle=ls, color=TRANSIENT, lw=1.0)
    log_ax.set_yscale("log")
    log_ax.set_yticks([30, 50, 100])
    log_ax.set_yticklabels(["30", "50", "100"])
    log_ax.minorticks_off()
    log_ax.set_ylabel("norm")
    log_ax.set_title("(b) the observer", loc="left")

    for ax in (curve_ax, log_ax):
        ax.axvline(0, color=FAINT, lw=0.7, ls=(0, (2, 2.5)), zorder=0)

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
        ax.set_title(title, loc="left")

    for ax in list(flat.ravel()):
        ax.set_xlim(-5000, 5000)
        ax.set_xticks([-5000, 0, 5000])
        ax.set_xticklabels(["$-5$k", "$t_{\\mathrm{gen}}$", "5k"])

    axes[0].set_ylabel(r"$\mathrm{PR}^{\mathrm{det}}$")
    axes[1].set_ylabel(r"$\mathrm{PR}^{\mathrm{det}}$")
    # (e) is named by its title alone: "displacement" beside "(d) parameters" is the
    # collision this pass exists to remove.
    axes[3].set_ylabel("components")

    handles = [
        Line2D([], [], color=STOCHASTIC, lw=1.5, label="generalises (4)"),
        Line2D([], [], color=TRANSIENT, ls="--", lw=1.0,
               label="no weight decay (2)"),
        mpl.patches.Patch(facecolor=STOCHASTIC, alpha=0.55, lw=0,
                          label="hash-family spread"),
    ]
    fig.tight_layout(rect=[0, 0.105, 1, 1], w_pad=1.0, h_pad=0.9)
    # one x label under six panels: placed against the measured axes box, because
    # supxlabel(y=...) and tight_layout(rect=...) do not know about each other and
    # leave a band of white between the ticks and the label
    fig.text(0.5, min(a.get_position().y0 for a in flat.ravel()) - 0.075,
             "steps since generalisation", ha="center", va="top")
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.4)
    return fig


# ----------------------------------------------------------------- figure 4
def fig_map(read: Reader):
    """Where the training logs of section 7 fall in the plane of the two diagnostics."""
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


# ----------------------------------------------------------------- figure 5
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


# ----------------------------------------------------------------- figure 6
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
    # The ramped-gain nuisance bound of section 6.3, read from the file that sets it
    # rather than copied: it moved from 1.16 to 1.53 when the controls were rerun, and a
    # copy here would have drawn the figure against a floor the text no longer states.
    controls = read.table("controls_scored")
    FLOOR = float(controls[(controls["mode"] == "qp")
                           & (controls.control == "obs_scale")].between.median())
    d["centre"] = d.right_step - HALF

    # (run, regime colour, marker). Whether a run generalised is read from its outcome
    # record and not written down here: the two low-data seeds exchanged outcomes when the
    # runs were redone, and a hard-coded flag would have drawn the one that generalises as
    # a control and put it in panel (b) under the wrong heading.
    runs = [
        ("grokpos_s0",   STOCHASTIC, "o"),
        ("lowdata20_s0", STOCHASTIC, "^"),
        ("lowdata15_s0", STOCHASTIC, "s"),
        ("lowdata15_s1", STOCHASTIC, "v"),
        ("lowdata15_s2", STOCHASTIC, "D"),
        ("wd0_s0",       TRANSIENT,  "o"),
        ("wd0_s1",       TRANSIENT,  "s"),
    ]
    generalises = {r: bool(np.isfinite(tgen.get(r, np.nan))) for r, _, _ in runs}
    runs = [(r, c, generalises[r],
             "-" if generalises[r] else ((0, (1, 1.6)) if c is TRANSIENT else "--"), mk)
            for r, c, mk in runs]

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
    changes = []
    for r, c, gen, ls, mk in runs:
        if not gen:
            continue
        g = d[d.run == r].sort_values("centre")
        off = (g.centre - tgen[r]).values
        ref = g.MG.values[np.abs(off).argmin()]
        y = g.MG.values - ref
        changes.append(y)
        # every marker carries the span of the window it summarises, so that the
        # reader can see that consecutive windows overlap by three quarters and
        # that the record localises nothing to better than +-20,000 steps
        for xi, yi in zip(off, y):
            bx.plot([(xi - HALF) / 1000, (xi + HALF) / 1000], [yi, yi], "-",
                    color=c, lw=0.6, alpha=0.28, zorder=2)
        bx.plot(off / 1000, y, linestyle=ls, color=c, marker=mk, ms=2.6,
                mec="white", mew=0.35, lw=0.9, zorder=3)
    bx.set_xlim(-108, 108)
    # The floor band is what the changes are being compared against, so it stays in
    # frame; the rest follows whatever the runs did.
    low, high = bounds(*changes, pad=0.06, step=2.5, include=[-FLOOR, FLOOR])
    bx.set_ylim(low, high)
    bx.set_xticks([-100, 0, 100])
    bx.set_xticklabels(["$-100$k", "$t_{\\mathrm{gen}}$", "100k"])
    bx.set_yticks([t for t in np.arange(-30, 31, 5.0) if low <= t <= high])
    bx.set_xlabel("steps since generalisation")
    bx.set_ylabel("change (components)")
    bx.set_title("(b) aligned on $t_{\\mathrm{gen}}$", loc="left")
    bx.text(105, 1.45, "floor", ha="right", va="bottom", **POINTER)

    # ---- (c) every stride-to-stride change, by outcome -----------------------
    cx.axvline(FLOOR, color=FAINT, lw=0.7, ls=(0, (2, 2.5)), zorder=1)
    ROWS = {True: 1.0, False: 0.0}
    seen = {True: 0, False: 0}
    nrun = {gen: max(2, sum(1 for _, _, g, _, _ in runs if g is gen))
            for gen in (True, False)}
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
    cx.set_title("(c) all changes", loc="left")
    cx.spines["left"].set_visible(False)
    cx.tick_params(axis="y", length=0)
    cx.text(1.45, 1.52, "floor", ha="left", va="top", **POINTER)

    # The counts are the runs actually drawn, so that a run changing outcome relabels the
    # legend instead of leaving it claiming a split the panels no longer show.
    def count(colour, gen):
        return sum(1 for _, c, g, _, _ in runs if c is colour and g is gen)

    handles = [
        Line2D([], [], color=STOCHASTIC, ls="-", marker="o", ms=2.6, mec="white", mew=0.35,
               label=f"stochastic, generalises ({count(STOCHASTIC, True)})"),
        Line2D([], [], color=STOCHASTIC, ls="--", marker="v", ms=2.6, mec="white", mew=0.35,
               label=f"stochastic, does not ({count(STOCHASTIC, False)})"),
        Line2D([], [], color=TRANSIENT, ls=(0, (1, 1.6)), marker="s", ms=2.6, mec="white",
               mew=0.35, label=f"transient, does not ({count(TRANSIENT, False)})"),
    ]
    fig.tight_layout(rect=[0, 0.080, 1, 1], w_pad=1.5)
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0,
               columnspacing=1.3)
    return fig


# ----------------------------------------------------------------- figure 7
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


# ----------------------------------------------------------------- figure 8
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


# ----------------------------------------------------------------- figure 9
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


# ---------------------------------------------------------------- figure 10
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


# ---------------------------------------------------------------- figure 11
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


# ---------------------------------------------------------------- figure 12
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


# ---------------------------------------------------------------- figure 13
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
         real["step"].to_numpy() / 1000.0, STOCHASTIC, "stochastically forced",
         "step ($10^3$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.78))
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


# ---------------------------------------------------------------- figure 14
def fig_signal(read: Reader):
    """Four logs that look alike, and the four levels the estimator reads off them."""
    series = read.table("curve_series")
    windows = read.table("curve_windows")
    ranks = sorted(series.r.unique())
    marks = ["o", "s", "^", "D"]

    # 400 samples, not the whole record: the drive fills one octave of a period near
    # sixteen samples, so ten thousand of them at an inch and a quarter wide is a block of
    # ink. The point of the row is that the four look alike, which needs them legible.
    span = 400

    fig = plt.figure(figsize=(5.5, 2.60))
    grid = fig.add_gridspec(2, 2 * len(ranks), height_ratios=[0.92, 1.0])

    for column, rank in enumerate(ranks):
        ax = fig.add_subplot(grid[0, 2 * column:2 * column + 2])
        one = series[(series.r == rank) & (series["sample"] < span)]
        ax.plot(one["sample"].to_numpy(), one.z.to_numpy(), "-", color=RECURRENT, lw=0.6)
        ax.set_title(f"$r = {rank}$", pad=2.5)
        ax.set_ylim(-3.0, 3.0)
        ax.set_yticks([-2, 0, 2])
        ax.set_xticks([0, 200, 400])
        if column:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("standardised")
        ax.set_xlabel("sample", labelpad=1.5)

    # The lower panels carry the same four series; only the quantity differs. The estimator
    # separates the ranks and the roughness does not, which is the whole of the smoothness
    # control and is otherwise a yes/no column in a table. Rank is the marker, because
    # every one of these is the same regime and so the same colour.
    ex = fig.add_subplot(grid[1, :len(ranks)])
    rx = fig.add_subplot(grid[1, len(ranks):])
    handles = []
    for rank, mark in zip(ranks, marks):
        one = windows[windows.r == rank].sort_values("centre")
        x = one.centre.to_numpy() / 1000.0
        for ax, column in ((ex, "MG"), (rx, "roughness")):
            ax.plot(x, one[column].to_numpy(), "-", color=RECURRENT, lw=0.9,
                    marker=mark, ms=2.6, markevery=5, mec="white", mew=0.3)
        handles.append(Line2D([], [], color=RECURRENT, lw=0.9, marker=mark, ms=2.6,
                              mec="white", mew=0.3, label=f"$r = {rank}$"))
        ex.axhline(float(one.truth.iloc[0]), color=FAINT, lw=0.7, ls=(0, (2, 2.5)),
                   zorder=0)

    ex.set_ylabel("components")
    rx.set_ylabel("roughness")
    ex.set_title("(a) the estimate", loc="left", pad=2.5)
    rx.set_title("(b) the roughness", loc="left", pad=2.5)
    ex.set_ylim(0.0, max(ranks) + 1.2)
    rx.set_ylim(*bounds(windows.roughness, pad=0.35))
    for ax in (ex, rx):
        ax.set_xlim(3.85, 6.15)
        ax.set_xticks([4, 5, 6])
        ax.set_xlabel("window centre ($10^3$)", labelpad=1.5)

    fig.tight_layout(rect=[0, 0.115, 1, 1], h_pad=0.9, w_pad=1.4)
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.8, columnspacing=1.6)
    return fig


# ---------------------------------------------------------------- figure 15
def fig_switch(read: Reader):
    """A known change in the number of phases at matched roughness, tracked as it happens."""
    d = read.table("geometry_switch")
    centres = np.sort(d.centre.unique())
    x = centres / 1000.0
    truth = d.groupby("centre").truth.first().reindex(centres).to_numpy(dtype=float)

    fig, (ax, rx) = plt.subplots(2, 1, figsize=(WIDTH, 1.86), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1.0]})
    # Named above each row rather than beside it: a row of a two-row figure is shorter than
    # either word set on end, and both ran off the canvas.
    for slot, column, ylabel in ((ax, "MG", "(a) the estimate, in components"),
                                 (rx, "roughness", "(b) the roughness")):
        stack = np.vstack([d[d.seed == s].sort_values("centre")[column].to_numpy()
                           for s in sorted(d.seed.unique())])
        slot.fill_between(x, stack.min(axis=0), stack.max(axis=0), facecolor=RECURRENT,
                          alpha=0.20, lw=0, zorder=3)
        slot.plot(x, np.median(stack, axis=0), "-", color=RECURRENT, lw=1.3, zorder=4)
        slot.set_title(ylabel, loc="left", pad=2.5)

    # The truth is a level the schedule holds, drawn as a band behind the estimate and only
    # where it exists. Over a ramp there are two systems inside one window and no number
    # the estimate could be right or wrong about, so those windows are struck out instead:
    # the old panel drew the estimate through them beside an unbroken reference.
    ax.plot(x, truth, "-", color=GREY, lw=5.5, alpha=0.42, solid_capstyle="butt", zorder=2)
    half = 0.5 * float(np.diff(x).min())
    missing = np.isnan(truth)
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        for slot in (ax, rx):
            slot.axvspan(x[lo] - half, x[hi - 1] + half, facecolor="none", edgecolor=FAINT,
                         hatch="////", lw=0.0, zorder=1)

    ax.set_ylim(0.0, 5.2)
    ax.set_yticks([0, 2, 4])
    # From zero, so that the flat trace is read against the fourfold change above it rather
    # than against its own noise: over every window of every seed the roughness lies
    # between 0.085 and 0.094.
    rx.set_ylim(0.0, 0.16)
    rx.set_yticks([0.0, 0.1])
    rx.set_xticks([10, 20, 30])
    rx.set_xlabel("window centre ($10^3$ optimiser steps)", labelpad=1.5)
    # Last, and on the shared axis: an axvspan added after a set_xlim brings the frame back
    # to the origin, which left a third of this one empty.
    rx.set_xlim(x.min(), x.max())

    handles = [Line2D([], [], color=GREY, lw=5.5, alpha=0.42, label="phases forced"),
               Line2D([], [], color=RECURRENT, lw=1.3,
                      label=r"$\hat{d}_{\mathrm{MG}}$, median of eight seeds"),
               Patch(facecolor="none", edgecolor=FAINT, hatch="////",
                     label="a ramp reaches in: no truth")]
    fig.tight_layout(rect=[0, 0.155, 1, 1.03], h_pad=0.4)
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=1.5, handletextpad=0.45, columnspacing=1.2)
    return fig


# ---------------------------------------------------------------- figure 16
def fig_shapes(read: Reader):
    """The object the neighbour search measures, in the plane of its first two coordinates."""
    d = read.table("curve_shapes")
    colour = {"qp": RECURRENT, "gd": TRANSIENT, "batch_proj": STOCHASTIC}
    order = ["one phase", "two phases", "a transient", "mini-batch noise"]

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.60))
    for ax, label in zip(axes, order):
        one = d[d.label == label]
        ax.plot(one.x.to_numpy(), one.y.to_numpy(), ".",
                color=colour[one.arm.iloc[0]], ms=0.9, alpha=0.55, mec="none")
        ax.set_title(label, pad=3.0)
        ax.set_aspect("equal", adjustable="box")
        span = 1.06 * max(np.abs(one.x).max(), np.abs(one.y).max())
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlabel("$z_t$", labelpad=1.0)
    axes[0].set_ylabel(r"$z_{t-\tau}$", labelpad=1.0)

    fig.tight_layout(w_pad=0.7)
    return fig


# ---------------------------------------------------------------- figure 17
def fig_exclusion(read: Reader):
    """Why the exclusion decides the value on a transient and leaves a torus alone."""
    d = read.table("theiler_sweep")
    d = d[(d.observer == "w_fro") & (d.theiler_label != "uncapped")]
    arms = [("fast", "recurrent, fast", RECURRENT, "-"),
            ("slow", "recurrent, slow", RECURRENT, (0, (3.5, 2))),
            ("transient", "transient", TRANSIENT, (0, (1, 1.6)))]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.78))
    handles = []
    for arm, label, colour, dash in arms:
        one = (d[d.arm == arm].groupby("theiler_used")
               .agg(MG=("MG", "median"), near=("frac_near_ref", "median"))
               .sort_index())
        x = one.index.to_numpy()
        ax.plot(x, one.MG.to_numpy(), color=colour, ls=dash, lw=1.1)
        bx.plot(x, one.near.to_numpy(), color=colour, ls=dash, lw=1.1)
        handles.append(Line2D([], [], color=colour, ls=dash, lw=1.1, label=label))

    for panel in (ax, bx):
        panel.set_xscale("symlog", linthresh=1.0)
        panel.set_xlim(0, 200)
        panel.set_xticks([0, 1, 10, 100])
        panel.set_xticklabels(["0", "1", "10", "100"])
        panel.set_xlabel("Theiler exclusion (samples)", labelpad=1.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(["1", "3", "10", "30"])
    ax.set_ylim(0.9, 45)
    ax.set_ylabel("components")
    ax.set_title("(a) the estimate", loc="left", pad=2.5)
    bx.set_ylabel("fraction")
    bx.set_ylim(-0.04, 1.0)
    bx.set_title("(b) neighbours that are returns", loc="left", pad=2.5)

    fig.tight_layout(rect=[0, 0.135, 1, 1], w_pad=1.5)
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=2.2, columnspacing=1.6)
    return fig


# ---------------------------------------------------------------- figure 18
def fig_surrogate(read: Reader):
    """Each run's fall against the null distribution built from that run alone.

    The band is the 5th to 95th percentile of the pooled draws, which is the null the
    test is against. It is not a median-to-maximum interval: a maximum over 195 draws
    is an extreme order statistic, so an observation lands below it almost by
    construction and the picture would then contradict the p values beside it.
    """
    raw = read.table("surrogate_depths")
    raw = raw[(raw.column == "weight_norm") & (raw.smooth == 201)]

    order = [("mod_wd1", "modular, seed 42", True),
             ("mod_wd1_s43", "modular, seed 43", True),
             ("mod_wd1_s44", "modular, seed 44", True),
             ("s5_wd1", "$S_5$ composition", True),
             ("mod_wd0", "modular, no decay", False),
             ("s5_wd0", "$S_5$, no decay", False)]

    fig, ax = plt.subplots(figsize=(5.5, 1.48))
    for i, (run, label, grok) in enumerate(order):
        g = raw[raw.run == run]
        draws = g[g.kind == "surrogate"].depth.to_numpy(dtype=float)
        draws = draws[np.isfinite(draws)]
        low, mid, high = np.percentile(draws, [5, 50, 95])
        observed = float(g[g.kind == "observed"].depth.iloc[0])
        y = len(order) - 1 - i
        colour = STOCHASTIC if grok else TRANSIENT
        ax.plot([low, high], [y, y], "-", color=FAINT, lw=6.0, alpha=0.9,
                solid_capstyle="butt", zorder=2)
        ax.plot([mid], [y], "|", color=GREY, ms=8, mew=1.2, zorder=3)
        ax.plot([observed], [y], "o", color=colour, ms=4.6, mec="white", mew=0.7,
                zorder=4)

    ax.axvline(1.0, color="#AAAAAA", lw=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.93, 5.4)
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xticklabels(["1", "2", "3", "5"])
    ax.minorticks_off()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label for _, label, _ in order][::-1])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("$D$, the fall of the estimate")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=FAINT, lw=6.0, label="surrogates, 5–95 %"),
               Line2D([], [], color=GREY, marker="|", ms=8, mew=1.2, ls="none",
                      label="their median"),
               Line2D([], [], color=STOCHASTIC, marker="o", ms=4.6, mec="white",
                      mew=0.7, ls="none", label="the run: generalises"),
               Line2D([], [], color=TRANSIENT, marker="o", ms=4.6, mec="white",
                      mew=0.7, ls="none", label="no weight decay")]
    fig.tight_layout(rect=[0, 0.135, 1, 1.02])
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=1.5,
               handletextpad=0.4, columnspacing=1.0)
    return fig


# ---------------------------------------------------------------- figure 19
def fig_timing(read: Reader):
    """When the collapse falls, against each run's own transition and against a fixed step."""
    d = read.table("rank_dip")
    meta = {m["run"]: m for m in read.object("rank_milestones")}
    runs = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    search = 4.0        # the dip is the minimum within +-4000 steps of t_gen: appendix G.2

    tgen = np.array([float(meta[run]["t_gen"]) for run in runs]) / 1000.0
    spaces = [("PR_pos_det", "in the parameters", "o", dict(mec="white", mew=0.7)),
              ("fn_PR_pos_det", "in function space", "^", dict(mfc="none", mew=1.2))]
    offsets = {}
    for stat, _, _, _ in spaces:
        one = d[d.stat == stat].set_index("run")
        offsets[stat] = np.array([float(one.loc[run, "at"]) for run in runs]) / 1000.0

    fig, ax = plt.subplots(figsize=(WIDTH, 1.98))

    # What the measurement could have reported at all: the minimum is searched in a window
    # of +-4000 steps about each run's own transition, so nothing outside this band could
    # have been found however the runs behaved. Drawn, because the fixed-step line below
    # leaves it and a reader is owed the reason no point follows it out.
    for sign in (-1, 1):
        ax.axhline(sign * search, color=GREY, lw=0.7, ls=(0, (1.5, 2.5)), zorder=1)
    ax.text(14.7, search - 0.3, "limit of the search", ha="right", va="top", **POINTER)

    # The one fixed absolute step that fits these eight minima best. A collapse happening at
    # a fixed step would put every minimum on this line; the four transitions span 9980
    # steps against a search window of 8000, so no fixed step can even keep all four inside
    # the band, and none of the eight is more than 1.6k from its own transition.
    fixed = float(np.mean([tgen + offsets[stat] for stat, _, _, _ in spaces]))
    grid = np.linspace(2.6, 14.9, 2)
    ax.plot(grid, fixed - grid, "--", color=GREY, lw=1.0, dashes=(4, 3), zorder=2,
            label="one fixed absolute step")
    ax.axhline(0.0, color=GREY, lw=0.6, zorder=1)

    for stat, label, mark, style in spaces:
        ax.plot(tgen, offsets[stat], mark, color=STOCHASTIC, ms=5.2, ls="none", zorder=4,
                label=label, **style)

    ax.set_xlim(2.6, 14.9)
    ax.set_ylim(-6.0, 5.4)
    # The units go on the tick labels: spelled out in the axis name, "minimum minus
    # t_gen, thousands of steps" is longer at 9 pt than the axes are tall, and an appendix
    # figure cannot grow to hold it.
    ax.set_xticks([4, 8, 12])
    ax.set_xticklabels(["4k", "8k", "12k"])
    ax.set_yticks([-4, 0, 4])
    ax.set_yticklabels(["$-4$k", "0", "$+4$k"])
    ax.set_xlabel("$t_{\\mathrm{gen}}$ of the run", labelpad=1.5)
    ax.set_ylabel("minimum $-\\ t_{\\mathrm{gen}}$")

    fig.tight_layout(rect=[0, 0.125, 1, 1.02])
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005), handlelength=1.6,
               handletextpad=0.45, columnspacing=1.3)
    return fig


# ------------------------------------------------------------------ the build
#
# In the order the archived generator drew them, which is the order they appear in the
# article.

PANELS: Dict[str, Callable[[Reader], Any]] = {
    "fig_method": fig_method,
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
    "fig_signal": fig_signal,
    "fig_switch": fig_switch,
    "fig_shapes": fig_shapes,
    "fig_exclusion": fig_exclusion,
    "fig_surrogate": fig_surrogate,
    "fig_timing": fig_timing,
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
    """Draw the requested figures into ``outdir``, all nineteen by default.

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

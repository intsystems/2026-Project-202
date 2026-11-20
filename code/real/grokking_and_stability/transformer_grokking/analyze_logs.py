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
from matplotlib.patches import Patch
from experiment_runner.plotting import CODE, FAINT, GREY, POINTER, RECURRENT, STOCHASTIC, TRANSIENT, WIDTH, bounds
from experiment_runner.plotting import draw


def fig_dip():
    """The trajectory-rank collapse at generalisation, aligned on t_gen.

    Every run here is a mini-batch transformer. The four that generalise carry
    weight decay and are stochastic; the two controls have none and are
    transient, exactly as fig_map() classifies the same two configurations. No
    run is recurrent, so RECURRENT does not appear.
    """
    d = pd.read_csv(HERE / 'logs/rank_windows.csv')
    meta = {m["run"]: m for m in pd.read_csv(HERE / 'logs/rank_milestones.csv').to_dict("records")}

    # panel (d) is a different statistic on the same windows: the delay estimate on the
    # parameter-norm log, at the window matched to the transition (grok.matched.window).
    # It is joined on mid_step, which that experiment takes from this very file, so the
    # two curves are sampled at identical instants and the alignment below is the same
    # alignment.
    e9 = pd.read_csv(HERE / 'logs/headline_trace.csv')
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
    curve = pd.read_csv(HERE / 'logs/mod_wd1_train.csv')
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


def fig_window():
    """The windowed log estimate against t_gen: the change test of section 7.1.

    Colour is the regime as everywhere else. None of these seven runs is
    recurrent, so RECURRENT never appears here: the five regularised runs are
    stochastic and the two weight-decay-zero runs transient, exactly as
    fig_map() classifies them. Generalisation is carried by line style and
    marker, never by hue.
    """
    d = pd.read_csv(HERE / 'logs/real_logs_windows.csv')
    d = d[d.column == "weight_norm"].copy()
    out = pd.read_csv(HERE / 'logs/exp8_outcomes.csv')
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
    controls = pd.read_csv(CODE / 'synthetic/signal_controls_and_sketches/observer_controls/logs/controls_scored.csv')
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


def fig_surrogate():
    """Each run's fall against the null distribution built from that run alone.

    The band is the 5th to 95th percentile of the pooled draws, which is the null the
    test is against. It is not a median-to-maximum interval: a maximum over 195 draws
    is an extreme order statistic, so an observation lands below it almost by
    construction and the picture would then contradict the p values beside it.
    """
    raw = pd.read_csv(HERE / 'logs/surrogates.csv')
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


def fig_timing():
    """When the collapse falls, against each run's own transition and against a fixed step."""
    d = pd.read_csv(HERE / 'logs/rank_dip.csv')
    meta = {m["run"]: m for m in pd.read_csv(HERE / 'logs/rank_milestones.csv').to_dict("records")}
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
    ax.set_xlabel("generalisation step of the run", labelpad=1.5)
    ax.set_ylabel("minimum offset (k)")

    fig.tight_layout(rect=[0, 0.125, 1, 1.02])
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005), handlelength=1.6,
               handletextpad=0.45, columnspacing=1.3)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_dip, fig_window, fig_surrogate, fig_timing,))

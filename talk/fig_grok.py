"""Slides 6, 10, 15-17: the geometric switch, the grokking signal, and the
independent measurement that decides what kind of simplification it is.

`fig_confound` used to live here and drew a three-panel figure that has since
been split into `fig_conf_map`, `fig_conf_fall` and `fig_conf_gain` in
fig_setup.py, one panel per slide. It has been deleted rather than left in place:
while it stayed, running the figure scripts in the documented order silently
overwrote a corrected figure with the old one, and that cost an afternoon once
already.
"""
import sys
sys.path.insert(0, "talk")

import json

import numpy as np
import matplotlib.pyplot as plt

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, POINTER, ANNOT, BOX, DATA, context, rows,
                         table, titles, strip, save)

GROKS = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
CTRLS = ["mod_wd0", "s5_wd0"]
# The two undecayed runs never generalise, so they have no t_gen of their own.
# Each is aligned on the generalisation step of the run it matches in task and
# configuration, which is how fig_dip of the article aligns them.
MATCH = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}

# Named runs appear in three figures; one table so the three cannot disagree.
# A reviewer struck out "s42" and its family as jargon. A seed is an
# implementation detail; what a listener needs is the task and which run of it.
RUN_ROWS = [("mod_wd1", "modular arithmetic, run 1", STOCHASTIC),
            ("mod_wd1_s43", "modular arithmetic, run 2", STOCHASTIC),
            ("mod_wd1_s44", "modular arithmetic, run 3", STOCHASTIC),
            ("s5_wd1", "product in $S_5$", STOCHASTIC),
            ("mod_wd0", "mod. arith., no weight decay", TRANSIENT),
            ("s5_wd0", "$S_5$, no weight decay", TRANSIENT)]


def milestones():
    return {m["run"]: m for m in
            json.loads((DATA / "grok.rank.dip/rank_milestones.json").read_text())}


def aligned(frame, run, meta, x="centre"):
    """One run's windows, with the step axis shifted to its generalisation step."""
    g = frame[frame.run == run].sort_values(x)
    return g[x].to_numpy() - meta[MATCH.get(run, run)]["t_gen"], g


def mean_over_runs(frame, runs, col, meta, grid, x="centre"):
    """The mean over runs of one column, each run first put on the common grid."""
    stack = []
    for run in runs:
        shift, g = aligned(frame, run, meta, x)
        stack.append(np.interp(grid, shift, g[col].to_numpy(),
                               left=np.nan, right=np.nan))
    return np.nanmean(np.vstack(stack), axis=0)


def tgen_axis(slot):
    slot.axvline(0, color=GREY, lw=1.2, ls=(0, (2, 2.5)), zorder=0)
    slot.set_xlim(-5000, 5000)
    slot.set_xticks([-5000, 0, 5000])
    slot.set_xticklabels(["$-5$k", r"$t_{\mathrm{gen}}$", "$+5$k"])


def fig_switch():
    """Slide 6: MG follows a real change of the active dimension; roughness does not."""
    t = table("valid.geometry/geometry_switch_trace.csv")
    g = t.groupby("centre").agg(truth=("truth", "median"), MG=("MG", "median"),
                                lo=("MG", "min"), hi=("MG", "max"),
                                rough=("roughness", "median")).reset_index()
    x = g.centre.to_numpy() / 1000.0

    H = 2.16
    with context():
        # Roughness on a twin axis put a flat line straight through the estimate
        # and read as part of it; a strip of its own says the same thing without
        # the collision, and on the same x so the two are read together.
        fig = plt.figure(figsize=(FULL, H))
        # A wider left margin than a single-row figure needs: the roughness strip
        # underneath is 0.29 in tall and its two-line axis name is what sets the
        # margin, not the main panel's.
        ax = fig.add_axes([0.108, 0.420, 0.864, 0.430])
        sx = fig.add_axes([0.108, 0.238, 0.864, 0.134], sharex=ax)

        # The truth is piecewise constant and undefined inside the two crossings;
        # step-drawing it as one line would invent values there.
        known = np.isfinite(g.truth.to_numpy())
        edges = np.flatnonzero(np.diff(known.astype(int)) != 0)
        for block in np.split(np.arange(len(g)), edges + 1):
            if not known[block[0]]:
                for slot in (ax, sx):
                    slot.axvspan(x[block].min(), x[block].max(),
                                 facecolor="none", edgecolor=STOCHASTIC,
                                 hatch="////", lw=0.9, alpha=0.55, zorder=0)
                continue
            # Thick enough to show as a halo around the estimate drawn over it:
            # at 3.4 pt the blue line hid it wherever the two agreed, which is
            # most of the record and exactly the part worth seeing.
            ax.plot(x[block], g.truth.to_numpy()[block], "-", color=GREY,
                    lw=4.6, alpha=0.45, zorder=1,
                    label="truth: phases switched on" if block[0] == 0 else None)

        ax.fill_between(x, g.lo, g.hi, color=RECURRENT, alpha=0.16, lw=0,
                        zorder=2)
        ax.plot(x, g.MG, "-", color=RECURRENT, lw=2.3, zorder=3,
                label=r"estimate $\hat d_{\mathrm{MG}}$")
        ax.set_ylim(0.4, 5.1)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylabel("components")
        ax.set_xlim(x.min(), x.max())
        ax.tick_params(labelbottom=False)
        # The band used to be captioned "grey band: the switch itself", which a
        # reviewer marked uninformative -- it named the shading without saying
        # what happens in it. It is a legend entry now, and what happens is on
        # the slide.
        #
        # It said "crossfade", and the next reviewer asked what a crossfade is:
        # the word is the implementation's (switch_envelope in systems/clocks.py
        # ramps one arm into the other linearly rather than stepping, so the
        # record stays continuous), and it appeared nowhere else in the deck. The
        # key now says what is in the record over that span, which is the thing a
        # listener needs in order to read the band.
        ax.axvspan(0, 0, facecolor="none", edgecolor=STOCHASTIC,
                   hatch="////", lw=0.9, alpha=0.55,
                   label="signals mixed: no truth")

        sx.plot(x, g.rough, "-", color=TRANSIENT, lw=2.1, zorder=3)
        sx.set_ylim(0, 0.19)
        sx.set_yticks([0, 0.1])
        sx.set_ylabel("rough-\nness", fontsize=9.4, linespacing=1.15)
        sx.set_xlabel("training step, thousands")


        h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=9.4,
                   bbox_to_anchor=(0.52, 0.995), handlelength=2.0,
                   columnspacing=1.0, handletextpad=0.4)
    save(fig, "p_switch")


def fig_signal():
    """Slide 10: the run, the estimate on its log, and the depth of the fall.

    Panel (c) is the statistic the claim is actually made on. The mean trace in
    (b) understates it, because each run reaches its minimum at its own offset
    and averaging flattens four dips that do not coincide; (c) measures each run
    at its own minimum, which is what tab:matched of the article reports.
    """
    meta = milestones()
    curve = table("grok.rank.dip/mod_wd1_train.csv")
    h = table("grok.matched.window/headline_trace.csv")
    h = h[h.column == "weight_norm"].rename(columns={"mid_step": "centre"})
    surr = table("grok.matched.surrogate/surrogate_seed_spread.csv")
    surr = surr[(surr.column == "weight_norm") & (surr.smooth == 201)]
    surr = surr.set_index("run")
    grid = np.arange(-5000, 5100, 100)
    shift = curve.step.to_numpy() - meta["mod_wd1"]["t_gen"]

    H = 2.16
    y0, hh = rows(H, bottom=0.70, top=0.26)   # bottom holds the legend row
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # 0.086, not 0.072: "accuracy" set at 10.5 pt is 0.19 in of rotated
        # label, and with the tick labels in front of it 0.43 in of margin left
        # the last of it hanging off the page.
        ax = fig.add_axes([0.086, y0, 0.222, hh])
        bx = fig.add_axes([0.408, y0, 0.222, hh])
        cx = fig.add_axes([0.766, y0, 0.206, hh])

        # FAINT for the train curve read as nothing at all once projected; it is
        # meant to be subdued, not absent.
        ax.plot(shift, curve.train_acc, "-", color=GREY, alpha=0.45, lw=2.1)
        ax.plot(shift, curve.val_acc, "-", color=STOCHASTIC, lw=2.3)
        ax.text(-4600, 0.88, "train", ha="left", va="center", **POINTER)
        ax.text(-4600, 0.15, "val", ha="left", va="center", color=STOCHASTIC,
                fontsize=9.6)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("accuracy")
        ax.set_xlabel(r"steps from $t_{\mathrm{gen}}$")

        for run in GROKS:
            sh, g = aligned(h, run, meta)
            bx.plot(sh, g.MG, "-", color=STOCHASTIC, lw=0.9, alpha=0.30,
                    zorder=2)
        bx.plot(grid, mean_over_runs(h, GROKS, "MG", meta, grid), "-",
                color=STOCHASTIC, lw=2.7, zorder=4,
                label="generalises (4 runs)")
        for run in CTRLS:
            sh, g = aligned(h, run, meta)
            bx.plot(sh, g.MG, color=TRANSIENT, lw=2.0, ls=(0, (4, 2.5)),
                    zorder=3,
                    label="no weight decay (2)" if run == CTRLS[0] else None)
        bx.set_ylim(1.1, 5.0)
        bx.set_yticks([2, 3, 4])
        bx.set_ylabel(r"$\hat d_{\mathrm{MG}}$")

        bx.set_xlabel(r"steps from $t_{\mathrm{gen}}$")

        for slot in (ax, bx):
            tgen_axis(slot)

        # No row names here. Spelled out they were 1.15 in wide and reached back
        # across panel (b); and the identity of a seed is not what the panel is
        # for. Colour carries the grouping, which panel (b)'s legend has just
        # named, and the p value is written on each bar.
        for i, (run, label, c) in enumerate(RUN_ROWS):
            y = len(RUN_ROWS) - 1 - i
            dpt, p = surr.loc[run, "observed"], surr.loc[run, "p_median"]
            cx.barh(y, dpt, height=0.68, color=c, alpha=0.78, lw=0)
            cx.text(dpt + 0.12, y, f"$p={p:g}$", va="center", ha="left",
                    fontsize=9.2, color=c)
        cx.axvline(1.0, color=GREY, lw=1.1)
        cx.set_yticks([])
        cx.spines["left"].set_visible(False)
        cx.set_xlim(0.9, 5.6)
        cx.set_xticks([1, 2, 3])
        cx.set_ylim(-0.6, 5.7)
        cx.set_xlabel("fall of the estimate")

        strip(fig, bx, 2, x=0.34, fontsize=9.6)
        titles(fig, H, [(0.086, "(a) The run"),
                        (0.395, "(b) Estimate on its log"),
                        (0.766, "(c) Depth, per run")], top=0.235)
    save(fig, "p_signal")


def fig_sketch():
    """Slide 22: the cost of the compression, on trajectories of known rank."""
    a = table("check.sketch.accuracy/sketch_accuracy.csv")
    cost = json.loads((DATA / "check.sketch.cost/sketch_cost.json").read_text())

    H = 2.16
    y0, hh = rows(H, bottom=0.70, top=0.26)   # bottom holds the legend row
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.075, y0, 0.335, hh])
        bx = fig.add_axes([0.625, y0, 0.347, hh])

        ax.plot([0.8, 11], [0.8, 11], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1)
        ax.plot(a.true_rank, a.pr_uncompressed, "o", color=GREY, ms=8.0,
                mfc="none", mew=1.6, zorder=3, label="all weights")
        ax.plot(a.true_rank, a.pr_sketched, "x", color=GOOD, ms=8.5, mew=2.1,
                zorder=4, label="through CountSketch")
        ax.set_xlim(0.5, 11)
        ax.set_ylim(0.5, 11)
        ax.set_xticks([1, 5, 10])
        ax.set_yticks([1, 5, 10])
        ax.set_xlabel("true rank of the trajectory")
        ax.set_ylabel("measured rank")
        ax.text(10.7, 0.8, r"they agree to $\leq 0.11$", ha="right",
                va="bottom", color=GOOD, **ANNOT)

        full, small = cost["full_float32_MB"], cost["sketched_float32_MB"]
        bx.barh(1, full, height=0.60, color=GREY, alpha=0.35, lw=0)
        bx.barh(0, small, height=0.60, color=GOOD, alpha=0.80, lw=0)
        bx.text(full * 1.10, 1, f"{full:.0f} MB", va="center", ha="left",
                fontsize=10.5, color=GREY)
        bx.text(small * 1.75, 0, f"{small:.1f} MB", va="center", ha="left",
                fontsize=10.5, color=GOOD)
        bx.set_xscale("log")
        bx.set_xlim(1.2, 900)
        bx.set_xticks([10, 100])
        bx.set_xticklabels(["10", "100"])
        bx.minorticks_off()
        bx.set_ylim(-0.55, 1.80)
        bx.set_yticks([1, 0])
        bx.set_yticklabels(["all weights", r"sketch, $\mathbb{R}^{1024}$"],
                           fontsize=9.8)
        bx.set_xlabel("storing the trajectory, MB")
        bx.text(1.45, 1.66, rf"$\times${cost['storage_ratio']:.0f} less memory,"
                rf"  $+{100 * cost['overhead_frac']:.1f}\,\%$ time",
                ha="left", va="center", color=RECURRENT, **ANNOT)

        # The two marker styles of (a) named under the figure. In the axes the
        # key sat on the diagonal it was meant to be compared against.
        strip(fig, ax, 2, fontsize=9.6)
        titles(fig, H, [(0.075, "(a) Compression costs no accuracy"),
                        (0.625, "(b) And it is cheap")], top=0.235)
    save(fig, "p_sketch")


def fig_dip():
    """Slide 16: the effective rank of the stored trajectory, run by run."""
    meta = milestones()
    d = table("grok.rank.dip/rank_windows.csv")
    grid = np.arange(-5000, 5100, 100)
    panels = [(0.075, "PR_pos_det", "(a) In the parameters", None),
              (0.404, "fn_PR_pos_det", "(b) In function space", None),
              (0.733, "move", "(c) Path length", "log")]

    H = 2.16
    y0, hh = rows(H, bottom=0.72, top=0.26)
    slots = {}
    with context():
        fig = plt.figure(figsize=(FULL, H))
        for x0, col, name, scale in panels:
            ax = fig.add_axes([x0, y0, 0.239, hh])
            slots[col] = ax
            for run in GROKS:
                sh, g = aligned(d, run, meta)
                ax.plot(sh, g[col], "-", color=STOCHASTIC, lw=1.0, alpha=0.30,
                        zorder=2)
            ax.plot(grid, mean_over_runs(d, GROKS, col, meta, grid), "-",
                    color=STOCHASTIC, lw=2.7, zorder=4,
                    label="generalises (4)")
            for run in CTRLS:
                sh, g = aligned(d, run, meta)
                ax.plot(sh, g[col], color=TRANSIENT, lw=2.0, ls=(0, (4, 2.5)),
                        zorder=3,
                        label="no weight decay (2)" if run == CTRLS[0] else None)
            tgen_axis(ax)
            if scale:
                ax.set_yscale(scale)
                # Plain numbers rather than 10^k: on a slide the exponent form
                # costs a beat of decoding for no information.
                ax.set_yticks([0.1, 1, 10, 100])
                ax.set_yticklabels(["0.1", "1", "10", "100"])
                ax.minorticks_off()
            titles(fig, H, [(x0, name)], top=0.235)

        first = slots["PR_pos_det"]
        first.set_ylabel("effective rank")
        strip(fig, first, 2, x=0.53)
    save(fig, "p_dip")


def fig_depth():
    """Slide 19a: depth alone does not separate the runs.

    This used to share a figure with the timing panel. Once the runs were named
    in words instead of by seed -- "modular arithmetic, run 1" rather than "s42",
    which a reviewer struck out as jargon -- the row labels were 1.5 in wide and
    no two-panel layout left room for both them and a scatter with a reference
    line. The two panels are two slides now, which is also how the three
    confounder panels were split earlier and for the same reason.
    """
    stat = "PR_pos_det"
    par = table("grok.rank.dip/rank_dip.csv")
    par = par[par.stat == stat].set_index("run")
    # For each undecayed run the window aligned on its partner's generalisation
    # step gives one depth, and the deepest fall anywhere in its own run gives
    # another. Both are drawn: the second decides whether depth separates the
    # runs at all, and it says it does not.
    ctrl = table("grok.rank.dip/rank_dip_controls_aligned.csv")
    ctrl = ctrl[ctrl.stat == stat].set_index("run")
    free = table("grok.rank.dip/rank_dip_controls.csv")
    free = free[free.stat == stat].set_index("run")
    depth = {run: par.loc[run, "depth"] for run in GROKS}
    depth.update({run: ctrl.loc[run, "depth"] for run in CTRLS})
    anywhere = {run: free.loc[run, "depth"] for run in CTRLS}

    H = 2.16
    y0, hh = rows(H, bottom=0.70, top=0.22)   # bottom holds the legend row
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # The left margin is set by the row names, which are words now.
        ax = fig.add_axes([0.330, y0, 0.642, hh])

        for i, (run, label, c) in enumerate(RUN_ROWS):
            y = len(RUN_ROWS) - 1 - i
            ax.barh(y, depth[run], height=0.66, color=c, alpha=0.75, lw=0)
            ax.text(depth[run] + 0.06, y, f"{depth[run]:.2f}", va="center",
                    ha="left", fontsize=8.8, color=c)
            if run in anywhere:
                ax.plot([anywhere[run]], [y], "|", color=c, ms=18, mew=2.6,
                        zorder=4, ls="none",
                        label=("deepest fall anywhere in that run"
                               if run == CTRLS[0] else None))
        ax.axvline(1.0, color=GREY, lw=1.1)
        ax.set_yticks(range(len(RUN_ROWS)))
        ax.set_yticklabels([lab for _, lab, _ in RUN_ROWS][::-1], fontsize=9.0)
        ax.set_xlim(0.9, 5.9)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(-0.6, len(RUN_ROWS) - 0.4)
        ax.set_xlabel("how far the effective rank fell at the transition")
        # y = 0.02, not 0, because the tall bar glyph in the key overflows a
        # legend box flush with the figure edge.
        strip(fig, ax, 1, x=0.55, fontsize=9.6)
    save(fig, "p_depth")


def fig_timing():
    """Slide 19b: the timing and the reversal are what separate the runs.

    A collapse that merely followed training progress would sit at a fixed
    absolute step, so its offset from t_gen would fall along the dashed line as
    t_gen grows fourfold across these four runs. It does not: every minimum stays
    within about 1.6k steps of its own run's transition.
    """
    meta = milestones()
    d = table("grok.rank.dip/rank_dip.csv")
    fn = d[d.stat == "fn_PR_pos_det"].set_index("run")
    par = d[d.stat == "PR_pos_det"].set_index("run")

    H = 2.16
    y0, hh = rows(H, bottom=0.70, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.235, y0, 0.520, hh])

        tg = np.array([meta[r]["t_gen"] for r in GROKS], dtype=float)
        span = np.linspace(3000, 15000, 50)
        # Short keys, because three long ones came to 6.67 in on a 5.95 in
        # figure: the first handle started off the left edge and the last label
        # ran off the right one. What the grey line means is on the slide.
        ax.plot(span / 1000, (tg.mean() - span) / 1000, color=GREY, lw=1.2,
                ls=(0, (4, 3)), zorder=1, label="a fixed absolute step")
        # The band is described on the slide; four legend entries needed two
        # rows and the second row landed on the axis name.
        ax.axhspan(-1.6, 1.6, color=STOCHASTIC, alpha=0.12, lw=0, zorder=0)
        ax.plot(tg / 1000, [par.loc[r, "at"] / 1000 for r in GROKS], "o",
                color=STOCHASTIC, ms=8.0, mec="white", mew=1.0, zorder=4,
                ls="none", label="minimum, parameters")
        ax.plot(tg / 1000, [fn.loc[r, "at"] / 1000 for r in GROKS], "^",
                color=STOCHASTIC, mfc="none", mew=1.7, ms=8.0, zorder=4,
                ls="none", label="minimum, function space")
        ax.axhline(0, color=GREY, lw=1.1)
        ax.set_xlim(2.5, 15)
        ax.set_ylim(-5.0, 2.9)
        ax.set_xticks([4, 8, 12])
        ax.set_yticks([-4, -2, 0, 2])
        ax.set_xlabel(r"$t_{\mathrm{gen}}$ of the run, thousands of steps")
        ax.set_ylabel(r"minimum $-\ t_{\mathrm{gen}}$, k")
        strip(fig, ax, 3, x=0.50, fontsize=9.2)
    save(fig, "p_timing")


if __name__ == "__main__":
    fig_switch()
    fig_signal()
    fig_sketch()
    fig_dip()
    fig_depth()
    fig_timing()

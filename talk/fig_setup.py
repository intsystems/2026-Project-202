"""Slide 7 and slides 11-14, added after the supervisor's review.

Two of his notes needed new figures rather than new words.

*"голову толкнули из решения --- надо формул написать/картинку дать".* The
transient arm was described in words on a slide whose figure showed only the
consequence. `fig_setup` shows the input instead: the two scalar logs the
estimator is actually handed, side by side. One decays monotonically over the
whole record and never comes back; the other oscillates and returns. That is the
whole difference the next slide then exploits, and it is visible without any
statistic.

*"Можно было бы на 3 слайда разнести".* The three ways of getting a stable wrong
number were three panels on one slide. They are now `fig_conf_map`,
`fig_conf_fall` and `fig_conf_gain`, one per slide, each large enough to read.

`fig_simplify` came later and reverses the reading of the other three: what looks
like a false alarm in the undecayed runs is a true detection of a real
simplification. See its own docstring.
"""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt

import json

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, BOX, DATA, context, rows, table, titles,
                         save)


def milestones():
    return {m["run"]: m for m in
            json.loads((DATA / "grok.rank.dip/rank_milestones.json").read_text())}


def strip(fig, ax, ncol, x=0.52):
    """The shared one-row legend along the bottom of a figure."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(x, 0.0), handlelength=2.2, fontsize=10.0)


def fig_setup():
    """Slide 7: the two input logs, standardised as the estimator sees them."""
    t = table("valid.theiler.contrast/example_traces.csv")
    x = t["sample"].to_numpy() / 1000.0

    H = 2.16
    y0, h = rows(H, bottom=0.56, top=0.26)
    # The transient needs the whole record for its decay to be visible; the
    # driven log turns over every dozen-odd samples, so across 8000 samples it is
    # a block of ink and only a zoom says anything.
    panels = [(0.088, 0.400, "transient", "(a) No drive: full-batch descent",
               TRANSIENT, None,
               "100 % of steps down,\nno sign change", True),
              (0.560, 0.412, "recurrent", "(b) With drive, first 200 steps",
               RECURRENT, 200,
               "1697 sign changes\nin the record", False)]

    with context():
        fig = plt.figure(figsize=(FULL, H))
        for x0, w, col, name, c, cut, note, ylab in panels:
            ax = fig.add_axes([x0, y0, w, h])
            # standardised exactly as alg:mg does before embedding: the raw logs
            # differ in mean and spread, and the shape is the whole point
            z = ((t[col] - t[col].mean()) / t[col].std()).to_numpy()
            xs, zs = ((x, z) if cut is None
                      else (t["sample"].to_numpy()[:cut], z[:cut]))
            ax.plot(xs, zs, "-", color=c, lw=1.7)
            ax.set_xlim(xs.min(), xs.max())
            # One y range for both, because both logs are standardised and the
            # comparison the slide makes is of shape at a common scale.
            ax.set_ylim(-3.2, 3.2)
            ax.set_yticks([-2, 0, 2])
            ax.set_xlabel("training step, thousands" if cut is None
                          else "training step")
            if ylab:
                ax.set_ylabel("scalar log $x_t$")
            ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right",
                    va="bottom", color=c, linespacing=1.3, bbox=BOX, zorder=5,
                    **ANNOT)
            titles(fig, H, [(x0, name)], top=0.235)
    save(fig, "p_setup")


def fig_conf_map():
    """Slide 11: neither published grokking setting lands where a value may be
    read as a dimension."""
    import pandas as pd

    tr = table("grok.diagnostics.logs/real_logs_summary.csv")
    tr = tr[tr.column == "weight_norm"]
    pc = pd.concat([table("grok.diagnostics.perceptron/dimension_probe_summary.csv"),
                    table("grok.diagnostics.perceptron/dimension_probe_summary_poly.csv")])
    pc = pc[pc.column == "train_loss"]

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.22)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.150, y0, 0.720, h])

        ax.add_patch(plt.Rectangle((8.0, 0.95), 5000 - 8.0, 0.15,
                                   facecolor=RECURRENT, alpha=0.14,
                                   edgecolor=RECURRENT, lw=1.1,
                                   ls=(0, (3, 3)), zorder=0))
        ax.text(3300, 1.025, "in this band the level may be read as a "
                "dimension", ha="right", va="center", color=RECURRENT, **ANNOT)
        undecayed = tr.run.astype(str).str.startswith("wd0")
        # Every one of the ten full-batch runs crosses its own trend exactly
        # twice, so ten markers at x = 2 stacked into a striped block that read
        # as a bar. A min-max range with a tick at the median is the same data
        # and does not pretend to be ten resolvable points; no jitter, because a
        # jitter here would invent a spread in the very quantity that matters.
        ax.plot([2.0, 2.0], [pc.ident.min(), pc.ident.max()], "-",
                color=TRANSIENT, lw=7.0, alpha=0.30, solid_capstyle="butt",
                zorder=2)
        ax.plot([2.0], [pc.ident.median()], "s", mfc="none", mec=TRANSIENT,
                mew=1.6, ms=9.0, zorder=3)
        ax.plot(tr[~undecayed].osc, tr[~undecayed].ident, "o", color=STOCHASTIC,
                ms=8.0, mec="white", mew=1.1, ls="none", zorder=4)
        ax.plot(tr[undecayed].osc, tr[undecayed].ident, "o", color=TRANSIENT,
                ms=8.0, mec="white", mew=1.1, ls="none", zorder=4)
        ax.text(2.6, 1.42, f"perceptron, full batch ({len(pc)})", ha="left",
                va="center", color=TRANSIENT, **ANNOT)
        # Below the cluster, not through it: right-aligned at the axis edge the
        # label ran back across the very five points it names.
        ax.text(4700, 1.30, f"transformer, mini-batch ({(~undecayed).sum()})",
                ha="right", va="center", color=STOCHASTIC, **ANNOT)
        ax.text(4.2, 1.90, f"the same, no weight decay ({undecayed.sum()})",
                ha="left", va="center", color=TRANSIENT, **ANNOT)
        ax.set_xscale("log")
        ax.set_xlim(1.15, 6000)
        ax.set_ylim(0.9, 2.05)
        ax.set_xticks([2, 10, 100, 1000])
        ax.set_xticklabels(["2", "10", "100", "1000"])
        ax.minorticks_off()
        ax.set_xlabel("times the log crossed its own trend, whole record")
        # Not the full definition: spelled out, the fraction is 2 in tall and the
        # panel is 1.4. It is defined on the slide before this one.
        ax.set_ylabel(r"$\rho_{\mathrm{ident}}$")
    save(fig, "p_conf_map")


def fig_conf_fall():
    """Slide 12: the estimate falls fivefold in runs that never generalise."""
    w = table("grok.diagnostics.logs/real_logs_windows.csv")
    w = w[w.column == "train_loss"]
    out = table("grok.extended.outcomes/exp8_outcomes.csv")

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.090, y0, 0.360, h])
        bx = fig.add_axes([0.610, y0, 0.362, h])

        for run, dash in (("wd0_s0", "-"), ("wd0_s1", (0, (4, 2.5)))):
            g = w[w.run == run].sort_values("right_step")
            ax.plot(g.right_step / 1000.0, g.MG, linestyle=dash,
                    color=TRANSIENT, lw=2.2, marker="D", ms=5.2, mec="white",
                    mew=0.8)
        g = w[w.run == "grokpos_s0"].sort_values("right_step")
        ax.plot(g.right_step / 1000.0, g.MG, "-", color=STOCHASTIC, lw=2.0,
                marker="o", ms=5.2, mec="white", mew=0.8)
        ax.set_yscale("log")
        ax.set_ylim(9, 190)
        ax.set_yticks([10, 20, 50, 100])
        ax.set_yticklabels(["10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlim(30, 125)
        ax.set_xticks([40, 60, 80, 100, 120])
        ax.set_xlabel("window right edge, thousands of steps")
        # Short enough to fit the axes height: a rotated label is as long as its
        # own text, and this panel is 1.38 in tall.
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}$, loss log")
        ax.text(123, 11.2, "no weight decay (2)", ha="right", va="top",
                color=TRANSIENT, **ANNOT)
        ax.text(123, 19.5, "a run that groks", ha="right", va="bottom",
                color=STOCHASTIC, **ANNOT)
        # A span rather than a pointer: any arrow from a clear spot to the
        # descent crossed the curve it was pointing at. It lives left of the
        # first window, where there is no data for it to cross.
        ax.annotate("", xy=(32.3, 63.8), xytext=(32.3, 12.1),
                    arrowprops=dict(arrowstyle="<->", lw=1.3, color=TRANSIENT,
                                    shrinkA=0, shrinkB=0))
        ax.text(33.4, 28, r"$\times 5$", fontsize=11.5, color=TRANSIENT,
                ha="left", va="center", bbox=BOX, zorder=5)

        # and none of that is generalisation: the same runs stay at chance
        order = out.sort_values("max_val", ascending=False)
        names = {"grokpos_s0": "groks", "lowdata15_s1": "lowdata15 s1",
                 "lowdata20_s0": "lowdata20", "lowdata15_s0": "lowdata15 s0",
                 "wd0_s0": "no WD s0", "wd0_s1": "no WD s1",
                 "lowdata15_s2": "lowdata15 s2"}
        for i, row in enumerate(order.itertuples()):
            y = len(order) - 1 - i
            highlight = str(row.run).startswith("wd0")
            c = TRANSIENT if highlight else (STOCHASTIC if row.groks else GREY)
            bx.barh(y, row.max_val, height=0.66, color=c,
                    alpha=0.85 if highlight else 0.55, lw=0)
            bx.text(row.max_val + 0.03, y, f"{row.max_val:.2f}", va="center",
                    ha="left", fontsize=9.0, color=c)
        bx.set_yticks(range(len(order)))
        bx.set_yticklabels([names[r] for r in order.run][::-1], fontsize=9.0)
        bx.set_xlim(0, 1.42)
        bx.set_xticks([0, 0.5, 1.0])
        bx.set_ylim(-0.6, len(order) - 0.4)
        bx.set_xlabel("best validation accuracy reached")

        # Not "nothing generalises": three of these runs do. The claim is about
        # the two gold ones, the same two whose estimate fell fivefold.
        titles(fig, H, [(0.090, "(a) The estimate falls fivefold"),
                        (0.610, "(b) Yet they never generalise")], top=0.235)
    save(fig, "p_conf_fall")


def fig_conf_gain():
    """Slide 13: a mechanism for that fall that involves no dimension at all."""
    c = table("valid.nuisance/controls_scored.csv")
    c = c[c["mode"] == "qp"]

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.22)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # A wide left margin, because the row names are words: at 0.175 the
        # longest two were cut off mid-syllable.
        ax = fig.add_axes([0.228, y0, 0.742, h])

        order = [("baseline", "baseline", GREY),
                 ("freq_half", r"frequency $/\,2$", GREY),
                 ("freq_double", r"frequency $\times 2$", GREY),
                 ("rotate", "rotated observer", GREY),
                 ("amp_ramp", "amplitude ramp", STOCHASTIC),
                 ("obs_scale", "log gain ramp", STOCHASTIC)]
        base = c[c.control == "baseline"].mg_all.median()
        for i, (key, label, col) in enumerate(order):
            v = c[c.control == key].mg_all.median()
            y = len(order) - 1 - i
            ax.barh(y, v - base, left=base, height=0.66, color=col,
                    alpha=0.30 if col is GREY else 0.80, lw=0)
            if key != "baseline":
                ax.text(max(v, base) + 0.05, y, f"{v - base:+.2f}", va="center",
                        ha="left", fontsize=9.8, color=col)
        ax.axvline(base, color=GREY, lw=1.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab for _, lab, _ in order][::-1], fontsize=9.8)
        ax.set_xlim(3.9, 6.5)
        ax.set_xticks([4, 4.5, 5, 5.5, 6])
        ax.set_ylim(-0.6, 5.7)
        ax.set_xlabel(r"$\hat d_{\mathrm{MG}}$ with the truth fixed at $r = 4$")
        ax.text(6.42, 5.60, "what a run without weight\ndecay supplies for "
                "free", ha="right", va="top", color=STOCHASTIC,
                linespacing=1.3, **ANNOT)
    save(fig, "p_conf_gain")


def fig_simplify():
    """Slide 14: the estimate is a simplification detector, and at WD = 0 the
    simplification it detects is real.

    Read as a confounder, the fall of the estimate in the undecayed runs looks
    like a false alarm. It is not one. Their trajectory really has collapsed onto
    a single direction -- the undetrended effective rank is 1.00 for the whole
    record of mod_wd0 -- and that direction is the monotone growth of the
    parameter norm, which rises on 100 per cent of the logged steps. The
    estimator says "this motion has become simple" and it is right. What it
    cannot say alone is which kind of simplification: a transient collapse at a
    phase transition, or a permanent degeneracy. That takes the detrended rank of
    the next slide.
    """
    meta = milestones()
    d = table("grok.rank.dip/rank_windows.csv")
    match = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
    groks = ["mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1"]
    ctrls = ["mod_wd0", "s5_wd0"]

    H = 2.16
    y0, h = rows(H, bottom=0.72, top=0.26)      # bottom holds the legend too
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.085, y0, 0.385, h])
        bx = fig.add_axes([0.585, y0, 0.387, h])

        for run in groks + ctrls:
            g = d[d.run == run].sort_values("centre")
            shift = g.centre.to_numpy() - meta[match.get(run, run)]["t_gen"]
            grok = run in groks
            c = STOCHASTIC if grok else TRANSIENT
            style = dict(color=c, lw=2.2 if not grok else 1.7,
                         alpha=1.0 if not grok else 0.72,
                         ls="-" if grok else (0, (4, 2.5)))
            lab = ("generalises (4)" if run == groks[0]
                   else "no weight decay (2)" if run == ctrls[0] else None)
            ax.plot(shift, g.pnorm, label=lab, **style)
            bx.plot(shift, g.PR_pos, **style)

        for slot in (ax, bx):
            slot.axvline(0, color=GREY, lw=1.2, ls=(0, (2, 2.5)), zorder=0)
            slot.set_xlim(-5000, 5000)
            slot.set_xticks([-5000, 0, 5000])
            slot.set_xticklabels(["$-5$k", r"$t_{\mathrm{gen}}$", "$+5$k"])

        # Short axis names, not descriptive ones: a rotated y label is as long
        # as its text, and "parameter norm ||theta||" set at 10.5 pt is 1.6 in
        # against 1.22 in of axes, so it overflowed both ends and was clipped.
        ax.set_ylabel(r"norm $\|\theta\|$")
        ax.set_ylim(25, 128)
        ax.text(-4700, 118, "rises on 100 % of steps", ha="left", va="center",
                color=TRANSIENT, bbox=BOX, zorder=5, **ANNOT)
        # The rose curves are named in the legend and discussed in the slide's
        # own text; a label here had nowhere to sit that was not on top of them.

        bx.set_ylabel("effective rank")
        bx.set_ylim(0.6, 6.4)
        bx.set_yticks([1, 2, 3, 4, 5])
        bx.axhline(1.0, color=GOOD, lw=1.3, ls=(0, (4, 3)), zorder=1)
        bx.text(4700, 1.16, "one direction only", ha="right", va="bottom",
                color=GOOD, bbox=BOX, zorder=5, **ANNOT)

        strip(fig, ax, 2, x=0.53)
        titles(fig, H, [(0.085, "(a) Parameter norm"),
                        (0.585, "(b) Rank of the trajectory")], top=0.235)
    save(fig, "p_simplify")


if __name__ == "__main__":
    fig_setup()
    fig_conf_map()
    fig_conf_fall()
    fig_conf_gain()
    fig_simplify()

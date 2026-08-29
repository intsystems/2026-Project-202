"""Backup slides: the delay lag, what is actually being estimated, and the
surrogate test behind the matched-window claim."""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, BOX, context, rows, table, titles, save)


def fig_tau():
    """The lag has to be set from a period no training log reveals."""
    d = table("valid.tau/tau_sensitivity.csv")
    d = d[(d.period == 400) & (d.max_E == 20) & (d.tau != "acorr")].copy()
    d["span_periods"] = d.span_periods.astype(float)

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.108, y0, 0.342, h])
        bx = fig.add_axes([0.600, y0, 0.372, h])

        # Every rank divided by its own truth: on this scale exact recovery is
        # the line at one, whatever r is, and all six ranks share a panel.
        # Six ranks in one blue was a reviewer's note. At six curves the marker
        # is the only thing telling them apart, and a marker is harder to follow
        # across a crossing than a hue is. Paul Tol's qualitative set, so the
        # article's palette is extended rather than replaced.
        marks = ["o", "s", "^", "v", "D", "X"]
        hues = ["#004488", "#117733", "#997700", "#BB5566", "#AA4499",
                "#44AA99"]
        for (r, mk), hue in zip(zip(sorted(d.r.unique()), marks), hues):
            g = d[d.r == r].groupby("span_periods").MG.median().sort_index()
            ax.plot(g.index, g.values / r, "-", marker=mk, color=hue,
                    ms=4.8, mec="white", mew=0.6, lw=1.6, label=f"$r={r}$")
        ax.axhline(1.0, color=GREY, lw=1.4, ls=(0, (4, 3)), zorder=1)
        ax.axvspan(0.15, 0.45, color=GOOD, alpha=0.15, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.04, 1.9)
        ax.set_ylim(0.12, 11)
        ax.set_xticks([0.05, 0.2, 1.0])
        ax.set_xticklabels(["0.05", "0.2", "1"])
        ax.set_yticks([0.25, 1, 4])
        ax.set_yticklabels(["0.25", "1", "4"])
        ax.minorticks_off()
        ax.set_xlabel("delay window / oscillation period")
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}} \,/\, r$")
        # (b) an unequal drive separates the active dimension from the effective
        # rank, and the estimate follows the first
        a = table("valid.anisotropy/aniso_summary.csv")
        k = a[a.r == 4].sort_values("rho")
        bx.axhline(4.0, color=GREY, lw=1.4, ls=(0, (4, 3)), zorder=1,
                   label=r"truth: $d_{\mathrm{act}} = 4$")
        bx.plot(k.rho, k.MG, "-o", color=RECURRENT, ms=5.8, mec="white",
                mew=0.8, lw=2.1, label=r"estimate $\hat d_{\mathrm{MG}}$")
        bx.plot(k.rho, k.pr_pos, "-s", color=TRANSIENT, ms=5.8, mfc="none",
                mew=1.5, lw=2.1, label="effective rank")
        bx.set_xlim(0.47, 1.03)
        bx.set_ylim(1.2, 5.9)
        bx.set_xticks([0.5, 0.75, 1.0])
        bx.set_yticks([2, 3, 4, 5])
        bx.set_xlabel(r"decay $q$ of the phase amplitudes")
        bx.set_ylabel("components")
        bx.legend(loc="lower right", bbox_to_anchor=(1.03, -0.04),
                  fontsize=8.8, labelspacing=0.22, handlelength=1.8,
                  handletextpad=0.4)

        # The shaded band is named here rather than across the curves, and the
        # six ranks are named by colour.
        ax.legend(loc="lower right", fontsize=8.6, ncol=2, handletextpad=0.3,
                  columnspacing=0.7, bbox_to_anchor=(1.04, -0.035),
                  handlelength=1.6, labelspacing=0.20,
                  title="shaded: the working range", title_fontsize=8.6)

        titles(fig, H, [(0.108, r"(a) Estimate against the lag $\tau$"),
                        (0.600, "(b) A dimension, not a rank")], top=0.235)
    save(fig, "p_backup_tau")


def fig_surrogate():
    """The surrogate comparison behind the matched-window claim."""
    s = table("grok.matched.surrogate/surrogate_summary.csv")
    s = s[(s.column == "weight_norm") & (s.smooth == 201)]
    # "s42" and its family were struck out as jargon: a seed is an
    # implementation detail, and what a listener needs is the task and the run.
    order = [("mod_wd1", "modular arithmetic, run 1", True),
             ("mod_wd1_s43", "modular arithmetic, run 2", True),
             ("mod_wd1_s44", "modular arithmetic, run 3", True),
             ("s5_wd1", "product in $S_5$", True),
             ("mod_wd0", "mod. arith., no weight decay", False),
             ("s5_wd0", "$S_5$, no weight decay", False)]

    H = 2.12
    # A deep bottom margin: the axis name is a two-level formula.
    y0, h = rows(H, bottom=0.66, top=0.30)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # The names down the left are the margin: the longest of them is 2.00 in
        # set at 10 pt, and 0.300 of the width is 1.78 in, so a fifth of "mod.
        # arith., no weight decay" was cut off the edge of the page. The axis
        # name is a 3.45 in formula and it still clears the right edge with the
        # axes pushed over, because the axes centre only moves 0.15 in.
        ax = fig.add_axes([0.360, y0, 0.612, h])

        for i, (run, label, groks) in enumerate(order):
            g = s[s.run == run]
            y = len(order) - 1 - i
            c = STOCHASTIC if groks else TRANSIENT
            ax.plot([g.surr_median.min(), g.surr_max.max()], [y, y], "-",
                    color=GREY, lw=8.5, alpha=0.22, solid_capstyle="butt")
            ax.plot(g.surr_median, np.full(len(g), y), "|", color=GREY, ms=12,
                    mew=1.6)
            ax.plot(g.observed.iloc[:1], [y], "o", color=c, ms=10.0,
                    mec="white", mew=1.1, zorder=4)
            ax.text(0.985, y, f"$p={g.p.median():g}$",
                    transform=ax.get_yaxis_transform(), va="center",
                    ha="right", fontsize=9.8, color=c)
        ax.axvline(1.0, color=GREY, lw=1.1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab for _, lab, _ in order][::-1], fontsize=10)
        ax.set_ylim(-0.6, 5.6)
        ax.set_xlim(0.9, 15)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xticklabels(["1", "2", "5", "10"])
        ax.minorticks_off()
        # "How far the estimate fell (log scale)" was struck out as informal and
        # uninformative. The statistic has a definition -- logs.depth() in
        # analysis/logs.py, with the two intervals fixed before any estimate was
        # read -- and it fits on one line, so it is written out.
        ax.set_xlabel(r"$D = \operatorname{med}_{[-3\mathrm{k},\,-1\mathrm{k}]}"
                      r"\hat d_{\mathrm{MG}}\ /\ "
                      r"\min_{[-1\mathrm{k},\,+2\mathrm{k}]}"
                      r"\hat d_{\mathrm{MG}}$,  steps from $t_{\mathrm{gen}}$",
                      fontsize=9.8)
        ax.text(0.0, 1.07, "grey: surrogates of the same log;   circle: "
                "the run", transform=ax.transAxes, ha="left", va="bottom",
                color=GREY, **ANNOT)
    save(fig, "p_backup_surrogate")


if __name__ == "__main__":
    fig_tau()
    fig_surrogate()

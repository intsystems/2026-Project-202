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
        marks = ["o", "s", "^", "v", "D", "X"]
        for r, mk in zip(sorted(d.r.unique()), marks):
            g = d[d.r == r].groupby("span_periods").MG.median().sort_index()
            ax.plot(g.index, g.values / r, "-", marker=mk, color=RECURRENT,
                    ms=4.8, mec="white", mew=0.6, lw=1.6, alpha=0.85,
                    label=f"$r={r}$")
        ax.axhline(1.0, color=GOOD, lw=1.4, ls=(0, (4, 3)))
        ax.axvspan(0.15, 0.45, color=GOOD, alpha=0.13, lw=0, zorder=0)
        ax.text(0.275, 2.6, "working\nrange", ha="center", va="bottom",
                color=GOOD, linespacing=1.25, **ANNOT)
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
        ax.legend(loc="lower right", fontsize=8.8, ncol=2, handletextpad=0.3,
                  columnspacing=0.8, bbox_to_anchor=(1.04, -0.035),
                  handlelength=1.8, labelspacing=0.22)

        # (b) an unequal drive separates the active dimension from the effective
        # rank, and the estimate follows the first
        a = table("valid.anisotropy/aniso_summary.csv")
        k = a[a.r == 4].sort_values("rho")
        bx.axhline(4.0, color=GREY, lw=1.4, ls=(0, (4, 3)))
        bx.text(0.485, 4.12, r"$d_{\mathrm{act}} = 4$", ha="left", va="bottom",
                color=GREY, **ANNOT)
        bx.plot(k.rho, k.MG, "-o", color=RECURRENT, ms=5.8, mec="white",
                mew=0.8, lw=2.1)
        bx.plot(k.rho, k.pr_pos, "-s", color=TRANSIENT, ms=5.8, mfc="none",
                mew=1.5, lw=2.1)
        # Named in place rather than in a legend box: at this panel size any box
        # large enough to read sat on one of the two curves it labelled.
        bx.text(0.60, 4.85, r"estimate $\hat d_{\mathrm{MG}}$", ha="left",
                va="bottom", color=RECURRENT, **ANNOT)
        bx.text(0.66, 2.25, "effective rank", ha="left", va="top",
                color=TRANSIENT, **ANNOT)
        bx.set_xlim(0.47, 1.03)
        bx.set_ylim(1.2, 5.9)
        bx.set_xticks([0.5, 0.75, 1.0])
        bx.set_yticks([2, 3, 4, 5])
        bx.set_xlabel(r"decay $q$ of the phase amplitudes")
        bx.set_ylabel("components")

        titles(fig, H, [(0.108, r"(a) Estimate against the lag $\tau$"),
                        (0.600, "(b) A dimension, not a rank")], top=0.235)
    save(fig, "p_backup_tau")


def fig_surrogate():
    """The surrogate comparison behind the matched-window claim."""
    s = table("grok.matched.surrogate/surrogate_summary.csv")
    s = s[(s.column == "weight_norm") & (s.smooth == 201)]
    order = [("mod_wd1", "mod. arith. s42", True),
             ("mod_wd1_s43", "mod. arith. s43", True),
             ("mod_wd1_s44", "mod. arith. s44", True),
             ("s5_wd1", "$S_5$ product", True),
             ("mod_wd0", "mod. arith., no WD", False),
             ("s5_wd0", "$S_5$, no WD", False)]

    H = 2.12
    y0, h = rows(H, bottom=0.52, top=0.34)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.232, y0, 0.740, h])

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
        ax.set_xlabel("how far the estimate fell (log scale)")
        ax.text(0.0, 1.07, "grey: surrogates of the same log;   circle: "
                "the run", transform=ax.transAxes, ha="left", va="bottom",
                color=GREY, **ANNOT)
    save(fig, "p_backup_surrogate")


if __name__ == "__main__":
    fig_tau()
    fig_surrogate()

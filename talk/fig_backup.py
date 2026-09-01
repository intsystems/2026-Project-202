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
    # Два ряда легенды под фигурой: семь ключей панели (a) и три ключа панели
    # (b) в один ряд не помещаются, а внутри осей они лежали на кривых.
    y0, h = rows(H, bottom=0.90, top=0.26)
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
        ax.axvspan(0.15, 0.45, color=GOOD, alpha=0.15, lw=0, zorder=0,
                   label="the working range")
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
        # Оба ключа под фигурой, каждый своим рядом. Ранги названы одними
        # числами с общим заголовком $r$: шесть ключей вида "$r=1$" в ряд с
        # ключом полосы не помещаются, а шесть цифр помещаются.
        ha_, la_ = ax.get_legend_handles_labels()
        fig.legend(ha_, la_, loc="lower center", ncol=7, fontsize=9.2,
                   bbox_to_anchor=(0.52, 0.135), handlelength=1.7,
                   columnspacing=0.9, handletextpad=0.35)
        hb_, lb_ = bx.get_legend_handles_labels()
        fig.legend(hb_, lb_, loc="lower center", ncol=3, fontsize=9.2,
                   bbox_to_anchor=(0.52, 0.0), handlelength=2.0,
                   columnspacing=1.1, handletextpad=0.4)

        titles(fig, H, [(0.108, r"(a) Estimate against the lag $\tau$"),
                        (0.600, "(b) A dimension, not a rank")], top=0.235)
    save(fig, "p_backup_tau")


def fig_surrogate():
    """The surrogate comparison behind the matched-window claim.

    The grey bar used to run from ``surr_median.min()`` to ``surr_max.max()``:
    its left end was a median and its right end was a maximum, two different
    statistics as the two ends of one interval. Nothing can be read off such a
    bar, and what it actually did was contradict the p values printed beside it.
    A maximum over 195 draws is an extreme order statistic, so the observed value
    lands below it almost by construction, and three of the four generalising
    runs therefore sat *inside* the grey -- "nothing unusual" -- while their own
    p said they were above 95 to 97 per cent of the surrogates.

    The bar is now the 5th to 95th percentile of the pooled draws (39 surrogates
    at each of five seeds, so 195), which is the null distribution the test is
    against, and every circle now lands where its p value says it should: the
    four generalising runs to the right of it, the two undecayed controls at its
    left edge. The p column is unchanged -- it is the per-seed median that
    tab:matched of the article reports.
    """
    raw = table("grok.matched.surrogate/surrogates.csv")
    raw = raw[(raw.column == "weight_norm") & (raw.smooth == 201)]
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

    H = 2.00
    # A deep bottom margin: the axis name is a two-level formula. The top margin
    # holds the legend row that replaced the prose caption. The whole figure is
    # 0.12 in shorter than the deck's standard so the slide can carry a third
    # line: the two controls have no generalisation step of their own, and a
    # listener who notices that needs the answer on the slide.
    y0, h = rows(H, bottom=0.66, top=0.34)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # The names down the left are the margin: the longest of them is 2.00 in
        # set at 10 pt, and 0.300 of the width is 1.78 in, so a fifth of "mod.
        # arith., no weight decay" was cut off the edge of the page. The axis
        # name is a 3.45 in formula and it still clears the right edge with the
        # axes pushed over, because the axes centre only moves 0.15 in.
        ax = fig.add_axes([0.360, y0, 0.612, h])

        for i, (run, label, groks) in enumerate(order):
            g = raw[raw.run == run]
            draws = g[g.kind == "surrogate"].depth.to_numpy(dtype=float)
            draws = draws[np.isfinite(draws)]
            lo, mid, hi = np.percentile(draws, [5, 50, 95])
            observed = float(g[g.kind == "observed"].depth.iloc[0])
            y = len(order) - 1 - i
            c = STOCHASTIC if groks else TRANSIENT
            ax.plot([lo, hi], [y, y], "-", color=FAINT, lw=9.0, alpha=0.85,
                    solid_capstyle="butt",
                    label="surrogates, 5–95 %" if i == 0 else None)
            # A darker tick, not a second shade of the same grey: the band and
            # the median used to be two greys a reviewer could not tell apart.
            ax.plot([mid], [y], "|", color=GREY, ms=13, mew=2.0, zorder=3,
                    label="their median" if i == 0 else None)
            ax.plot([observed], [y], "o", color=c, ms=10.0, mec="white",
                    mew=1.1, zorder=4,
                    label=("the run: generalises" if i == 0 else
                           "no decay" if i == 4 else None))
            ax.text(0.985, y, f"$p={s[s.run == run].p.median():g}$",
                    transform=ax.get_yaxis_transform(), va="center",
                    ha="right", fontsize=9.8, color=c)
        ax.axvline(1.0, color=GREY, lw=1.1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab for _, lab, _ in order][::-1], fontsize=10)
        ax.set_ylim(-0.6, 5.6)
        # The old limit of 15 was set by a maximum that is no longer drawn; the
        # widest band now ends at 4.4, so the axis stops just past it and the
        # whole figure is a factor of three less empty.
        ax.set_xlim(0.93, 7.6)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5])
        ax.set_xticklabels(["1", "2", "3", "5"])
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
        h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=9.4,
                   bbox_to_anchor=(0.52, 0.995), handlelength=1.8,
                   columnspacing=1.1, handletextpad=0.4)
    save(fig, "p_backup_surrogate")


if __name__ == "__main__":
    fig_tau()
    fig_surrogate()

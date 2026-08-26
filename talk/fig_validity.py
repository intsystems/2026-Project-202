"""Slides 4-9: which estimator, what it recovers, and the two regimes whose
numbers may not be read as a dimension.

These four figures exist because an earlier draft of the deck asserted three
things it did not show, and a reader called all three.

* *Why MG and not something else?* It is not an accuracy claim. MG and LB agree
  to 0.02 MAE on this system, and at r <= 20 the article reports LB as the more
  accurate of the two. The estimator that scatters is TwoNN. `fig_estimator`.
* *How can the active dimension of a decaying run "be" one?* It is measured, not
  assumed: at zero Theiler exclusion the estimator returns 1.20 on the transient
  and is unmoved on the torus. `fig_theiler`.
* *Why treat mini-batch noise as a case with a right answer?* It is not one --
  no invariant set, no active dimension. What the noise figure scores is the
  *drive*: 2.5 per cent additive noise on a clean rank-one drive moves the
  estimate from 1 to 11, so the noise is counted rather than filtered. The
  diagonal there is the truth of the clean drive and is labelled as such, never
  as a truth for the noisy arms, which have none. `fig_noise`.

Colour on `fig_estimator` distinguishes estimators, not regimes; everywhere else
the palette carries its usual regime meaning.

Three of the four keep a one-row legend under the figure. Moving it inside the
axes was tried and reverted: at this panel size a three-entry legend covers a
third of the panel, and it landed on the curves it was naming. A 0.16 in strip
along the bottom is the cheaper of the two costs.
"""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, BOX, context, rows, table, titles, save)

HELD = [1, 3, 5, 8]        # the withheld ranks every score below is computed on
SKIP = ["acc_probe", "loss_step"]   # degenerate, and fails the zero-lr check


def digits(arm=None):
    d = table("sys.digits.parameter/sweep_raw.csv")
    d = d[(~d.eta_zero) & (~d.observer.isin(SKIP))]
    return d if arm is None else d[d.arm == arm]


def observer_errors():
    """MAE of MG per scalar observer, on the withheld ranks of the qp arm."""
    g = digits("qp")
    g = g[g.r.isin(HELD)]
    out = []
    for obs, h in g.groupby("observer"):
        m = h.groupby("r")[["traj_PR", "MG"]].median()
        out.append(float((m.MG - m.traj_PR).abs().mean()))
    return np.array(sorted(out))


def strip(fig, ax, ncol, x=0.52):
    """The shared one-row legend along the bottom of a figure."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(x, 0.0), handlelength=2.2, fontsize=10.0)


def fig_estimator():
    """Slide 4: why MG, and how little of the result rests on that choice."""
    d = digits("qp")
    med = d.groupby("r")[["traj_PR", "MG", "LB", "TwoNN", "PRdelay",
                          "specPR0", "roughness"]].median()
    held = med.loc[HELD]

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # The right panel carries words down its y axis, so it needs a wider gap
        # than cols() gives by default.
        ax = fig.add_axes([0.075, y0, 0.375, h])
        bx = fig.add_axes([0.597, y0, 0.375, h])

        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1)
        ax.text(4.9, 3.05, "truth", rotation=31, color=GREY, ha="center",
                va="center", **ANNOT)
        ax.plot(med.traj_PR, med.TwoNN, ":", color=STOCHASTIC, marker="^",
                ms=5.6, mec="white", mew=0.6, lw=1.8, zorder=3, label="TwoNN")
        ax.plot(med.traj_PR, med.LB, "--", color=RECURRENT, marker="s", ms=6.6,
                mfc="none", mew=1.4, lw=1.5, zorder=4, label="LB")
        ax.plot(med.traj_PR, med.MG, "-", color=RECURRENT, marker="o", ms=4.6,
                mec="white", mew=0.7, lw=2.2, zorder=5, label="MG")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 9.8)
        ax.set_ylim(0.9, 17)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yticks([1, 2, 4, 8])
        ax.set_yticklabels(["1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlabel("true trajectory rank $r$")
        ax.set_ylabel("estimate, components")
        ax.legend(loc="upper left", handletextpad=0.5, labelspacing=0.28,
                  handlelength=2.2, bbox_to_anchor=(-0.02, 1.04))
        ax.text(9.7, 0.94, "MG and LB lie\non top of each other", ha="right",
                va="bottom", color=RECURRENT, linespacing=1.25, **ANNOT)

        # (b) every statistic we could compute on the same series, scored the
        # same way. Three of the six need no neighbour search at all, and one of
        # those is within a factor of two of MG.
        stats = [("MG", "MG", RECURRENT), ("LB", "LB", RECURRENT),
                 ("TwoNN", "TwoNN", STOCHASTIC),
                 ("PRdelay", "delay PR", GREY),
                 ("specPR0", "spectral PR", GREY),
                 ("roughness", "roughness", GREY)]
        for i, (col, label, c) in enumerate(stats):
            mae = float((held[col] - held.traj_PR).abs().mean())
            rho = float(held[col].corr(held.traj_PR, method="spearman"))
            y = len(stats) - 1 - i
            bx.barh(y, mae, height=0.62, color=c, alpha=0.75, lw=0)
            bx.text(mae + 0.13, y, rf"{mae:.2f}   $\rho={rho:.2f}$",
                    va="center", ha="left", fontsize=9.6, color=c)
        bx.set_yticks(range(len(stats)))
        bx.set_yticklabels([lab for _, lab, _ in stats][::-1], fontsize=9.8)
        bx.set_xlim(0, 7.6)
        bx.set_xticks([0, 1, 2, 3])
        bx.set_ylim(-0.6, 6.75)
        bx.set_xlabel("mean absolute error, components")
        # The two numbers on every bar need a key, or the rho is decoration.
        bx.text(1.0, 6.60, r"error and rank correlation $\rho$ with the truth",
                ha="right", va="top", color=GREY,
                transform=bx.get_yaxis_transform(), **ANNOT)

        titles(fig, H, [(0.075, "(a) Three estimators, one log"),
                        (0.597, "(b) All six, scored alike")], top=0.235)
    save(fig, "p_estimator")


def fig_recovery():
    """Slide 5: recovery where the estimand exists, and the observer spread.

    The mini-batch arms are deliberately absent. There the diagonal is not the
    truth, because no active dimension is defined at all; they get their own
    slide rather than being scored against a line they cannot meet.
    """
    d = digits()
    series = [("qp", "fast drive", RECURRENT, "-", "o"),
              ("qp_slow", r"same drive, $25\times$ slower", RECURRENT, "--",
               "s")]

    H = 2.16
    y0, h = rows(H, bottom=0.72, top=0.26)      # bottom holds the legend too
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.075, y0, 0.391, h])
        bx = fig.add_axes([0.581, y0, 0.391, h])

        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1)
        # 18 degrees, not 45: the axes are log-log but the decades per inch
        # differ, so a line of slope one is not drawn at slope one.
        ax.text(6.8, 4.8, "truth", rotation=18, color=GREY, ha="center",
                va="center", **ANNOT)
        # A ceiling drawn as a line beats a ceiling pointed at with an arrow:
        # the reader sees where the curves stop rising instead of being told.
        ax.axhline(8.0, color=FAINT, lw=1.3, ls=(0, (2, 2.5)), zorder=1)
        ax.text(9.7, 8.6, r"ceiling $\approx 8$", ha="right", va="bottom",
                color=GREY, **ANNOT)
        for arm, label, c, ls, mk in series:
            g = d[d.arm == arm]
            t = g.groupby("r").traj_PR.median().values
            y = g.groupby("r").MG.median().values
            q1 = g.groupby("r").MG.quantile(0.25).values
            q3 = g.groupby("r").MG.quantile(0.75).values
            o = np.argsort(t)
            ax.fill_between(t[o], q1[o], q3[o], color=c, alpha=0.15, lw=0,
                            zorder=2)
            ax.plot(t, y, ls, color=c, marker=mk, ms=5.4, mec="white", mew=0.8,
                    lw=2.1, label=label, zorder=4, clip_on=False)

        # The transient is one point, not a curve over r. With the drive off, r
        # only picks the direction the run is kicked in before it decays back;
        # nothing is excited in r directions, the construction claims
        # d_act = 1 whatever r is (MODES["gd"] in digits_parameter.py), and all
        # seven values of r land on top of each other. Drawing them as a series
        # would invite exactly the reading that r means something here.
        g = d[d.arm == "gd"]
        ax.plot([g.traj_PR.median()], [g.MG.median()], "D", color=TRANSIENT,
                ms=7.8, mec="white", mew=0.9, zorder=5, clip_on=False,
                label="no drive at all")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 9.8)
        ax.set_ylim(0.8, 44)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yticks([1, 2, 4, 8, 16])
        ax.set_yticklabels(["1", "2", "4", "8", "16"])
        ax.minorticks_off()
        ax.set_xlabel("true trajectory rank $r$")
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}$")
        ax.annotate("no drive:\ntruth 1, estimate 16", xy=(1.16, 16.6),
                    xytext=(2.15, 27.0), color=TRANSIENT, ha="left",
                    va="center", linespacing=1.25,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=TRANSIENT,
                                    shrinkA=1, shrinkB=4), **ANNOT)

        # (b) the same measurement, one point per scalar observer: the spread is
        # a factor of five, so "about half a component" is as much a property of
        # the observer as of the estimator.
        vals = observer_errors()
        bx.plot(vals, np.zeros_like(vals), "o", color=RECURRENT, ms=9.0,
                mec="white", mew=1.1, alpha=0.85, zorder=3, clip_on=False)
        bx.axvline(np.median(vals), color=GREY, lw=1.2, ls=(0, (3, 2.5)),
                   zorder=1)
        bx.text(np.median(vals) + 0.06, 0.52, f"median {np.median(vals):.2f}",
                ha="left", va="bottom", color=GREY, **ANNOT)
        # Named, not just ranked: the supervisor asked which observers these
        # are, and the answer is part of the result -- the best is a fixed
        # projection of the parameters, the worst the full-batch loss.
        for v, side, ha in ((vals[0], "parameter\nprojection", "center"),
                            (vals[-1], "full-batch\nloss", "center")):
            bx.text(v, -0.32, f"{side}\n{v:.2f}", ha=ha, va="top",
                    color=RECURRENT, linespacing=1.25, **ANNOT)
        bx.set_xlim(0.15, 2.45)
        bx.set_ylim(-1.60, 1.15)
        bx.set_xticks([0.5, 1.0, 1.5, 2.0])
        bx.set_yticks([])
        bx.spines["left"].set_visible(False)
        bx.set_xlabel("error of a single observer")

        strip(fig, ax, 3)
        titles(fig, H, [(0.075, "(a) Estimate against truth"),
                        (0.581, "(b) One point per observer")], top=0.235)
    save(fig, "p_recovery")


def fig_theiler():
    """Slide 8: the transient's estimand is one, and it is measured, not assumed."""
    d = table("valid.theiler.contrast/sweep_windows.csv")
    order = ["0", "1", "2", "5", "10", "20", "50", "100", "150", "uncapped"]
    labels = ["0", "2", "10", "50", "150", r"$\infty$"]
    ticks = [0, 2, 4, 6, 8, 9]
    arms = [("fast", "torus", RECURRENT, "-", "o"),
            ("slow", "slow torus", RECURRENT, "--", "s"),
            ("transient", "transient, no drive", TRANSIENT, ":", "D")]
    x = np.arange(len(order))
    frozen = float(d[(d.arm == "transient")
                     & (d.theiler_label == "frozen")].theiler_used.median())
    pos = float(np.interp(frozen, [50, 100], [6, 7]))

    H = 2.16
    y0, h = rows(H, bottom=0.72, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # A wider left margin than the other figures: this y axis is labelled up
        # to 100, and three digits plus the axis name did not fit in 0.075.
        ax = fig.add_axes([0.090, y0, 0.3835, h])
        bx = fig.add_axes([0.5885, y0, 0.3835, h])

        for slot in (ax, bx):
            slot.axvline(pos, color=FAINT, lw=1.3, ls=(0, (2, 2.5)), zorder=0)
        # The rule's label goes in (a), high above every curve: in (b) the only
        # free space next to that line was on top of the slow torus.
        ax.text(pos - 0.35, 400, f"our rule\n$W_T = {frozen:.0f}$", ha="right",
                va="center", color=GREY, linespacing=1.25, **ANNOT)

        ax.axhline(1.0, color=GOOD, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.text(9.40, 1.09, r"truth: $d_{\mathrm{act}} = 1$", ha="right",
                va="bottom", color=GOOD, **ANNOT)
        for arm, label, c, ls, mk in arms:
            g = d[d.arm == arm].groupby("theiler_label")
            ax.plot(x, [g.MG.median()[k] for k in order], ls, color=c,
                    marker=mk, ms=5.4, mec="white", mew=0.7, lw=2.0,
                    label=label, zorder=3)
            bx.plot(x, [g.frac_near_ref.median()[k] for k in order], ls,
                    color=c, marker=mk, ms=5.4, mec="white", mew=0.7, lw=2.0,
                    zorder=3)

        ax.set_yscale("log")
        ax.set_ylim(0.42, 900)
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(["1", "10", "100"])
        ax.minorticks_off()
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}$")
        # Below the truth line, where the panel is empty, with a short arrow to
        # the point. Bare text there read as a label for the green line; text in
        # the gap above the point sat on the slow torus instead. The arrow is
        # what makes the reference unambiguous, so the number gets one.
        ax.annotate("1.20", xy=(0.04, 1.10), xytext=(0.80, 0.56),
                    ha="left", va="center", color=TRANSIENT,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=TRANSIENT,
                                    shrinkA=2, shrinkB=3), **ANNOT)
        ax.text(8.55, 340, "174", ha="center", va="center", color=TRANSIENT,
                **ANNOT)

        bx.set_ylim(-0.08, 1.12)
        bx.set_yticks([0, 0.5, 1.0])
        bx.set_ylabel("fraction of returns")
        bx.text(9.40, 0.07, "not one\nreturn left", ha="right", va="bottom",
                color=TRANSIENT, linespacing=1.25, **ANNOT)

        for slot in (ax, bx):
            slot.set_xlim(-0.45, 9.45)
            slot.set_xticks(ticks)
            slot.set_xticklabels(labels)
            slot.set_xlabel("$W_T$, steps excluded in time")

        strip(fig, ax, 3)
        titles(fig, H, [(0.090, "(a) What the estimator returns"),
                        (0.5885, "(b) Why: nothing comes back")], top=0.235)
    save(fig, "p_theiler")


def fig_noise():
    """Slide 9: noise is not filtered out, it is counted -- and then detected.

    The dashed diagonal is the truth *of the clean drive*, and it is labelled
    that way. The noisy arms have no active dimension to be right or wrong
    about; what the panel shows is that adding 2.5 per cent noise to a drive of
    rank one moves the estimate to 11, which is a statement about the estimator
    and not a score against a line those arms could ever meet.
    """
    d = digits()
    # The synthetic noise-only arm is deliberately absent from panel (a). Its
    # median estimate is 15.11 and the real mini-batch arm's is also 15.11, so
    # the two drew one horizontal line through each other and the marker for the
    # interesting one vanished into the curve for the redundant one. Panel (b)
    # keeps both, and there their agreement is the point rather than the mess:
    # two different stochastic mechanisms, one diagnostic verdict.
    series = [("qp", "clean drive", RECURRENT, "-", "o"),
              ("mixed", r"drive $+\ 2.5\,\%$ noise", STOCHASTIC, "--", "^")]
    ident = [("qp", "drive", RECURRENT, "o"),
             ("qp_slow", "slow drive", RECURRENT, "s"),
             ("mixed", r"$+$ noise", STOCHASTIC, "^"),
             ("noise", "noise alone", STOCHASTIC, "v"),
             ("batch", "mini-batch", STOCHASTIC, "P"),
             ("gd", "transient", TRANSIENT, "D")]

    H = 2.16
    y0, h = rows(H, bottom=0.72, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.075, y0, 0.375, h])
        bx = fig.add_axes([0.600, y0, 0.372, h])

        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1)
        ax.text(5.4, 2.95, "clean-drive truth", rotation=16, color=GREY,
                ha="center", va="center", **ANNOT)
        for arm, label, c, ls, mk in series:
            g = d[d.arm == arm]
            t = g.groupby("r").traj_PR.median().values
            y = g.groupby("r").MG.median().values
            ax.plot(t, y, ls, color=c, marker=mk, ms=5.4, mec="white", mew=0.8,
                    lw=2.1, label=label, zorder=4, clip_on=False)
        # the genuine mini-batch arm: its own effective rank is 6.6 whatever r
        # is, so it is one point rather than a curve
        b = d[d.arm == "batch"]
        ax.plot([b.traj_PR.median()], [b.MG.median()], "P", color=STOCHASTIC,
                ms=9.5, mec="white", mew=1.1, zorder=5, label="real mini-batch")
        ax.annotate("15, whatever $r$ is",
                    xy=(b.traj_PR.median(), b.MG.median()), xytext=(2.05, 48),
                    ha="left", va="center", color=STOCHASTIC, bbox=BOX,
                    zorder=6,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=STOCHASTIC,
                                    shrinkA=2, shrinkB=6), **ANNOT)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 9.8)
        ax.set_ylim(0.8, 80)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
        ax.minorticks_off()
        ax.set_xlabel("rank $r$ of the deterministic drive")
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}$")
        ax.annotate(r"$1 \to 11$", xy=(1.04, 10.2), xytext=(1.40, 3.2),
                    fontsize=10.5, color=STOCHASTIC, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.1, color=STOCHASTIC,
                                    shrinkA=1, shrinkB=4))

        n = len(ident)
        bx.axvspan(0.95, 1.10, color=RECURRENT, alpha=0.14, lw=0, zorder=0)
        # Under the band, not rotated inside it: rotated inside, the word sat on
        # the very markers whose being in the band is the thing it certifies.
        bx.text(1.02, n + 0.98, "valid", ha="center", va="top",
                fontsize=9.8, color=RECURRENT)
        for i, (arm, label, c, mk) in enumerate(ident):
            v = d[d.arm == arm].dropna(subset=["ident_ratio"]).ident_ratio
            y = np.full(len(v), n - 1 - i) + np.linspace(-0.16, 0.16, len(v))
            bx.plot(v.to_numpy(), y, mk, color=c, ms=5.6, mec="white", mew=0.7,
                    ls="none", clip_on=False, zorder=3)
        bx.set_xscale("log")
        bx.set_xlim(0.9, 2.3)
        bx.set_ylim(-0.55, n + 1.05)
        bx.set_xticks([1, 1.5, 2])
        bx.set_xticklabels(["1", "1.5", "2"])
        bx.minorticks_off()
        bx.set_yticks(range(n))
        bx.set_yticklabels([lab for _, lab, _, _ in ident][::-1], fontsize=9.4)
        bx.set_xlabel(r"$\rho_{\mathrm{ident}} = \hat d(2E)\,/\,\hat d(E)$")

        strip(fig, ax, 3, x=0.50)
        titles(fig, H, [(0.075, "(a) Noise is counted, not filtered"),
                        (0.600, "(b) But it is visible")], top=0.235)
    save(fig, "p_noise")


if __name__ == "__main__":
    fig_estimator()
    fig_recovery()
    fig_theiler()
    fig_noise()

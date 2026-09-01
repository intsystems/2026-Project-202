"""Slides 5-11: which estimator, what it recovers, and the two regimes whose
numbers may not be read as a dimension.

These figures exist because an earlier draft of the deck asserted three things it
did not show, and a reader called all three.

* *Why MG and not something else?* It is not an accuracy claim. MG and LB agree
  to 0.02 MAE on this system, and at r <= 20 the article reports LB as the more
  accurate of the two. The estimator that scatters is TwoNN. `fig_estimator`.
* *How can the active dimension of a decaying run "be" one?* It is measured, not
  assumed: at zero Theiler exclusion the estimator returns 1.20 on the transient
  and is unmoved on the torus. `fig_theiler`.
* *Why treat mini-batch noise as a case with a right answer?* It is not one --
  no invariant set, no active dimension. What the noise figure scores is the
  *drive*: 2.5 per cent additive noise on a clean rank-one drive moves the
  estimate from 1 to 11, so the noise is counted rather than filtered.
  `fig_noise`, with the diagnostic that catches it in `fig_ident`.

**No figure here carries prose inside the axes any more.** A reviewer went
through the deck and struck out every in-panel sentence, with two reasons that
are worth keeping written down. First, a reader cannot tell which words are the
data's and which are the speaker's: "MG and LB lie on top of each other" is an
argument, and an argument that sits where a key belongs will be believed as a
measurement. Second, everything those sentences said is either a legend entry or
a line of the slide. So the grey diagonal is now a legend entry reading "ground
truth" rather than a rotated word lying on itself, and the commentary moved under
the figure where it can be disagreed with.

Two panels were split off into figures of their own for the same reason: the ten
observers of `fig_observers` and the diagnostic of `fig_ident` each had one
sentence standing in for a plot, and each is now the plot.
"""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, context, rows, cols, table, titles,
                         strip, save)

HELD = [1, 3, 5, 8]        # the withheld ranks every score below is computed on
SKIP = ["acc_probe", "loss_step"]   # degenerate, and fails the zero-lr check

#: The human names of actdim.observers.REGISTRY, which is where they are defined.
#: A reviewer asked of an unlabelled panel "is that the train or the test loss,
#: and where is the weight norm?" -- a question the panel had no way to answer.
OBSERVER_NAMES = {
    "loss_full": "full-batch loss",
    "loss_probe": "probe-set loss",
    "w_fro": "parameter norm",
    "c_norm": "subspace norm",
    "fn_fro": "function-space norm",
    "g_fro": "gradient norm",
    "g_proj": "gradient projection",
    "c_proj1": "fixed parameter projection",
    "fn_proj1": "function-space projection",
    "margin": "probe margin",
}


def digits(arm=None):
    d = table("sys.digits.parameter/sweep_raw.csv")
    d = d[(~d.eta_zero) & (~d.observer.isin(SKIP))]
    return d if arm is None else d[d.arm == arm]


def observer_errors():
    """MAE of MG per scalar observer, on the withheld ranks of the qp arm."""
    g = digits("qp")
    g = g[g.r.isin(HELD)]
    out = {}
    for obs, h in g.groupby("observer"):
        m = h.groupby("r")[["traj_PR", "MG"]].median()
        out[obs] = float((m.MG - m.traj_PR).abs().mean())
    return out


def fig_estimator():
    """Slide 5: why MG, and how little of the result rests on that choice."""
    d = digits("qp")
    med = d.groupby("r")[["traj_PR", "MG", "LB", "TwoNN", "PRdelay",
                          "specPR0", "roughness"]].median()
    held = med.loc[HELD]

    H = 2.16
    # The bottom margin holds a legend strip now: the four keys used to sit in
    # the top left corner of panel (a), on top of the curves they name.
    y0, h = rows(H, bottom=0.70, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # The right panel carries words down its y axis, so it needs a wider gap.
        ax = fig.add_axes([0.075, y0, 0.375, h])
        bx = fig.add_axes([0.597, y0, 0.375, h])

        # In the legend, not written along itself: named where the estimators are
        # named, which is what was asked for.
        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1, label="ground truth")
        ax.plot(med.traj_PR, med.TwoNN, ":", color=STOCHASTIC, marker="^",
                ms=5.6, mec="white", mew=0.6, lw=1.8, zorder=3, label="TwoNN")
        ax.plot(med.traj_PR, med.LB, "--", color=RECURRENT, marker="s", ms=6.6,
                mfc="none", mew=1.4, lw=1.5, zorder=4, label="LB")
        ax.plot(med.traj_PR, med.MG, "-", color=RECURRENT, marker="o", ms=4.6,
                mec="white", mew=0.7, lw=2.2, zorder=5, label="MG")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 9.8)
        ax.set_ylim(0.9, 20)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yticks([1, 2, 4, 8])
        ax.set_yticklabels(["1", "2", "4", "8"])
        ax.minorticks_off()
        # The truth is q, the number of rationally independent phases with F an
        # embedding -- not the participation ratio, which the construction
        # happens to make equal to q and which is only the check that all q
        # components have resolvable amplitudes (definition memo, sections 8
        # and 10). Declaring PR the ground truth is exactly what that memo
        # forbids.
        ax.set_xlabel(r"$d_{\mathrm{act}}(R)$: phases forced")
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}(\mathcal{W})$")


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
        bx.set_ylim(-0.6, 6.4)
        bx.set_xlabel("mean absolute error, components")

        strip(fig, ax, 4, fontsize=9.6)
        titles(fig, H, [(0.075, "(a) Three estimators, one log"),
                        (0.597, "(b) All six, scored alike")], top=0.235)
    save(fig, "p_estimator")


def fig_recovery():
    """Slide 8: recovery where the estimand exists, and where it is missed.

    Two panels, and the second is not decoration. On a log axis a curve half a
    component above the diagonal and a curve on it are the same curve, so panel
    (a) answers *does the estimate follow q* and cannot answer *by how much does
    it miss* -- which is the whole content of the word "lands" in the slide's
    title. Panel (b) is the same medians minus q on a linear axis, so the miss is
    read off directly. It is also what makes the second arm legible: slow the
    drive by 25 at a fixed window and the estimate stalls near 3.5 whatever q is,
    which in (a) is a line gently bending and in (b) is a fall to -4.5.

    A single panel also left this figure sitting in the middle third of the slide
    with an inch of white on either side, which reads as a missing plot -- and
    the four-entry legend under it was wider than the figure, so its last key was
    cut off at the right edge. Both are fixed by the same change; the legend is
    now measured against the figure width in ``slide_style.strip``.

    The mini-batch arms are deliberately absent. There the diagonal is not the
    truth, because no active dimension is defined at all; they get their own
    slide rather than being scored against a line they cannot meet.
    """
    d = digits()
    series = [("qp", "fast forcing", RECURRENT, "-", "o"),
              ("qp_slow", "slow forcing", RECURRENT, "--", "s")]

    H = 2.16
    y0, h = rows(H, bottom=0.70, top=0.26)      # bottom holds the legend row
    # A wider left margin than the two-panel default: the y label here is
    # \hat d, and at 0.075 the hat itself fell off the left edge of the figure.
    (xa, w), (xb, _) = cols(2, left=0.098, right=0.022, gap=0.108)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([xa, y0, w, h])
        bx = fig.add_axes([xb, y0, w, h])

        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1, label="ground truth")
        # One grey dashed key for both panels: the diagonal in (a) is the zero
        # line in (b). Labelling it twice would put "ground truth" in the legend
        # twice; labelling it in (b) only would leave (a)'s diagonal nameless,
        # which is the thing a reviewer asked to have named.
        bx.axhline(0.0, color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for arm, label, c, ls, mk in series:
            g = d[d.arm == arm]
            m = g.groupby("r")[["traj_PR", "MG"]].median()
            q1 = g.groupby("r").MG.quantile(0.25).values
            q3 = g.groupby("r").MG.quantile(0.75).values
            t, y = m.traj_PR.values, m.MG.values
            o = np.argsort(t)
            ax.fill_between(t[o], q1[o], q3[o], color=c, alpha=0.15, lw=0,
                            zorder=2)
            ax.plot(t, y, ls, color=c, marker=mk, ms=5.4, mec="white", mew=0.8,
                    lw=2.1, label=label, zorder=4, clip_on=False)
            bx.fill_between(t[o], (q1 - t)[o], (q3 - t)[o], color=c, alpha=0.15,
                            lw=0, zorder=2)
            bx.plot(t, y - t, ls, color=c, marker=mk, ms=5.4, mec="white",
                    mew=0.8, lw=2.1, zorder=4, clip_on=False)

        # The transient is one point, not a curve over r. With the drive off, r
        # only picks the direction the run is kicked in before it decays back;
        # nothing is excited in r directions, the construction claims
        # d_act = 1 whatever r is (MODES["gd"] in digits_parameter.py), and all
        # seven values of r land on top of each other. Drawing them as a series
        # would invite exactly the reading that r means something here.
        #
        # It stays out of (b): a residual of +15 on an axis that has to resolve
        # 0.5 would set the scale for the panel and flatten everything in it.
        g = d[d.arm == "gd"]
        ax.plot([g.traj_PR.median()], [g.MG.median()], "D", color=TRANSIENT,
                ms=7.8, mec="white", mew=0.9, zorder=5, clip_on=False,
                label="no forcing")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 24)
        ax.set_yticks([1, 2, 4, 8, 16])
        ax.set_yticklabels(["1", "2", "4", "8", "16"])
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}(\mathcal{W})$")

        bx.set_xscale("log")
        bx.set_ylim(-5.1, 2.4)
        bx.set_yticks([-4, -2, 0, 2])
        bx.set_ylabel(r"$\hat d_{\mathrm{MG}} - d_{\mathrm{act}}$")

        for slot in (ax, bx):
            slot.set_xlim(0.85, 9.8)
            slot.set_xticks([1, 2, 4, 8])
            slot.set_xticklabels(["1", "2", "4", "8"])
            slot.minorticks_off()
            # The truth is q, the number of rationally independent phases with F
            # an embedding -- not the participation ratio, which the
            # construction happens to make equal to q and which is only the
            # check that all q components have resolvable amplitudes
            # (definition memo, sections 8 and 10). Declaring PR the ground
            # truth is exactly what that memo forbids.
            slot.set_xlabel(r"$d_{\mathrm{act}}(R)$: phases forced")

        titles(fig, H, [(xa, "(a) Fast forcing tracks the truth"),
                        (xb, "(b) The miss, in components")], top=0.235)
        strip(fig, ax, 4, fontsize=9.6)
    save(fig, "p_recovery")


def fig_observers():
    """Slide 7: the same measurement, one bar per scalar observer, named.

    The predecessor of this figure was a strip of ten unlabelled dots with the
    best and the worst annotated. A reviewer's questions of it -- is that the
    train or the test loss, and where is the parameter norm -- were unanswerable
    from the panel, and both are answered by writing the names down. The spread
    is a factor of five, so "about half a component" is as much a property of the
    observer as of the estimator.
    """
    err = observer_errors()
    order = sorted(err, key=err.get)

    H = 2.16
    y0, h = rows(H, bottom=0.52, top=0.22)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.295, y0, 0.675, h])

        for i, obs in enumerate(order):
            y = len(order) - 1 - i
            ax.barh(y, err[obs], height=0.68, color=RECURRENT, alpha=0.75, lw=0)
            ax.text(err[obs] + 0.035, y, f"{err[obs]:.2f}", va="center",
                    ha="left", fontsize=8.8, color=RECURRENT)
        # No median line and no legend. Wherever the line went its key landed on
        # the value labels: at the top on the two smallest errors, at the bottom
        # on the two largest. The median is a single number and it is in the
        # slide's own text, which is cheaper than a rule across ten bars.
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([OBSERVER_NAMES[o] for o in order][::-1],
                           fontsize=8.8)
        ax.set_xlim(0, 2.25)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("mean abs. error of $\\hat d_{\\mathrm{MG}}$, components")
    save(fig, "p_observers")


def fig_theiler():
    """Slide 10: the transient's estimand is one, and it is measured, not assumed."""
    d = table("valid.theiler.contrast/sweep_windows.csv")
    order = ["0", "1", "2", "5", "10", "20", "50", "100", "150", "uncapped"]
    labels = ["0", "2", "10", "50", "150", r"$\infty$"]
    ticks = [0, 2, 4, 6, 8, 9]
    # Один словарь на всю колоду: те же три режима названы так же, как в
    # `fig_recovery`. "slow torus" было названо странным, и справедливо -- тор не
    # медленный, медленный привод.
    arms = [("fast", "fast forcing", RECURRENT, "-", "o"),
            ("slow", "slow forcing", RECURRENT, "--", "s"),
            # Just "transient": beside two drives the word is unambiguous, and
            # the four keys plus "the path itself: dimension 1" do not fit the
            # figure with anything longer.
            ("transient", "transient", TRANSIENT, ":", "D")]
    x = np.arange(len(order))
    frozen = float(d[(d.arm == "transient")
                     & (d.theiler_label == "frozen")].theiler_used.median())
    pos = float(np.interp(frozen, [50, 100], [6, 7]))

    H = 2.16
    y0, h = rows(H, bottom=0.70, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        # A wider left margin than the other figures: this y axis is labelled up
        # to 100, and three digits plus the axis name did not fit in 0.075 --
        # nor in 0.090, where the hat over the d fell off the left edge.
        ax = fig.add_axes([0.104, y0, 0.3765, h])
        bx = fig.add_axes([0.5955, y0, 0.3765, h])

        for slot in (ax, bx):
            slot.axvline(pos, color=FAINT, lw=1.3, ls=(0, (2, 2.5)), zorder=0)
        # A symbol and a number where the line is, not a sentence about it.
        ax.text(pos - 0.25, 430, f"$W_T = {frozen:.0f}$", ha="right",
                va="center", color=GREY, **ANNOT)

        ax.axhline(1.0, color=GOOD, lw=1.5, ls=(0, (4, 3)), zorder=1,
                   label="the path itself: dimension 1")
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
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}(\mathcal{W})$")
        # The two bare readings that used to be annotated in this panel -- 1.20
        # at no exclusion, 174 at unbounded exclusion -- are gone: "не вижу
        # смысла числа слева подписывать". They were the last numbers written
        # inside an axes anywhere in the deck, and both of them are in the
        # slide's own two lines already, where a reader can see them next to what
        # they mean. The y axis is logarithmic and labelled 1, 10, 100, so the
        # panel still says that the gold curve starts at one and ends above a
        # hundred; the annotation only repeated it in smaller type.

        bx.set_ylim(-0.08, 1.12)
        bx.set_yticks([0, 0.5, 1.0])
        bx.set_ylabel("fraction of returns")

        for slot in (ax, bx):
            slot.set_xlim(-0.45, 9.45)
            slot.set_xticks(ticks)
            slot.set_xticklabels(labels)
            slot.set_xlabel("$W_T$, steps excluded in time")

        strip(fig, ax, 4, fontsize=9.0)
        titles(fig, H, [(0.104, "(a) Only the transient moves"),
                        (0.5955, "(b) The transient never returns")],
               top=0.235)
    save(fig, "p_theiler")


def fig_noise():
    """Slide 11a: noise is not filtered out, it is counted.

    The dashed diagonal is the truth *of the clean drive*, and the legend says
    so. The noisy arms have no active dimension to be right or wrong about; what
    the panel shows is that adding 2.5 per cent noise to a drive of rank one
    moves the estimate to 11, which is a statement about the estimator and not a
    score against a line those arms could ever meet.

    The synthetic noise-only arm is deliberately absent. Its median estimate is
    15.11 and the real mini-batch arm's is also 15.11, so the two drew one
    horizontal line through each other and the marker for the interesting one
    vanished into the curve for the redundant one. Both appear in `fig_ident`,
    where their agreement is the point rather than the mess.
    """
    d = digits()
    series = [("qp", "clean drive", RECURRENT, "-", "o"),
              ("mixed", r"$+\ 2.5\,\%$ noise", STOCHASTIC, "--", "^")]

    H = 2.16
    y0, h = rows(H, bottom=0.70, top=0.26)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.215, y0, 0.560, h])

        ax.plot([0.9, 8.6], [0.9, 8.6], color=GREY, lw=1.2, ls=(0, (4, 3)),
                zorder=1, label="clean-drive truth")
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
                ms=9.5, mec="white", mew=1.1, zorder=5,
                label="real mini-batch")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 9.8)
        ax.set_ylim(0.8, 34)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
        ax.minorticks_off()
        ax.set_xlabel(r"$q$, phases of the deterministic drive")
        ax.set_ylabel(r"$\hat d_{\mathrm{MG}}(\mathcal{W})$")

        strip(fig, ax, 4, fontsize=9.4)
    save(fig, "p_noise")


def fig_ident():
    """Slide 11b: the diagnostic that separates the regimes.

    Six arms, one identifiability ratio each. Two different noise mechanisms --
    Gaussian forcing of the update and genuine mini-batch sampling -- land on the
    same verdict, which is the reason to trust the diagnostic rather than a
    coincidence to hide.
    """
    d = digits()
    arms = [("qp", "fast forcing", RECURRENT, "o"),
            ("qp_slow", "slow forcing", RECURRENT, "s"),
            ("mixed", r"drive $+\ 2.5\,\%$ noise", STOCHASTIC, "^"),
            ("noise", "rank-$r$ Gaussian forcing", STOCHASTIC, "v"),
            ("batch", "real mini-batch descent", STOCHASTIC, "P"),
            ("gd", "full-batch transient", TRANSIENT, "D")]

    H = 2.16
    y0, h = rows(H, bottom=0.70, top=0.22)
    with context():
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.320, y0, 0.650, h])

        n = len(arms)
        ax.axvspan(0.95, 1.10, color=RECURRENT, alpha=0.14, lw=0, zorder=0,
                   label=r"$\rho_{\mathrm{ident}} \approx 1$: readable")
        for i, (arm, label, c, mk) in enumerate(arms):
            v = d[d.arm == arm].dropna(subset=["ident_ratio"]).ident_ratio
            y = np.full(len(v), n - 1 - i) + np.linspace(-0.16, 0.16, len(v))
            ax.plot(v.to_numpy(), y, mk, color=c, ms=5.6, mec="white", mew=0.7,
                    ls="none", clip_on=False, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 2.3)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([1, 1.5, 2])
        ax.set_xticklabels(["1", "1.5", "2"])
        ax.minorticks_off()
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for _, lab, _, _ in arms][::-1], fontsize=9.2)
        ax.set_xlabel(r"$\rho_{\mathrm{ident}} = "
                      r"\hat d_{\mathrm{MG}}(2E)\,/\,\hat d_{\mathrm{MG}}(E)$")
        strip(fig, ax, 1, fontsize=9.6)
    save(fig, "p_ident")


if __name__ == "__main__":
    fig_estimator()
    fig_recovery()
    fig_observers()
    fig_theiler()
    fig_noise()
    fig_ident()

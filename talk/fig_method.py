"""Slides 4 and 22: the pipeline, and the shapes the neighbour search sees.

`fig_pipeline` was rebuilt after a reviewer asked whether panels (a) and (b) were
"correctly matched". They were not, and for a worse reason than he had in mind.

``curve_series.csv`` is decimated by eight for drawing (``SERIES_STRIDE`` in
experiments/curves.py: ten thousand points at 1.5 inches wide is ink, not a
curve). The old panels (b) and (c) rebuilt the delay embedding *from that
decimated series* with ``tau = 4``, so the lag they drew was 32 optimiser steps
where the estimator's is 4. The picture was a delay plane of a different lag than
the one every number in the deck was computed at.

Panel (b) now reads ``curve_shapes.csv`` instead: the estimator's own
reconstruction of the same record, with its own tau, written out by the
experiment. Nothing is recomputed here, so nothing can disagree. Both panels are
the qp arm at rank 1, seed 0, observer ``w_fro`` -- the one (arm, rank) pair that
both tables hold, which is why the figure moved from rank 3 to rank 1.

The cost is that the lag itself is no longer illustrable: a lag of four steps
cannot be drawn on a series sampled every eighth step, and the raw record is not
shipped. The delay vector is therefore written as a formula on the method slide,
which is where a reviewer also asked for it.
"""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, BOX, context, rows, table, strip, save)

# The frozen eight-direction configuration, from actdim.frozen.eight_direction().
# Quoted rather than recomputed so the figure cannot drift from the estimator.
TAU = 4          # delay lag, optimiser steps
M_NEIGH = 20     # k_neighbors
THEILER = 76     # (E - 1) * tau at E = 20, under the cap of 150
WINDOW = 8000    # samples per window; the record is 10000
SHAPE_THIN = 5   # curve_shapes keeps every fifth reconstructed point


def fig_pipeline():
    """One scalar log -> the estimator's delay plane -> the neighbour statistic."""
    ser = table("valid.curves/curve_series.csv")
    g = ser[ser.r == 1].sort_values("sample")
    step, z = g["sample"].to_numpy(), g.z.to_numpy()

    shp = table("valid.curves/curve_shapes.csv")
    p = shp[(shp.arm == "qp") & (shp.r == 1)]
    px, py = p.x.to_numpy(), p.y.to_numpy()

    H = 2.16
    y0, h = rows(H, bottom=0.72, top=0.26)   # bottom holds the legend row too
    with context():
        # Explicit axes rather than subplots + tight_layout: two of these three
        # panels must be square (a closed loop drawn on a stretched axis is an
        # ellipse and misreads as anisotropy), and an equal-aspect axis makes
        # tight_layout place each title against the axes box rather than the
        # slot, so the last two collided.
        side = h * H / FULL                       # square in inches
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.078, y0, 0.3294, h])
        bx = fig.add_axes([0.4894, y0, side, h])
        cx = fig.add_axes([0.7567, y0, side, h])
        for slot, name in ((ax, "(a) The log, first 400 steps"),
                           (bx, "(b) Delay plane"),
                           (cx, "(c) Neighbours")):
            box = slot.get_position()
            fig.text(0.5 * (box.x0 + box.x1), (H - 0.225) / H, name,
                     ha="center", va="bottom", fontsize=10.2)

        # A zoom, and the title says so. The whole record at this width is a
        # solid block of ink: at rank one the log turns over every few samples,
        # and 10000 steps across 2 in resolves nothing.
        cut = step <= 400
        ax.plot(step[cut], z[cut], "-", color=RECURRENT, lw=1.5, alpha=0.9)
        ax.plot(step[cut], z[cut], ".", color=RECURRENT, ms=3.4, mew=0)
        ax.set_xlabel("optimiser step $t$")
        ax.set_ylabel("$x_t$")
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 200, 400])
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-2.6, 2.6)

        # (b) the estimator's own reconstruction, first two delay coordinates.
        # The rows are in time order at a constant stride, so the Theiler
        # exclusion is expressible in rows: 76 steps is 76 / 5 rows.
        Y = np.column_stack([px, py])
        i = int(0.37 * len(Y))
        d = np.linalg.norm(Y - Y[i], axis=1)
        near = np.abs(np.arange(len(Y)) - i) <= THEILER // SHAPE_THIN
        order = np.argsort(np.where(~near, d, np.inf))[:M_NEIGH]

        bx.plot(px, py, ".", color=RECURRENT, ms=1.6, alpha=0.55, mew=0)
        # The excluded samples are shown here and not in (c): at rank one the
        # loop's period is about 25 steps, so the 76 steps either side of the
        # reference wrap the loop three times and land all over it. Exactly one
        # of them falls inside (c)'s zoom, which would have made the key there a
        # promise the panel could not keep -- and their being spread out is the
        # recurrent regime's whole property.
        bx.plot(Y[near, 0], Y[near, 1], ".", color=TRANSIENT, ms=4.6, mew=0,
                zorder=4, ls="none", label=r"excluded: $|\Delta t| \leq W_T$")
        bx.plot([Y[i, 0]], [Y[i, 1]], "o", color=STOCHASTIC, ms=6.5,
                mec="white", mew=0.9, zorder=5, ls="none",
                label="reference point")
        bx.set_xlabel("$x_t$")
        bx.set_ylabel(r"$x_{t-\tau}$")
        lim = 1.12 * max(np.abs(px).max(), np.abs(py).max())
        bx.set_xlim(-lim, lim)
        bx.set_ylim(-lim, lim)
        bx.set_xticks([-2, 0, 2])
        bx.set_yticks([-2, 0, 2])

        # (c) the same reference point, zoomed to its neighbourhood
        cx.plot(Y[~near, 0], Y[~near, 1], ".", color=FAINT, ms=1.8, mew=0)
        rm = d[order].max()
        cx.add_patch(Circle((Y[i, 0], Y[i, 1]), rm, fill=False, lw=1.2,
                            ec=RECURRENT, ls=(0, (3, 2)), zorder=4))
        for j in order:
            cx.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], "-",
                    color=RECURRENT, lw=0.8, alpha=0.8, zorder=4)
        cx.plot(Y[order, 0], Y[order, 1], "o", color=RECURRENT, ms=3.6,
                mec="white", mew=0.5, zorder=5, ls="none",
                label="the $m = 20$ nearest")
        cx.plot([Y[i, 0]], [Y[i, 1]], "o", color=STOCHASTIC, ms=6.5,
                mec="white", mew=0.9, zorder=7, ls="none")
        pad = 1.5 * rm
        cx.set_xlim(Y[i, 0] - pad, Y[i, 0] + pad)
        cx.set_ylim(Y[i, 1] - pad, Y[i, 1] + pad)
        cx.set_xticks([])
        cx.set_yticks([])
        # The power law as this panel's axis name: it is what the panel is for,
        # and it costs no row of its own.
        cx.set_xlabel(r"$m(r) \propto r^{\,d}$")
        # A symbol with a leader, not a sentence: the radius has to be named
        # somewhere and the formula under the panel uses it.
        cx.annotate("$r_m$", xy=(Y[i, 0] - 0.68 * rm, Y[i, 1] + 0.68 * rm),
                    xytext=(Y[i, 0] - 1.36 * rm, Y[i, 1] + 1.18 * rm),
                    color=RECURRENT, fontsize=11, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=RECURRENT,
                                    shrinkA=1, shrinkB=1))

        h1, l1 = bx.get_legend_handles_labels()
        h2, l2 = cx.get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.52, 0.0), handlelength=1.6,
                   fontsize=9.4, columnspacing=1.1, handletextpad=0.4)
    save(fig, "p_pipeline")


def fig_shapes():
    """The set the neighbour search actually measures, one panel per regime."""
    s = table("valid.curves/curve_shapes.csv")
    # The middle row is the *estimand*, not "the truth": two of these four
    # regimes have none. A non-recurrent transient samples no invariant
    # occupation measure, so it has no A_R to recover -- the curve's own
    # dimension of 1 is a different quantity and must not be written here as
    # though it were d_act(R). A stationary mini-batch run may well have an
    # occupation dimension, but it counts the directions batch noise excites, so
    # "undefined" was also wrong. Both wordings follow
    # icomp_v2/active_components_definition.md, sections 5, 11 and 13.
    # The first two captions used to read "closed loop" and "filled torus", and
    # they were measured out of the figure. In two delay coordinates the 2-torus
    # is an annulus, and its transverse width here is 7.1 per cent of the radius
    # against the loop's 5.1: the two panels differ by two per cent of a radius.
    # "Filled torus" next to "closed loop" therefore promised a difference the
    # pixels do not carry, and a listener comparing the panels would have been
    # right to say they are the same ring. The captions now name what was driven,
    # which is the thing that actually differs, and the panels make the honest
    # point instead: at one and at two components the projection looks the same,
    # which is why the number has to be estimated rather than seen.
    order = [("qp", 1, "one phase\ndriven",
              r"$d_{\mathrm{act}}(R) = 1$", RECURRENT, True),
             ("qp", 2, "two phases\ndriven",
              r"$d_{\mathrm{act}}(R) = 2$", RECURRENT, True),
             ("gd", 1, "transient:\ntraversed once",
              "no invariant measure", TRANSIENT, False),
             ("batch_proj", 5, "batch noise\nalone",
              "noise counted too", STOCHASTIC, False)]
    reported = [r"$\hat d_{\mathrm{MG}} = 1.2$", r"$\hat d_{\mathrm{MG}} = 2.2$",
                r"$\hat d_{\mathrm{MG}} \approx 16$",
                r"$\hat d_{\mathrm{MG}} \approx 15$"]

    H = 2.12
    # Four rows of type share this figure with four square panels, and the
    # spacing is arithmetic rather than guesswork because the guesses collided:
    # the estimator's answer and the caption under it were set from the panel
    # position and overlapped by the height of one line. Distances in inches from
    # the bottom edge, then converted once.
    side = 1.05 / FULL                       # square panels, 1.05 in a side
    y0 = 0.61 / H                            # clears the two rows of numbers
    step = side + 0.0813
    with context():
        fig = plt.figure(figsize=(FULL, H))
        for j, ((arm, r, name, truth, c, ok), rep) in enumerate(zip(order,
                                                                    reported)):
            ax = fig.add_axes([0.030 + j * step, y0, side, side * FULL / H])
            g = s[(s.arm == arm) & (s.r == r)]
            ax.plot(g.x, g.y, ".", color=c, ms=2.4, alpha=0.55, mew=0)
            # Equal aspect with the box fixed: a closed loop drawn on a stretched
            # axis comes out an ellipse, and this figure is entirely about shape.
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ("left", "bottom"):
                ax.spines[spine].set_visible(False)
            box = ax.get_position()
            mid = 0.5 * (box.x0 + box.x1)
            fig.text(mid, 1.72 / H, name, ha="center", va="bottom",
                     fontsize=10.2, color=c, linespacing=1.25)
            fig.text(mid, 0.205, truth, ha="center", va="bottom", fontsize=9.0,
                     color=GREY)
            fig.text(mid, 0.100, rep, ha="center", va="bottom", fontsize=11,
                     color=GOOD if ok else STOCHASTIC,
                     fontweight="normal" if ok else "bold")
        fig.text(0.5, 0.012, "grey: the estimand.   Coloured: the "
                 "estimate.", ha="center", va="bottom", color=GREY,
                 fontsize=9.6)
    save(fig, "p_shapes")


if __name__ == "__main__":
    fig_pipeline()
    fig_shapes()

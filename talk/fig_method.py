"""Slides 3 and 20: the pipeline, and the shapes the neighbour search sees."""
import sys
sys.path.insert(0, "talk")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from slide_style import (FULL, RECURRENT, STOCHASTIC, TRANSIENT, GOOD, GREY,
                         FAINT, ANNOT, BOX, context, rows, table, save)

TAU = 4            # the frozen delay lag
THEILER = 76       # the exclusion used on this system


def fig_pipeline():
    """One scalar -> delay coordinates -> a neighbour statistic.

    The three panels are one sentence read left to right, and the sentence only
    closes if the reader believes that panel (b) is built out of panel (a). One
    moment of the log is therefore marked in rose in both: those two samples of
    (a), a lag of 2 tau apart, are the two coordinates of that one dot in (b).
    Without the mark the audience is asked to accept on faith that a squiggle
    turns into a cloud.

    What the panels cannot show is the arithmetic done on the ten distances, and
    that arithmetic is the whole method, so it is written once along the bottom.
    """
    ser = table("valid.curves/curve_series.csv")
    z = ser[ser.r == 3].sort_values("sample").z.to_numpy()

    H = 2.16
    y0, h = rows(H, bottom=0.64, top=0.26)
    with context():
        # Explicit axes rather than subplots + tight_layout: two of these three
        # panels must be square (a circle drawn on a stretched axis is an ellipse
        # and misreads as anisotropy), and an equal-aspect axis makes
        # tight_layout place each title against the axes box rather than the
        # slot, so the last two collided.
        side = h * H / FULL                       # square in inches
        fig = plt.figure(figsize=(FULL, H))
        ax = fig.add_axes([0.078, y0, 0.3390, h])
        bx = fig.add_axes([0.4890, y0, side, h])
        cx = fig.add_axes([0.7457, y0, side, h])
        for slot, name in ((ax, "(a) One scalar log"),
                           (bx, "(b) Delay plane"),
                           (cx, "(c) Neighbours")):
            box = slot.get_position()
            fig.text(0.5 * (box.x0 + box.x1), (H - 0.225) / H, name,
                     ha="center", va="bottom", fontsize=10.2)

        n = 64
        ax.plot(np.arange(n), z[:n], "-", color=RECURRENT, lw=1.4, alpha=0.85)
        ax.plot(np.arange(n), z[:n], ".", color=RECURRENT, ms=3.8, mew=0)
        ax.set_xlabel("training step $t$")
        ax.set_ylabel("$x_t$")
        ax.set_xticks([0, 20, 40, 60])
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-2.8, 3.5)

        mark = 40
        used = (mark - 2 * TAU, mark)
        ax.plot(list(used), z[list(used)], "o", color=STOCHASTIC, ms=6.8,
                mec="white", mew=0.9, zorder=5)
        ax.annotate("", xy=(used[1], z[used[1]]), xytext=(used[0], z[used[0]]),
                    arrowprops=dict(arrowstyle="-", lw=1.3, color=STOCHASTIC,
                                    ls=(0, (2, 1.6)), shrinkA=4, shrinkB=4))
        ax.text(mark - TAU, 3.35, r"lag $2\tau$", color=STOCHASTIC,
                ha="center", va="top", fontsize=10.5)

        # delay plane, the whole record
        bx.plot(z[: -2 * TAU], z[2 * TAU:], ".", color=RECURRENT, ms=1.5,
                alpha=0.55, mew=0)
        bx.plot([z[used[0]]], [z[used[1]]], "o", color=STOCHASTIC, ms=7.8,
                mec="white", mew=1.0, zorder=5)
        bx.annotate("those two samples,\none point here",
                    xy=(z[used[0]], z[used[1]]), xytext=(-2.80, -2.85),
                    color=STOCHASTIC, ha="left", va="bottom", linespacing=1.25,
                    bbox=BOX, zorder=6,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=STOCHASTIC,
                                    shrinkA=2, shrinkB=5), **ANNOT)
        bx.set_xlabel(r"$x_{t-2\tau}$")
        bx.set_ylabel("$x_t$")
        bx.set_xlim(-3.0, 3.0)
        bx.set_ylim(-3.0, 3.0)
        bx.set_xticks([-2, 0, 2])
        bx.set_yticks([-2, 0, 2])

        # the neighbour construction, on the same plane
        Y = np.column_stack([z[2 * TAU:], z[TAU: -TAU], z[: -2 * TAU]])
        i = 700
        d = np.linalg.norm(Y - Y[i], axis=1)
        t = np.arange(len(Y))
        near_in_time = np.abs(t - i) <= THEILER
        allowed = ~near_in_time
        order = np.argsort(np.where(allowed, d, np.inf))[:10]

        cx.plot(Y[allowed, 2], Y[allowed, 0], ".", color=FAINT, ms=1.8, mew=0)
        cx.plot(Y[near_in_time, 2], Y[near_in_time, 0], ".", color=TRANSIENT,
                ms=3.6, mew=0, zorder=3)
        rm = d[order].max()
        cx.add_patch(Circle((Y[i, 2], Y[i, 0]), rm, fill=False, lw=1.2,
                            ec=RECURRENT, ls=(0, (3, 2)), zorder=4))
        for j in order:
            cx.plot([Y[i, 2], Y[j, 2]], [Y[i, 0], Y[j, 0]], "-",
                    color=RECURRENT, lw=1.0, alpha=0.85, zorder=4)
        cx.plot(Y[order, 2], Y[order, 0], "o", color=RECURRENT, ms=4.0,
                mec="white", mew=0.7, zorder=5)
        cx.plot([Y[i, 2]], [Y[i, 0]], "o", color=STOCHASTIC, ms=6.8,
                mec="white", mew=0.9, zorder=6)
        pad = 1.55 * rm
        cx.set_xlim(Y[i, 2] - pad, Y[i, 2] + pad)
        cx.set_ylim(Y[i, 0] - pad, Y[i, 0] + pad)
        cx.set_xticks([])
        cx.set_yticks([])
        cx.set_xlabel(r"$x_{t-2\tau}$, zoom of (b)")
        cx.set_ylabel("$x_t$")
        cx.annotate("$r_m$", xy=(Y[i, 2] - 0.68 * rm, Y[i, 0] + 0.68 * rm),
                    xytext=(Y[i, 2] - 1.38 * rm, Y[i, 0] + 1.24 * rm),
                    color=RECURRENT, fontsize=11, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=RECURRENT,
                                    shrinkA=1, shrinkB=1))
        cx.text(0.02, 0.02, "gold: excluded\nby $W_T$",
                transform=cx.transAxes, color=TRANSIENT, ha="left",
                va="bottom", linespacing=1.25, bbox=BOX, zorder=7, **ANNOT)

        fig.text(0.5, 0.030, r"$\hat d$ comes from how fast the neighbour "
                 r"radius $r_m$ grows with $m$", ha="center", va="bottom",
                 color=RECURRENT, **ANNOT)
    save(fig, "p_pipeline")


def fig_shapes():
    """The set the neighbour search actually measures, one panel per regime."""
    s = table("valid.curves/curve_shapes.csv")
    order = [("qp", 1, "one phase:\nclosed loop", r"$d_{\mathrm{act}} = 1$",
              RECURRENT, True),
             ("qp", 2, "two phases:\nfilled torus", r"$d_{\mathrm{act}} = 2$",
              RECURRENT, True),
             ("gd", 1, "transient:\ntraversed once", r"$d_{\mathrm{act}} = 1$",
              TRANSIENT, False),
             ("batch_proj", 5, "mini-batch noise:\nno structure",
              r"$d_{\mathrm{act}}$ undefined", STOCHASTIC, False)]
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
            fig.text(mid, 0.205, truth, ha="center", va="bottom", fontsize=9.8,
                     color=GREY)
            fig.text(mid, 0.100, rep, ha="center", va="bottom", fontsize=11,
                     color=GOOD if ok else STOCHASTIC,
                     fontweight="normal" if ok else "bold")
        fig.text(0.5, 0.012, "grey: the truth.   Coloured: what the estimator "
                 "returned.", ha="center", va="bottom", color=GREY,
                 fontsize=9.6)
    save(fig, "p_shapes")


if __name__ == "__main__":
    fig_pipeline()
    fig_shapes()

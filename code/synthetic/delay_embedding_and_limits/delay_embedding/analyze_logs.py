"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from actdim.estimator.embedding import resolve_theiler
from synthetic.digits_validation.estimator_calibration.settings import eight_direction
from experiment_runner.plotting import FAINT, GREY, INK, POINTER, RECURRENT, STOCHASTIC, TRANSIENT, WIDTH
from experiment_runner.plotting import draw


def fig_method():
    """The three steps of the estimator on one recurrent log of known active dimension."""
    # Read off the estimator rather than written down here, so that a picture of the method
    # cannot drift from the method. Under theiler="embedding" the rule uses the
    # configuration alone, so the empty series it is handed is never looked at.
    cfg = eight_direction()
    tau = int(cfg.tau)
    exclusion = resolve_theiler(cfg, np.empty(0, dtype=float), tau)

    series = pd.read_csv(HERE / 'logs/curve_series.csv')
    log = series[(series.r == 1) & (series["sample"] <= 400)].sort_values("sample")

    # The estimator's own reconstruction of the same record, written out by the experiment
    # at the lag every number in the paper was computed at. Rebuilding it here from the
    # decimated series would draw a delay plane of some other lag.
    cloud = pd.read_csv(HERE / 'logs/curve_shapes.csv')
    cloud = cloud[(cloud.arm == "qp") & (cloud.r == 1)]
    points = np.column_stack([cloud.x.to_numpy(), cloud.y.to_numpy()])
    # One window's reconstruction, thinned to about 1400 rows in time order at a constant
    # stride, so the exclusion converts into a number of rows.
    thin = max(1, round((cfg.window - (cfg.max_E - 1) * tau) / len(points)))

    centre = int(0.37 * len(points))
    distance = np.linalg.norm(points - points[centre], axis=1)
    excluded = np.abs(np.arange(len(points)) - centre) <= exclusion // thin
    nearest = np.argsort(np.where(excluded, np.inf, distance))[:cfg.k_neighbors]
    radius = float(distance[nearest].max())

    # Placed by hand rather than by tight_layout: two of the three panels must be square,
    # or a closed loop drawn on a stretched axis becomes an ellipse and reads as
    # anisotropy, and tight_layout sets a title against an equal-aspect axes box rather
    # than against the slot, which puts the three titles on three different lines.
    height, foot, head = 1.68, 0.58, 0.18
    y0, h = foot / height, (height - foot - head) / height
    side = (height - foot - head) / WIDTH        # a square, in fractions of the width

    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0.072, y0, 0.312, h])
    bx = fig.add_axes([0.497, y0, side, h])
    cx = fig.add_axes([0.769, y0, side, h])
    for slot, name in ((ax, "(a) the log, first 400 steps"),
                       (bx, "(b) the delay plane"), (cx, "(c) one neighbourhood")):
        box = slot.get_position()
        fig.text(0.5 * (box.x0 + box.x1), (height - head + 0.025) / height, name,
                 ha="center", va="bottom", fontsize=9, color=INK)

    # A zoom, and the title of the panel it is drawn in says the whole record is longer.
    # At rank one the log turns over every few samples, so ten thousand of them across an
    # inch and a half resolve nothing at all.
    ax.plot(log["sample"].to_numpy(), log.z.to_numpy(), "-", color=RECURRENT, lw=0.8)
    ax.set_xlabel("optimiser step $t$", labelpad=1.5)
    ax.set_ylabel("$x_t$")
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 200, 400])
    ax.set_ylim(-2.8, 2.8)
    ax.set_yticks([-2, 0, 2])

    # The cloud is the data and carries the regime's colour; the exclusion is a mark on
    # it and carries the neutral one. Drawn the other way round, as it was at first, the
    # loop reads as though every sample on it had been excluded.
    bx.plot(points[:, 0], points[:, 1], ".", color=RECURRENT, ms=1.1, mew=0, alpha=0.45,
            zorder=2)
    # The excluded samples belong in (b) and not in (c): the record wraps the loop many
    # times over, so the seventy-six steps either side of the reference land all over it --
    # thirty-one points whose largest angular gap around the loop is 26 degrees -- and only
    # one of them falls inside (c)'s zoom. Their being spread out is the recurrent regime's
    # whole property. How many steps one turn takes is deliberately not claimed: both files
    # this figure reads are decimated above the rate the log turns over at, so a period
    # measured from either is an alias.
    bx.plot(points[excluded, 0], points[excluded, 1], ".", color=GREY, ms=3.6, mew=0,
            ls="none", zorder=4)
    bx.plot([points[centre, 0]], [points[centre, 1]], "o", mfc="white", mec=RECURRENT,
            mew=1.3, ms=5.4, ls="none", zorder=5)
    bx.set_xlabel("$x_t$", labelpad=1.5)
    bx.set_ylabel(r"$x_{t-\tau}$")
    span = 1.12 * np.abs(points).max()
    bx.set_xlim(-span, span)
    bx.set_ylim(-span, span)
    bx.set_xticks([-2, 0, 2])
    bx.set_yticks([-2, 0, 2])

    cx.plot(points[~excluded, 0], points[~excluded, 1], ".", color=FAINT, ms=1.3, mew=0,
            zorder=2)
    cx.add_patch(Circle(tuple(points[centre]), radius, fill=False, lw=0.8, ec=RECURRENT,
                        ls=(0, (3, 2)), zorder=4))
    for j in nearest:
        cx.plot([points[centre, 0], points[j, 0]], [points[centre, 1], points[j, 1]], "-",
                color=RECURRENT, lw=0.5, alpha=0.8, zorder=4)
    cx.plot(points[nearest, 0], points[nearest, 1], ".", color=RECURRENT, ms=3.4, mew=0,
            ls="none", zorder=5)
    cx.plot([points[centre, 0]], [points[centre, 1]], "o", mfc="white", mec=RECURRENT,
            mew=1.3, ms=5.4, ls="none", zorder=6)
    pad = 1.5 * radius
    cx.set_xlim(points[centre, 0] - pad, points[centre, 0] + pad)
    cx.set_ylim(points[centre, 1] - pad, points[centre, 1] + pad)
    cx.set_xticks([])
    cx.set_yticks([])
    # The power law as the panel's axis name: it is what the panel is for and it costs no
    # row of its own. The radius has to be named somewhere, and this is the one pointer.
    cx.set_xlabel(r"$m(r) \propto r^{\,d}$", labelpad=1.5)
    cx.annotate(r"$r_m$", xy=(points[centre, 0] - 0.70 * radius,
                              points[centre, 1] + 0.70 * radius),
                xytext=(points[centre, 0] - 1.34 * radius,
                        points[centre, 1] + 1.16 * radius),
                ha="left", va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=GREY, shrinkA=1, shrinkB=1),
                **POINTER)

    # Built by hand rather than collected from the axes: a key drawn at the marker size
    # the panel uses is a dot two pixels across, which names nothing.
    handles = [Line2D([], [], color=RECURRENT, marker=".", ms=6, mew=0, alpha=0.45,
                      ls="none", label="the reconstruction"),
               Line2D([], [], color=GREY, marker=".", ms=8, mew=0, ls="none",
                      label=r"excluded: $|\Delta t| \leq W_T$"),
               Line2D([], [], mfc="white", mec=RECURRENT, mew=1.3, marker="o", ms=5.4,
                      ls="none", label="reference point"),
               Line2D([], [], color=RECURRENT, marker=".", ms=8, mew=0, ls="none",
                      label=r"its $m$ nearest")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handlelength=0.8, handletextpad=0.35, columnspacing=1.1)
    return fig


def fig_shapes():
    """The object the neighbour search measures, in the plane of its first two coordinates."""
    d = pd.read_csv(HERE / 'logs/curve_shapes.csv')
    colour = {"qp": RECURRENT, "gd": TRANSIENT, "batch_proj": STOCHASTIC}
    order = ["one phase", "two phases", "a transient", "mini-batch noise"]

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.60))
    for ax, label in zip(axes, order):
        one = d[d.label == label]
        ax.plot(one.x.to_numpy(), one.y.to_numpy(), ".",
                color=colour[one.arm.iloc[0]], ms=0.9, alpha=0.55, mec="none")
        ax.set_title(label, pad=3.0)
        ax.set_aspect("equal", adjustable="box")
        span = 1.06 * max(np.abs(one.x).max(), np.abs(one.y).max())
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlabel("$z_t$", labelpad=1.0)
    axes[0].set_ylabel(r"$z_{t-\tau}$", labelpad=1.0)

    fig.tight_layout(w_pad=0.7)
    return fig


if __name__ == "__main__":
    draw(HERE, panels=(fig_method, fig_shapes,))

"""Slide style for the ICOMP talk figures.

Separate from actdim.figures.style on purpose. That module is calibrated for the
5.5 in text width of the article at 9 pt; a figure drawn at those settings and
thrown on a projector is unreadable from the third row. Here the type is 9.6-11 pt
at the size the figure actually appears on the slide, the lines are twice as
thick, and each figure carries at most four panels.

Every word inside a figure is English even though the deck around it is Russian.
The figures are meant to be lifted into the paper unchanged, and a figure that has
to be relabelled first is a figure that gets redrawn instead. What the audience
reads in Russian is the two lines of prose under the figure, which is where the
translation belongs.

Three numbers here are load-bearing.

:data:`FULL` is the exact ``\\textwidth`` of the deck in inches, so every figure
is inserted at ``width=\\textwidth`` and LaTeX never rescales it. An earlier
version drew at 5.85 in and inserted at ``0.93\\textwidth`` to buy room for text;
that shrank every label by seven per cent and made the panels squat, which is the
one thing a projector cannot forgive. The room now comes from carrying two or
three lines of text per slide instead of seven.

:data:`HEIGHT` is what a figure may be tall while a slide still has room for
those lines: the beamer body is about 208 pt, so 2.3 in of figure leaves 40 pt,
which is three lines of ``\\small``.

:data:`ANNOT` is the floor on type inside a panel. It used to be 8.5 pt in a
dozen places, which reads on a laptop and disappears in a lecture hall. Nothing
below 9.5 pt is drawn any more; where that did not fit, the annotation was cut
rather than shrunk, because an annotation nobody can read is not shorter than no
annotation, it is worse.

Palette is unchanged from the article (Paul Tol high-contrast), so a listener who
later opens the paper sees the same hue mean the same regime.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# \textwidth of this deck: 429.64 TeX pt = 5.945 in. Inserted at width=\textwidth,
# so the scale factor is 1 and the type is the size set below.
FULL = 5.945

# What a figure may be tall on a slide that still carries two lines of text.
HEIGHT = 2.30

RECURRENT = "#004488"   # deep blue   -- deterministic and recurrent
STOCHASTIC = "#BB5566"  # brick rose  -- stochastically driven
TRANSIENT = "#997700"   # dark gold   -- deterministic and transient
GOOD = "#117733"        # green       -- talk only: "this is the verified claim"
GREY = "#666666"
FAINT = "#BBBBBB"
INK = "#222222"

RC = {
    "font.size": 10.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 9.6,
    "legend.fontsize": 9.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # A 0.9 pt spine in #999 is a suggestion of an axis on a laptop and nothing
    # at all through a projector's gamma; the frame has to survive being thrown
    # on a wall, so it is darker and a touch heavier than the article's.
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#7F7F7F",
    "axes.labelcolor": INK,
    "axes.labelpad": 3.0,
    "axes.titlepad": 5.0,
    "axes.titlecolor": INK,
    "axes.grid": False,
    "text.color": INK,
    "xtick.color": "#7F7F7F",
    "ytick.color": "#7F7F7F",
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 3.4,
    "ytick.major.size": 3.4,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "lines.linewidth": 1.9,
    "lines.markersize": 5.0,
    "legend.frameon": False,
    "legend.borderpad": 0.0,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.0,
    "legend.columnspacing": 1.2,
    "figure.dpi": 160,
    "savefig.dpi": 320,
}

# The panel title, and the two annotation styles permitted inside the axes.
PANEL = dict(fontsize=11.0)
POINTER = dict(color=GREY, fontsize=9.6)
ANNOT = dict(fontsize=9.6)
NOTE = dict(fontsize=9.6)

# Backing for an annotation that has to sit over data. Without it the reader
# spends a second deciding whether a word is a label or a curve, and on a slide
# a second is the whole budget.
BOX = dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.4)

DATA = Path("code/data")
OUT = Path("talk/figures")


def context():
    return mpl.rc_context(RC)


def rows(height=HEIGHT, bottom=0.52, top=0.30):
    """Axes ``y`` and ``height`` in figure fractions, from margins in inches.

    ``bottom`` has to hold the tick labels and the axis name, ``top`` the panel
    title. Computing them rather than writing them down is what keeps the panels
    the same size after a figure's height changes, which it did twice.
    """
    return bottom / height, (height - bottom - top) / height


def title_y(height=HEIGHT, top=0.30):
    """Baseline for a panel title just above the axes."""
    return (height - top + 0.045) / height


def titles(fig, H, pairs, top=0.28):
    """Panel titles placed against the figure rather than the axes.

    An equal-aspect axis makes matplotlib set a title against the axes box
    instead of the slot it was handed, and two of them collided that way in an
    earlier draft. Writing them as figure text at a fixed distance from the top
    edge puts every panel title in the deck on one baseline.

    The titles say what the panel shows *and* how it came out. A neutral title
    plus a finding buried in an in-panel note makes the audience hunt for the
    point while the speaker has already moved on.
    """
    for x, name in pairs:
        fig.text(x, (H - top) / H, name, ha="left", va="bottom", **PANEL)


def cols(n, left=0.075, right=0.028, gap=0.115):
    """``(x, width)`` for ``n`` panels that fill the width, in figure fractions.

    The panels used to be placed by hand, and nobody added the numbers up: two
    of them came to two thirds of the width with 1.3 in of nothing between. This
    is worth more than any font change, because it is the only way to make a
    panel bigger without taking room from the text under it. A panel 2.32 in
    wide instead of 1.72 in is the same figure a third larger.

    ``gap`` has to hold the right panel's y label and tick labels, so a panel
    with words down its y axis needs a wider one than a panel with numbers.
    """
    w = (1.0 - left - right - (n - 1) * gap) / n
    return [(left + i * (w + gap), w) for i in range(n)]


def key(ax, entries, x=0.98, y=0.97, ha="right", va="top", dy=0.125,
        fontsize=9.8, weight="normal"):
    """A colour key written into the axes, one coloured line per entry.

    A legend box big enough to read costs about 0.28 in below the axes, which is
    an eighth of a 2.16 in figure, and beamer has no eighth to spare. Coloured
    words in a corner cost nothing, and they are read faster: nobody has to
    carry a swatch from the legend across the panel to the curve.
    """
    for i, (text, colour) in enumerate(entries):
        ax.text(x, y - i * dy, text, transform=ax.transAxes, ha=ha, va=va,
                color=colour, fontsize=fontsize, fontweight=weight, zorder=6)


def table(name: str):
    """One CSV out of the frozen data directory, by <experiment>/<file> path."""
    import pandas as pd
    return pd.read_csv(DATA / name)


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    width = float(fig.get_size_inches()[0])
    if abs(width - FULL) > 1e-6:
        raise ValueError(f"{stem}: width {width:.3f} in, need {FULL} in -- "
                         "otherwise LaTeX rescales the figure and the labels "
                         "stop being the size they were drawn at")
    with context():
        for suffix in ("pdf", "png"):
            fig.savefig(OUT / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf  ({width:.2f} x {fig.get_size_inches()[1]:.2f} in)")

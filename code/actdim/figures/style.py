"""The palette, the figure size, and the rcParams the article's figures are drawn at.

The numbers here are measured, not chosen: :data:`WIDTH` is the exact text width of the
ICOMP style file and the palette separations below were measured under simulated colour
blindness. The rules that hold them together are in the docstring of
:mod:`actdim.figures.panels`, which is where they were written down after a review found
a figure misrepresenting its own data.

Two of these are load-bearing and are checked rather than trusted. :func:`save` refuses a
figure that is not :data:`WIDTH` inches wide, and it never passes ``bbox_inches="tight"``,
because a legend wider than the axes then expands the canvas past the text width, LaTeX
scales the figure down to fit, and the 8 pt type shrinks with it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt

# The exact \textwidth of the ICOMP style, so LaTeX never rescales a figure. Figures in
# the appendices may be up to this wide and about 2.2 in tall.
WIDTH = 5.5

# Paul Tol high-contrast qualitative, in fixed order: one hue per regime.
#
# Measured: worst-pair separation 45.3 dE under simulated protanopia and 50.7 dE under
# simulated deuteranopia, minimum WCAG contrast 4.21 against white. This replaces an
# earlier Okabe-Ito-style blue/orange/purple set whose worst pair separated by only
# 8.8 dE under deuteranopia. Re-run those checks before changing any colour.
RECURRENT = "#004488"   # deep blue
STOCHASTIC = "#BB5566"  # brick rose
TRANSIENT = "#997700"   # dark gold
GREY = "#666666"        # for reference lines and pointers
FAINT = "#BBBBBB"
BAND = "#004488"

INK = "#333333"

# Deliberately light: 0.5 pt spines, short 0.5 pt ticks, no black, small markers, and no
# gridlines. fig_observers draws its own row guides, being a dot plot that needs them.
# The document sets 10 pt on 11 pt. Figure type at 9 pt reads a little below the body and
# a little below the caption, which is the convention; the 8 pt it was set at earlier read
# as small beside its own caption, and the tick labels at 7 pt read as small again beside
# the axis names.
RC: Dict[str, Any] = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#AAAAAA",
    "axes.labelcolor": INK,
    "axes.labelpad": 2.5,
    "axes.titlepad": 4.0,
    "axes.titlecolor": INK,
    "axes.grid": False,
    "text.color": INK,
    "xtick.color": "#AAAAAA",
    "ytick.color": "#AAAAAA",
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.4,
    "legend.frameon": False,
    "legend.borderpad": 0.0,
    "legend.handletextpad": 0.45,
    "legend.borderaxespad": 0.0,
    "figure.dpi": 200,
    "savefig.dpi": 300,
}

# The one annotation style permitted inside the axes: at most one short pointer per
# panel, where a mark would otherwise be unreadable.
POINTER = dict(color=GREY, fontsize=7.0)


def bounds(*series: Any, pad: float = 0.06, step: float = 0.0,
           include: Any = None) -> tuple:
    """Limits that contain everything drawn, padded and rounded outward to ``step``.

    The panels set their limits deliberately rather than letting matplotlib choose, so
    that the same quantity is drawn on the same scale wherever it appears. What a
    hand-written limit cannot do is survive the data moving under it: a limit chosen for
    one campaign silently stops containing the next, and the panel then prints its data in
    the margin or crops it away entirely. This keeps the deliberate part -- the padding,
    the rounding, the ticks the caller sets -- and takes the range from the data.

    ``include`` names values the panel needs inside the frame whatever the data does: a
    reference line, a floor, the zero of a difference.
    """
    import numpy as np

    values: List[float] = []
    for block in series:
        arr = np.asarray(block, dtype=float).ravel()
        values.extend(arr[np.isfinite(arr)].tolist())
    if include is not None:
        arr = np.asarray(include, dtype=float).ravel()
        values.extend(arr[np.isfinite(arr)].tolist())
    if not values:
        raise ValueError("no finite values to bound")

    low, high = min(values), max(values)
    margin = pad * (high - low) if high > low else pad * max(abs(high), 1.0)
    low, high = low - margin, high + margin
    if step > 0:
        import math

        low = step * math.floor(low / step)
        high = step * math.ceil(high / step)
    return low, high

FORMATS = ("pdf", "png")  # PDF for LaTeX, PNG for preview


def context():
    """The article's rcParams, for the duration of one figure.

    A context rather than a global mutation: importing a module should not change how
    every other plot in the process is drawn.
    """
    return mpl.rc_context(RC)


def save(fig: Any, stem: str, outdir: Path) -> List[Path]:
    """Write one figure into a directory the caller chose, as PDF and as PNG.

    No ``bbox_inches="tight"``: it expands the canvas past 5.5 in when a legend is wider
    than the axes, and LaTeX then scales the figure down and the type with it. The
    figures reserve space for their legends with ``tight_layout(rect=...)`` instead.
    """
    width = float(fig.get_size_inches()[0])
    if abs(width - WIDTH) > 1e-9:
        raise ValueError(f"figure {stem} must be {WIDTH} in wide, not {width:.4f}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    with context():
        for suffix in FORMATS:
            path = outdir / f"{stem}.{suffix}"
            fig.savefig(path)
            written.append(path)
    plt.close(fig)
    return written

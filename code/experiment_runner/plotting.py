"""Shared formatting and output directory for experiment figures."""
import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CODE = Path(__file__).resolve().parents[1]

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



def draw(here, panels=(), summary=None):
    """Render the experiment figures to PDF."""
    output = options(here)
    for panel in panels:
        with mpl.rc_context(RC):
            figure = panel()
        figure.savefig(output / f"{panel.__name__}.pdf")
        plt.close(figure)
    if summary is not None:
        draw_summary(here, output, summary)
    print(output)


def draw_summary(here, output, summary):
    """Draw a small comparison for experiments published as tables."""
    filename, x, columns, xlabel, ylabel, kind = summary
    table = pd.read_csv(here / "logs" / filename)
    medians = table.groupby(x)[list(columns)].median()
    ax = medians.plot(kind="bar", rot=32) if kind == "bar" else medians.plot(marker="o", linestyle="none" if kind == "scatter" else "-")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "summary.pdf")
    plt.close(ax.figure)

def options(here):
    parser = argparse.ArgumentParser(description="Regenerate PDFs from the bundled CSVs.")
    parser.add_argument("--output", type=Path, default=CODE / "runs" / "redrawn" / here.name)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args.output

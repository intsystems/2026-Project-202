"""Matplotlib style in points at the figure's final manuscript width."""

import matplotlib as mpl


# The NeurIPS text block is 5.5 in wide. Other editions scale all the
# manuscript figures together when they place them in a different text block.
TEXT_WIDTH = 5.5
LABEL = TITLE = 7.5
TICK = LEGEND = ANNOTATION = 6.5
INK = "#20262B"


def pt(size, width, fraction=1.0):
    """Convert printed points to points in the source PDF."""
    return size * width / (TEXT_WIDTH * fraction)


def figure_style(width, fraction=1.0):
    """Style a canvas of width (inches) included at fraction of the text block."""
    def size(points):
        return pt(points, width, fraction)

    return mpl.rc_context({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": size(ANNOTATION),
        "axes.labelsize": size(LABEL),
        "axes.titlesize": size(TITLE),
        "xtick.labelsize": size(TICK),
        "ytick.labelsize": size(TICK),
        "legend.fontsize": size(LEGEND),
        "legend.title_fontsize": size(LEGEND),
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": size(0.55),
        "axes.axisbelow": True,
        "axes.labelpad": size(3),
        "axes.titlepad": size(4),
        "xtick.major.width": size(0.55),
        "ytick.major.width": size(0.55),
        "xtick.major.size": size(2.6),
        "ytick.major.size": size(2.6),
        "xtick.major.pad": size(2.5),
        "ytick.major.pad": size(2.5),
        "grid.color": "#9EABB4",
        "grid.linewidth": size(0.5),
        "lines.linewidth": size(1.05),
        "pdf.fonttype": 42,
    })

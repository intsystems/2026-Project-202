"""Draw the manuscript's delay-reconstruction examples from the bundled CSVs."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import figure_style, pt


FIGURES = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGURES.parents[2] / "code"))
from synthetic.delay_embedding_and_limits.delay_embedding.analyze_logs import fig_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    width, fraction = 5.5, .86  # The experiment canvas and its width in appendix.tex.
    with figure_style(width, fraction):
        fig = fig_shapes()
        fig.set_figheight(1.75)
        for ax in fig.axes:
            ax.set_title(ax.get_title(), pad=pt(4, width, fraction))
        fig.tight_layout(w_pad=.7)
        fig.savefig(args.output / "icomp_shapes.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()

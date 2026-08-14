"""Every mark a figure draws has to be inside the figure.

The panels set their axis limits deliberately, which is right and is argued for in
`actdim.figures.style`. The failure that comes with it is silent: a limit chosen for one
campaign stops containing the next, and because most panels draw with ``clip_on=False`` so
that a marker on a limit is not cut in half, the points are then drawn in the margin rather
than dropped. The figure looks finished. Two figures shipped that way after the
recomputation -- `fig_pairs` moved from 18--26 to 11--16 against an axis fixed at
17.7--27.7, and `fig_regimes` panel (b) put its transient row at 1.79--1.91 against an axis
fixed at 0.92--1.66 -- and neither the build nor the test suite said anything.
"""
from __future__ import annotations

import pytest

from actdim.figures import extent
from actdim.figures.panels import NAMES
from actdim.figures.sources import data_root

pytestmark = pytest.mark.skipif(not data_root().is_dir(),
                                reason="data/ is not present")


@pytest.mark.parametrize("name", NAMES)
def test_every_mark_is_inside_its_panel(name):
    found = extent.audit([name]).get(name, [])
    assert not found, "\n".join(item.describe() for item in found)


def test_the_check_sees_a_limit_that_stops_containing_its_data():
    """The guard itself, on a panel built to fail: an unclipped point in the margin, and a
    clipped series with nothing inside the frame at all."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2)
    ax.plot([0.5, 5.0], [1.0, 1.0], "o", clip_on=False)
    ax.set_xlim(0, 2)
    bx.plot([10.0, 11.0], [1.0, 1.0], "-")           # clipped, entirely outside
    bx.set_xlim(0, 2)
    found = extent.overflows(fig, "made up")
    plt.close(fig)

    kinds = {item.kind for item in found}
    assert kinds == {"in the margin", "nothing visible"}


def test_the_check_passes_a_clipped_series_that_crosses_the_window():
    """A line running past a limit with clipping on is what a chosen limit is for, and
    must not be reported: every panel of fig_dip does it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([-10.0, 10.0], [1.0, 1.0], "-")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 2)
    found = extent.overflows(fig, "made up")
    plt.close(fig)
    assert not found

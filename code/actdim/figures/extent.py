"""Whether every mark a panel draws lies inside the panel.

The figures set their axis limits by hand, which is right: an automatic limit moves
whenever the data moves, and two figures of the same quantity then cannot be compared. The
cost is that a limit chosen for one dataset silently stops containing the next one. Every
panel that draws with ``clip_on=False`` -- most of them, so that a marker sitting on a
limit is not cut in half -- then draws its points outside the frame instead of dropping
them, and the figure looks finished while its data is in the margin.

That is what the recomputed campaign did to two figures: ``fig_pairs`` moved from 18--26 to
11--16 against an axis fixed at 17.7--27.7, and the transient row of ``fig_regimes``
panel (b) moved to 1.79--1.91 against an axis fixed at 0.92--1.66.

:func:`overflows` reports it. It is a check on the drawing and not on the data, so it reads
the artists back off the axes rather than being told what was plotted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: Fraction of the axis span a mark may sit beyond a limit before it is reported. Not
#: zero: a reference line is often drawn exactly to the limit, and a marker's own radius
#: legitimately overhangs it.
TOLERANCE = 0.005


@dataclass(frozen=True)
class Overflow:
    """One artist that leaves its axes, and by how far."""

    figure: str
    panel: str
    artist: str
    axis: str                                     # "x" or "y"
    limits: Tuple[float, float]
    span: Tuple[float, float]                     # the data's own extent
    beyond: float                                 # fraction of the axis span, worst side
    kind: str = "in the margin"                   # or "nothing visible"

    def describe(self) -> str:
        return (f"{self.figure} {self.panel}: {self.artist} spans "
                f"{self.span[0]:.4g} to {self.span[1]:.4g} on {self.axis}, "
                f"outside {self.limits[0]:.4g} to {self.limits[1]:.4g} "
                f"by {100 * self.beyond:.0f}% of the axis ({self.kind})")


def _points(artist: Any) -> Sequence[np.ndarray]:
    """The (x, y) a drawn artist occupies, or nothing where it has no data."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if isinstance(artist, Line2D):
        return [np.column_stack([artist.get_xdata(orig=False),
                                 artist.get_ydata(orig=False)])]
    if isinstance(artist, Collection):
        out = []
        for path in artist.get_paths() or []:
            if len(path.vertices):
                out.append(np.asarray(path.vertices, dtype=float))
        # A filled region carries its geometry in its paths, and matplotlib leaves its
        # offsets at the default single (0, 0). Reading those as data puts every
        # fill_between at the origin and reports it as outside every axis but one.
        if not out:
            offsets = artist.get_offsets()
            if len(offsets):
                out.append(np.asarray(offsets, dtype=float))
        return out
    if isinstance(artist, Patch):
        path = artist.get_path().transformed(artist.get_patch_transform())
        return [np.asarray(path.vertices, dtype=float)] if len(path.vertices) else []
    return []


def overflows(fig: Any, name: str = "") -> List[Overflow]:
    """Every unclipped artist drawn outside its own axis limits.

    Clipping is what decides whether this is a defect. A line that runs past the limit
    with clipping on is cropped at the frame, which is what a chosen limit is for. The
    same line with ``clip_on=False`` -- which these panels set so that a marker sitting on
    a limit is not cut in half -- is drawn in the margin instead, and that is the failure
    this function exists to catch.

    Only artists positioned in data coordinates on both axes are examined: ``axhline`` and
    ``axvline`` span 0 to 1 in axes coordinates by construction, and reading those as data
    would report every reference line in the article.
    """
    found: List[Overflow] = []
    for index, ax in enumerate(fig.get_axes()):
        panel = ax.get_title(loc="left") or ax.get_title() or f"panel {index}"
        limits = {"x": ax.get_xlim(), "y": ax.get_ylim()}
        spans = {k: abs(v[1] - v[0]) for k, v in limits.items()}
        for artist in list(ax.lines) + list(ax.collections) + list(ax.patches):
            if not artist.get_visible():
                continue
            if artist.get_transform() is not ax.transData:
                continue
            clipped = bool(artist.get_clip_on())
            label = str(getattr(artist, "get_label", lambda: "")() or "")
            for block in _points(artist):
                if block.size == 0:
                    continue
                block = np.asarray(block, dtype=float)
                good = np.isfinite(block).all(axis=1)
                if not good.any():
                    continue
                # A clipped artist that crosses the window is doing what a chosen limit is
                # for. A clipped artist that misses the window entirely has been cropped
                # away, and the panel is blank where its data should be. The test is
                # whether the artist's own extent overlaps the frame, not whether one of
                # its vertices is inside it: a segment can cross the window with both
                # endpoints outside, which is the ordinary case in fig_dip.
                overlaps = True
                for axis, column in (("x", 0), ("y", 1)):
                    low, high = sorted(limits[axis])
                    values = block[good, column]
                    overlaps &= (values.min() <= high) and (values.max() >= low)
                vanished = clipped and not overlaps
                if clipped and not vanished:
                    continue
                for axis, column in (("x", 0), ("y", 1)):
                    values = block[good, column]
                    low, high = sorted(limits[axis])
                    span = spans[axis] or 1.0
                    beyond = max((low - values.min()) / span,
                                 (values.max() - high) / span)
                    if beyond > TOLERANCE:
                        found.append(Overflow(
                            figure=name, panel=panel, artist=label or type(artist).__name__,
                            axis=axis, limits=(low, high),
                            span=(float(values.min()), float(values.max())),
                            beyond=float(beyond),
                            kind="nothing visible" if vanished else "in the margin"))
    return found


def spills(fig: Any, name: str = "") -> List[Overflow]:
    """Every legend or axis label drawn past the edge of the canvas.

    The canvas is exactly the text width and is never expanded by ``bbox_inches``, so
    anything past its edge is cropped on the page rather than shrinking the figure. A
    legend is the usual offender: it is laid out at the type size and stops fitting when
    the type grows or an entry is renamed.
    """
    found: List[Overflow] = []
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.dpi
    items = [(legend, "legend") for legend in fig.legends]
    for ax in fig.get_axes():
        if ax.get_legend() is not None:
            items.append((ax.get_legend(), "legend"))
    for artist, what in items:
        box = artist.get_window_extent()
        for axis, low, high, limit in (("x", box.x0, box.x1, width),
                                       ("y", box.y0, box.y1, height)):
            beyond = max(-low / limit, (high - limit) / limit)
            if beyond > TOLERANCE:
                found.append(Overflow(
                    figure=name, panel="canvas", artist=what, axis=axis,
                    limits=(0.0, float(limit)), span=(float(low), float(high)),
                    beyond=float(beyond), kind="past the canvas edge"))
    return found


def audit(names: Sequence[str] = (), allow_archive: bool = False) -> Dict[str, Any]:
    """Draw every figure and report what leaves its axes. Writes nothing."""
    import matplotlib.pyplot as plt

    from .panels import NAMES, Reader, draw

    from .style import context

    wanted = list(names) if names else list(NAMES)
    out: Dict[str, Any] = {}
    for name in wanted:
        # Drawn under the article's rcParams, since the type size is what decides whether
        # a legend fits: measuring it at matplotlib's defaults would measure a different
        # figure from the one that is published.
        with context():
            fig = draw(name, Reader(allow_archive=allow_archive))
            out[name] = overflows(fig, name) + spills(fig, name)
            plt.close(fig)
    return out


def main(argv: Sequence[str] = ()) -> int:
    report = audit(tuple(argv))
    total = 0
    for name, found in report.items():
        if not found:
            continue
        total += len(found)
        for item in found:
            print(item.describe())
    if total:
        print(f"\n{total} artist(s) drawn outside their axes")
        return 1
    print(f"{len(report)} figure(s): every mark is inside its panel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The Theiler-excluded nearest-neighbour query, which every estimate is built on.

One query serves the pooled estimate, the per-point variant and TwoNN. They differ only in
how the same distances are reduced, and running three queries to get three reductions of one
matrix was the largest avoidable cost in the archived tree.

The exclusion is the point of this module. A trajectory sampled faster than it moves has, as
the nearest neighbours of any point, the points immediately before and after it in time; the
distances between those measure the tangent line, not the set the trajectory fills, and the
estimate that comes back is about 1.2 whatever the system is. Excluding every candidate
within ``W`` samples in time forces the neighbours to be genuine returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Neighbours:
    """Distances to the ``k`` nearest valid neighbours of every point, floored.

    ``floor_fraction`` is the share of those distances that reached the floor. It is counted
    here, at the only place that knows how many distances there were, and carried up to the
    degeneracy indicator.
    """

    distances: np.ndarray
    floor_fraction: float


def neighbour_distances(points: np.ndarray, k: int, theiler: int,
                        floor: Optional[float] = 1e-8) -> Optional[Neighbours]:
    """Distances to the ``k`` nearest neighbours of each point, excluding temporal ones.

    Returns ``None`` when the cloud cannot supply ``k`` valid neighbours per point, which is
    a refusal and not a small number: a window that short has no estimate, and returning one
    anyway is how a plot comes to show a dimension where there was no measurement.

    The exclusion is enforced by over-asking. At most ``2W + 1`` candidates around a point
    are excluded, so a query for ``k + 2W + 1`` always leaves ``k`` valid ones; a stable sort
    on the validity mask then pulls those to the front without disturbing the ascending order
    the tree returned them in.
    """
    from sklearn.neighbors import KDTree

    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    if n_points == 0 or k < 2:
        return None
    tree = KDTree(points)

    if theiler <= 0:
        if n_points < k + 2:
            return None
        distances, _ = tree.query(points, k=k + 1)
        distances = distances[:, 1:]  # the first neighbour of a point is itself
    else:
        excluded = 2 * theiler + 1
        if n_points - excluded < k:
            return None
        distances, indices = tree.query(points, k=min(n_points, k + excluded))
        valid = np.abs(indices - np.arange(n_points)[:, None]) > theiler
        order = np.argsort(~valid, axis=1, kind="stable")
        distances = np.take_along_axis(distances, order, axis=1)[:, :k]

    if floor is None:
        return Neighbours(distances, 0.0)
    # Counted after flooring and with a relative tolerance, so that a distance the floor
    # clipped and one that was already exactly at it both register. Both mean the same thing:
    # two delay vectors the reconstruction cannot tell apart.
    floored = np.maximum(distances, floor)
    fraction = float((floored <= floor * 1.000001).mean())
    return Neighbours(floored, fraction)

"""Counting directions in a cloud of points, stated once.

Two different counts are wanted and the archived tree kept confusing them.

The **numerical rank** is a hard count: how many singular values are above a relative
threshold. It answers "is this trajectory confined to a subspace at all", and it is a step
function of the threshold, which is why the threshold is an argument here and never a
literal buried in a caller.

The **participation ratio** is a soft count, ``(sum s^2)^2 / sum s^4`` over the singular
values. It answers "how many directions carry comparable energy", which is the quantity
the article calls the effective rank and scores every estimate against: a trajectory with
one strong direction and nine weak ones has numerical rank ten and effective rank near
one, and it is the second number that says what a dimension estimator can find.

The archived tree had four copies of the three lines below -- ``system.rank_pr``,
``generators.state_rank``, ``e7b_theiler_quick.participation_ratio`` and
``e8_anisotropy.participation_ratio`` -- with relative thresholds of 1e-5, 1e-8, 1e-8 and
1e-7. Two of them centred the cloud first and two did not, so the same trajectory had two
ranks depending on which script measured it. Centring is an argument here, defaulting to
on, because a covariance rank is a statement about the fluctuation and not about where the
cloud sits.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

#: Relative singular-value threshold for a hard rank. Anything below this times the
#: largest singular value is numerical noise from the SVD, not a direction.
RANK_TOL = 1e-8

#: The threshold the published trajectory and update ranks of the image-data system were
#: measured at (``system.rank_pr`` used ``sqrt(1e-10)``). A rank is a step function of its
#: threshold, so that system keeps the value its results were produced with rather than
#: silently moving to ``RANK_TOL``.
TRAJECTORY_RANK_TOL = 1e-5


def participation_ratio(weights) -> float:
    """``(sum w)^2 / sum w^2`` for non-negative weights: how many are of comparable size.

    Equals ``n`` when all ``n`` weights are equal and ``1`` when one dominates. The
    weights are variances -- squared singular values or eigenvalues -- never amplitudes.
    Returns ``nan`` for an all-zero spectrum, where the question has no answer, rather
    than dividing by zero and letting a warning escape into a result table.
    """
    w = np.asarray(weights, dtype=float).ravel()
    total = float(w.sum())
    square = float(np.sum(w * w))
    if not np.isfinite(total) or square <= 0.0:
        return float("nan")
    return total * total / square


def spectrum(matrix, center: bool = True) -> np.ndarray:
    """Squared singular values of the row cloud, largest first.

    With ``center`` the mean row is removed first, which makes this the eigenvalue
    spectrum of the covariance. Without it, a cloud far from the origin reads as
    one-dimensional whatever its shape.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.size == 0:
        return np.zeros(0)
    if center:
        a = a - a.mean(axis=0, keepdims=True)
    s = np.linalg.svd(a, compute_uv=False)
    return s * s


def effective_rank(matrix, center: bool = True) -> float:
    """Participation ratio of the row cloud: the article's effective rank."""
    return participation_ratio(spectrum(matrix, center=center))


def numerical_rank(matrix, center: bool = True, tol: float = RANK_TOL) -> int:
    """How many singular values exceed ``tol`` times the largest."""
    s2 = spectrum(matrix, center=center)
    if s2.size == 0 or s2[0] <= 0.0:
        return 0
    return int(np.sum(np.sqrt(s2) > np.sqrt(s2[0]) * tol))


class RankReport(NamedTuple):
    """Both counts and the conditioning, from one decomposition."""

    rank: int
    effective_rank: float
    singular_ratio: float  # smallest over largest: 0 means a direction is not excited


def rank_report(matrix, center: bool = True, tol: float = RANK_TOL) -> RankReport:
    """The hard rank, the effective rank and the singular-value ratio, from one SVD.

    Callers want all three together -- the rank says a subspace exists, the effective rank
    says the directions are comparably excited, the ratio says how far the weakest is from
    the strongest -- and computing them separately means three decompositions of the same
    matrix.
    """
    s2 = spectrum(matrix, center=center)
    if s2.size == 0 or s2[0] <= 0.0:
        return RankReport(0, float("nan"), 0.0)
    s = np.sqrt(s2)
    return RankReport(int(np.sum(s > s[0] * tol)),
                      participation_ratio(s2),
                      float(s[-1] / s[0]))


def orthonormal(shape, rng: np.random.Generator, columns: Optional[int] = None) -> np.ndarray:
    """A random frame with orthonormal columns, drawn from ``rng``.

    One place for ``qr(standard_normal(...))[0]``, which appears in every system that
    needs a generic subspace of a given dimension.
    """
    q = np.linalg.qr(rng.standard_normal(shape))[0]
    return q if columns is None else q[:, :columns]


def unit(vector) -> np.ndarray:
    """A vector scaled to unit length; unchanged if it is already zero."""
    v = np.asarray(vector, dtype=float)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n

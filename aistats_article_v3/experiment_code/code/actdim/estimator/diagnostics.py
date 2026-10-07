"""Whether a window may be read as a dimension at all.

An estimate exists for any window; whether it means anything is a separate question, and
these three statistics are what answer it. They are computed from the series and the estimate
alone, so a run cannot report an estimate without being able to report them.

**Identifiability ratio**, ``d(2E) / d(E)``. Near unity when a dimension is resolvable, and
growing towards 2 when the estimate is a property of the embedding space rather than of the
data -- which is what the stochastically driven regime looks like, where there is no
low-dimensional invariant set for the estimate to be about.

**Trend-crossing count**, the sign changes of the window's residual about its least-squares
line. The ratio alone cannot separate the stochastic regime from a decaying transient, where
it also reads about 1, for the wrong reason: a transient traces a curve it never returns to,
so the neighbour statistic has no returns to work with and reports about 29 whatever the true
dimension is. The count is near zero there and large on a recurrent orbit. It tests
non-monotonicity of the series, not recurrence in the reconstructed space, so the two are
only useful together.

**Degeneracy indicator**, raised when too many neighbour distances or per-point sums reach
their numerical floors. It identifies no regime. It says the window is quantised or constant
enough that the floors, not the data, set the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EstimatorConfig
from .mle import estimate


def ratio(low: float, high: float) -> float:
    """``high / low``, or NaN where the division would not mean anything.

    A non-positive or non-finite denominator is a failed estimate, and dividing by it would
    turn one into a large finite ratio that looks like a diagnosis.
    """
    if not np.isfinite(low) or low <= 0 or not np.isfinite(high):
        return float("nan")
    return float(high / low)


def identifiability_ratio(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> float:
    """The estimate at twice the embedding dimension, over the estimate at ``max_E``.

    Both embeddings are built on the same window, so the ratio compares two embedding
    dimensions and not two stretches of record. Doubling ``max_E`` roughly doubles the cost
    of the neighbour query, which is why the article computes this on a subset of observers.
    """
    at_e = estimate(x, cfg, seed).MG
    at_2e = estimate(x, cfg.replace(max_E=2 * cfg.max_E), seed).MG
    return ratio(at_e, at_2e)


def trend_crossings(x: np.ndarray) -> float:
    """Sign changes of the residual about the window's least-squares line.

    A count, returned as a float, because a window holding a non-finite sample has no count
    and NaN is the honest answer. The archived versions -- there were three, in three
    experiment scripts -- fitted with ``polyfit``, which raises from inside a worker on such
    a window. The line is fitted in closed form here instead: two moments, no solver.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or not np.isfinite(x).all():
        return float("nan")
    t = np.arange(len(x), dtype=np.float64)
    t_centred = t - t.mean()
    denom = float(t_centred @ t_centred)
    if denom <= 0:
        return float("nan")
    slope = float(t_centred @ (x - x.mean())) / denom
    residual = x - x.mean() - slope * t_centred
    return float(np.count_nonzero(np.diff(np.signbit(residual))))


@dataclass(frozen=True, eq=False)
class Diagnostics:
    """The three, together, because no one of them settles anything on its own."""

    identifiability_ratio: float
    trend_crossings: float
    degenerate: bool

    # No ``admissible`` verdict is offered. Two of these three have no threshold in the
    # article -- the reference values are ranges measured on known systems -- and a boolean
    # here would be a threshold invented by this module and then quoted as if it were one.
    # A caller states its own cut, next to the result it is cutting.


def diagnose(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> Diagnostics:
    """All three diagnostics of one window."""
    at_e = estimate(x, cfg, seed)
    at_2e = estimate(x, cfg.replace(max_E=2 * cfg.max_E), seed)
    return Diagnostics(
        identifiability_ratio=ratio(at_e.MG, at_2e.MG),
        trend_crossings=trend_crossings(x),
        degenerate=bool(at_e.degenerate),
    )

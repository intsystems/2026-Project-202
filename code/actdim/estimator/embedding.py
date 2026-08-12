"""The delay reconstruction: standardise, dither, embed, and size the Theiler exclusion.

This is the bottom of the estimator. Everything above it works on the object this module
returns, so the series is standardised once, dithered once and embedded once however many
statistics are asked for -- in the archived tree the same window was embedded up to four
times by four callers, which was slow and, worse, gave each caller its own dither.

The order of the first three steps is the order of appendix A and it matters. Standardising
first makes the dither and the distance floor mean the same thing on every observer: the
dither is 1e-9 of a standard deviation, not 1e-9 of whatever units the observer happens to
be in, and the 1e-8 floor is a distance in those same units. An unstandardised gradient norm
and an unstandardised loss are not comparable at the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import EstimatorConfig

#: An autocorrelation time is the first lag at which the correlation falls below this.
ACF_THRESHOLD = 1.0 / np.e


def standardise(x: np.ndarray) -> np.ndarray:
    """``(x - mean) / std``, the first line of appendix A.

    A constant series standardises to zeros rather than to ``0/0``: the division would raise
    a RuntimeWarning and produce NaN, and it is the degeneracy check downstream, not a
    silent NaN here, that should report a flat window.

    A series holding an infinity standardises to NaN for the same reason. Centring it would
    evaluate ``inf - inf``, which warns, and NumPy's warning would surface wherever the
    caller happened to be rather than where the bad sample is.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        return np.full(x.shape, np.nan)
    spread = x.std()
    centred = x - x.mean()
    if not np.isfinite(spread) or spread <= 0.0:
        return centred
    return centred / spread


def dither(x: np.ndarray, scale: Optional[float], rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise of the given scale, to break exact ties in the neighbour search.

    Quantised observers -- an accuracy, a loss that has converged to machine precision --
    produce exactly coincident delay vectors, and a zero distance makes the log-ratio
    infinite. ``scale=None`` skips it, which is only useful for showing what the dither is
    worth. The generator is passed in rather than made here so that two processes scoring the
    same window get the same dither.
    """
    x = np.asarray(x, dtype=np.float64)
    if scale is None:
        return x
    return x + rng.normal(0.0, scale, size=len(x))


def delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """The delay vectors of a series, as an ``(n - (m-1)*tau, m)`` array.

    Raises when the series is shorter than one delay vector, which is a caller error worth
    hearing about rather than an empty array to trip over three frames later.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = len(x) - (m - 1) * tau
    if n_points <= 0:
        raise ValueError(f"series of {len(x)} samples is too short for m={m}, tau={tau}")
    return np.column_stack([x[i:i + n_points] for i in range(0, m * tau, tau)])


def autocorrelation_time(series: np.ndarray, threshold: float = ACF_THRESHOLD,
                         max_lag: Optional[int] = None) -> int:
    """First lag at which the autocorrelation falls below ``threshold``.

    Samples closer together than this are correlated by the continuity of the trajectory
    rather than by the geometry of the set it fills, which is what the Theiler exclusion is
    sized to remove. Returns ``len(series)`` when the correlation never decays -- the honest
    answer for a monotone record, and the reason the transient arm asks for an exclusion of
    about 1600 samples.

    Returns 0 for a series with no spread, and for one with a non-finite sample: the archived
    version scanned every lag of such a series to no purpose, at quadratic cost, and returned
    the length. Neither is used by the estimator, which rejects a non-finite window earlier.
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < 3 or not np.isfinite(x).all():
        return 0
    if x.std() < 1e-12:
        return 0

    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    for lag in range(1, max_lag + 1):
        if np.dot(x[:-lag], x[lag:]) / denom < threshold:
            return lag
    return n


def resolve_tau(cfg: EstimatorConfig, x: np.ndarray) -> int:
    """Turn ``tau="acorr"`` into a lag measured from the window itself.

    A delay vector spans ``(max_E - 1) * tau`` samples, and a torus is only unfolded when
    that span covers a real fraction of the oscillation period. ``"acorr"`` takes a quarter
    of the autocorrelation time, the textbook choice, so that the estimator adapts to the
    signal's own timescale and a comparison across regimes is about the regimes. The span is
    capped at an eighth of the window, so the embedding still has points to work with.
    """
    if cfg.tau != "acorr":
        return int(cfg.tau)
    measured = autocorrelation_time(np.asarray(x, dtype=np.float64))
    tau = max(1, int(round(measured / 4.0)))
    return int(min(tau, max(1, len(x) // (8 * max(1, cfg.max_E - 1)))))


def resolve_theiler(cfg: EstimatorConfig, series: np.ndarray, tau: int) -> int:
    """The Theiler exclusion in samples, capped.

    The rule of appendix A is ``max((max_E - 1) * tau, t_acf)``: at least the span of one
    delay vector, and wider where the record is oversampled. The cap applies to every
    setting, including an integer one, so that the cost of the neighbour query -- which grows
    linearly in the exclusion, because the query has to ask for ``k + 2W + 1`` candidates to
    be sure of ``k`` valid ones -- has one bound and one place to raise it.
    """
    setting = cfg.theiler
    if setting is None or setting == 0 or setting == "none":
        base = 0
    elif setting == "embedding":
        base = (cfg.max_E - 1) * tau
    elif setting == "autocorr":
        base = max((cfg.max_E - 1) * tau, autocorrelation_time(series))
    else:
        base = int(setting)
    return int(min(base, cfg.theiler_cap))


@dataclass(frozen=True, eq=False)
class Reconstruction:
    """One window, prepared: the standardised series and its delay vectors.

    ``reason`` is empty when the reconstruction is usable and otherwise names why it is not,
    so that a caller can tell a window that was too short from one that was flat. The two are
    reported differently: nothing at all can be computed on the first, while the companion
    statistics are still defined on the second.
    """

    series: np.ndarray
    points: np.ndarray
    tau: int
    theiler: int
    reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.reason


def reconstruct(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> Reconstruction:
    """Standardise, dither, embed, and resolve the lag and the exclusion.

    Refuses three windows, each for a stated reason rather than by returning something that
    looks like a measurement: one shorter than the embedding needs, one with a non-finite
    sample, and one with no spread at all.
    """
    x = np.asarray(x, dtype=np.float64)
    empty = np.empty((0, 0), dtype=np.float64)

    # The length gate uses the configured lag, not the resolved one, because "acorr" cannot
    # be resolved until the window is known to be long enough to measure it on.
    span = cfg.max_E * (1 if cfg.tau == "acorr" else int(cfg.tau))
    if len(x) < span + 20:
        return Reconstruction(x, empty, 0, 0, "short")
    if not np.isfinite(x).all():
        return Reconstruction(x, empty, 0, 0, "nonfinite")
    if x.std() <= 0.0:
        return Reconstruction(x, empty, 0, 0, "flat")

    z = dither(standardise(x), cfg.dither, np.random.default_rng(seed))
    tau = resolve_tau(cfg, x)
    # Resolved on the dithered series, as the archived implementation did. A dither of 1e-9
    # cannot move an integer lag, but keeping the order identical keeps the numbers identical.
    theiler = resolve_theiler(cfg, z, tau)
    try:
        points = delay_embedding(z, cfg.max_E, tau)
    except ValueError:
        return Reconstruction(z, empty, tau, theiler, "embedding")
    return Reconstruction(z, points, tau, theiler)

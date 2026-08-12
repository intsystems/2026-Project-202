"""The estimate: pooled maximum likelihood on nearest-neighbour distances.

Appendix A of the article, implemented. Under the Poisson model the per-point statistic

    S_i = sum_{j=1}^{m-1} log( r_m(y_i) / r_j(y_i) ),    d * S_i ~ Gamma(m - 1, 1)

is shared by both poolings. Averaging the per-point estimates ``(m-1)/S_i`` is Levina and
Bickel's original; it is biased upwards by ``(m-1)/(m-2)`` because ``E[1/S_i] = d/(m-2)``,
and the mean is dominated by the few points with anomalously small ``S_i``. Pooling the
likelihood before inverting, as MacKay and Ghahramani argue one should, gives the estimate
this package reports. Both are returned, because the article compares them.

Three departures from the pooled formula are deliberate and each is stated in the article.

*The returned value is ``(N(m-1) - 1) / S``, not ``N(m-1) / S``.* It is the exactly unbiased
maximum likelihood estimate for the pooled Gamma sample. At the sizes used here the two
differ by less than 1e-5.

*A degenerate window is returned, not discarded.* The flag travels with the value and the
pipeline decides. Dropping it here would mean the estimator silently choosing which windows
count, and a caller taking a median over what was left could not say what it had taken a
median over.

*The value is never clamped to ``max_E``.* The archived kernel returned exactly ``max_E``
whenever the raw estimate exceeded ``2 * max_E``. That is one of the two silent defects the
old report records: it converts a divergent estimate into a plausible number, and it would
have turned the transient regime -- where the estimate reaches about 29 against a true
dimension of 1 -- into a result that looked like a measurement. A divergent estimate is
reported as divergent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import EstimatorConfig
from .embedding import Reconstruction, reconstruct
from .neighbours import neighbour_distances


@dataclass(frozen=True, eq=False)
class Estimate:
    """What the neighbour search yields on one window.

    ``degenerate`` is the article's degeneracy indicator: too many distances or too many
    per-point sums reached their numerical floors, so the geometry the estimate describes is
    the floor's and not the data's. It identifies no dynamical regime; it says the window
    cannot be read. The two fractions are kept beside it so that a marked window can be
    diagnosed without recomputing it.
    """

    MG: float = float("nan")
    LB: float = float("nan")
    TwoNN: float = float("nan")
    degenerate: bool = True
    floor_distance_fraction: float = float("nan")
    floor_sum_fraction: float = float("nan")
    n_points: int = 0
    tau: int = 0
    theiler: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """The record as it is written to a table, under the archived column names."""
        return {
            "MG": self.MG,
            "LB": self.LB,
            "TwoNN": self.TwoNN,
            "degenerate": self.degenerate,
            "frac_floor": self.floor_distance_fraction,
            "frac_sumfloor": self.floor_sum_fraction,
            "tau_used": float(self.tau),
            "theiler_used": float(self.theiler),
        }


def estimate(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> Estimate:
    """Score one window: standardise, embed, query, pool.

    ``seed`` seeds the dither and nothing else. Two processes scoring the same window with
    the same seed get the same number to the last bit.
    """
    return estimate_from(reconstruct(x, cfg, seed), cfg)


def estimate_from(rec: Reconstruction, cfg: EstimatorConfig) -> Estimate:
    """The estimate from an already-built reconstruction.

    Separate from :func:`estimate` so that the companion statistics of
    ``actdim.estimator.companions`` can share the one embedding, rather than each rebuilding
    it with its own dither.
    """
    if not rec.usable:
        return Estimate(tau=rec.tau, theiler=rec.theiler)

    found = neighbour_distances(rec.points, cfg.k_neighbors, rec.theiler,
                                floor=cfg.floor_distance)
    if found is None:
        return Estimate(tau=rec.tau, theiler=rec.theiler)

    distances = found.distances
    r_m = distances[:, -1:]
    sums = np.sum(np.log(r_m / distances[:, :-1]), axis=1)
    n = len(sums)
    sum_floor_fraction = float((sums <= cfg.floor_ratio_sum).mean())
    sums = np.maximum(sums, cfg.floor_ratio_sum)
    total = float(sums.sum())

    mg = lb = float("nan")
    if np.isfinite(total) and total > 0:
        mg = (n * (cfg.k_neighbors - 1) - 1) / total
        local = (cfg.k_neighbors - 1) / sums
        local = local[np.isfinite(local)]
        # Guarded: the mean of an empty selection is a RuntimeWarning and a NaN, and the
        # archived implementation took it unguarded.
        lb = float(np.mean(local)) if len(local) else float("nan")

    degenerate = bool(found.floor_fraction > cfg.degenerate_fraction
                      or sum_floor_fraction > cfg.degenerate_fraction)
    return Estimate(
        MG=float(mg),
        LB=float(lb),
        TwoNN=twonn(distances, cfg),
        degenerate=degenerate,
        floor_distance_fraction=found.floor_fraction,
        floor_sum_fraction=sum_floor_fraction,
        n_points=n,
        tau=rec.tau,
        theiler=rec.theiler,
    )


def twonn(distances: np.ndarray, cfg: EstimatorConfig) -> float:
    """Facco's two-nearest-neighbour estimate, from the same distances.

    It uses only the ratio ``mu = r_2 / r_1``, whose distribution under the Poisson model is
    Pareto with exponent ``d``, so it does not depend on the choice of ``m`` the way the
    pooled estimate does, and it is fitted on the empirical CDF rather than pooled, so a few
    degenerate points cannot dominate it. The top ``twonn_discard`` of the CDF is dropped
    before the fit: there the tail is thin and the fit would be set by a handful of points.
    """
    if distances.shape[1] < 2:
        return float("nan")
    r1, r2 = distances[:, 0], distances[:, 1]
    good = np.isfinite(r1) & (r1 > cfg.floor_distance)
    if int(good.sum()) < cfg.twonn_min_points:
        return float("nan")

    mu = np.sort(r2[good] / r1[good])
    m = len(mu)
    cut = int(m * (1.0 - cfg.twonn_discard))
    mu, cdf = mu[:cut], (np.arange(1, m + 1) / m)[:cut]
    usable = (mu > 1) & (cdf < 1)
    if int(usable.sum()) < 5:
        return float("nan")
    a, b = np.log(mu[usable]), -np.log1p(-cdf[usable])
    return float((a @ b) / (a @ a))  # a fit through the origin: the model has no intercept

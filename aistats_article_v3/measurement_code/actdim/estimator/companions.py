"""The statistics reported beside every estimate, so that none is reported alone.

Each is sensitive to an effect that can produce a dimension signal with no geometry behind
it, and each is cheaper than the estimate it accompanies:

* **roughness**, ``std(diff x) / std(x)``, needs no embedding at all. It measures departure
  from local smoothness, and on the article's real logs it moves with the estimate closely
  enough that a result which does not report it cannot claim the estimate saw anything else.
* the **linear participation ratio** of the delay covariance is the dimension of the smallest
  linear subspace the delay vectors lie near, weighted by energy. It is exactly 1 on a
  straight line and 2 on a sinusoid. On the article's data it recovers the active dimension
  better than the estimate does at eight directions, which is the strongest argument against
  reading the estimate as evidence of nonlinear manifold geometry.
* the **spectral participation ratio** counts resolvable lines in the power spectrum. It is
  the sharpest null available for a quasiperiodic system and the one a phase-randomised
  surrogate cannot provide: randomising the phases of an r-torus leaves an r-torus, so that
  surrogate tests determinism against a linear Gaussian process, not r directions against a
  broader spectrum. This counts modes directly.
* the **autocorrelation time** says how oversampled the record is, and it sizes the Theiler
  exclusion, so a run that quotes an estimate without it cannot say what was excluded.

None of them touches the neighbour search, which is why a window marked degenerate still has
all four -- see ``windows.summarise``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .config import EstimatorConfig
# One implementation, in the module that also uses it to size the Theiler exclusion. The
# archived tree had it in one place and four copies of the participation ratio in four.
from .embedding import autocorrelation_time  # noqa: F401  -- re-exported deliberately


def roughness(x: np.ndarray) -> float:
    """``std(diff x) / std(x)``: the null model for a dimension estimate.

    The finiteness check comes first because ``std`` of a series holding an infinity
    evaluates ``inf - inf`` and warns; a window like that has no roughness, and NaN says so.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all():
        return float("nan")
    spread = x.std()
    if spread <= 0.0:
        return float("nan")
    return float(np.diff(x).std() / spread)


def participation_ratio(weights: np.ndarray) -> float:
    """``(sum w)^2 / sum w^2`` for non-negative weights: an effective count of them.

    The one implementation. The archived tree had four, differing in whether they squared
    a singular value first, and two of them disagreed on the same data.
    """
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    squared = float((weights ** 2).sum())
    if not np.isfinite(total) or not np.isfinite(squared) or squared <= 0.0 or total <= 0.0:
        return float("nan")
    return total ** 2 / squared


def delay_participation_ratio(points: np.ndarray) -> float:
    """Participation ratio of the delay covariance's spectrum.

    Computed from the singular values of the centred delay matrix rather than by forming the
    covariance, which squares the condition number for nothing.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.size == 0 or not np.isfinite(points).all():
        return float("nan")
    centred = points - points.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    return participation_ratio(spectrum)


def spectral_participation_ratio(x: np.ndarray, bins: int = 256) -> float:
    """Participation ratio of the periodogram: an effective count of spectral lines.

    ``bins`` bands the spectrum before counting, and it is not a detail. 256 bands span
    [0, 0.5] in steps of 0.00195, which resolves a drive band at f0 = 1/16 and does not
    resolve one at f0 = 1/400; on the slow torus the banded statistic saturates at 2 whatever
    r is. ``bins=0`` keeps the native FFT resolution. The zero-frequency bin is dropped, so
    the statistic does not depend on the mean of the window.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all():
        return float("nan")
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    power = power[1:]
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0 or (bins and len(power) < bins):
        return float("nan")
    if bins:
        edges = np.linspace(0, len(power), bins + 1).astype(int)
        power = np.array([power[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    return participation_ratio(power)


def companion_statistics(x: np.ndarray, cfg: EstimatorConfig) -> Dict[str, Any]:
    """The statistics that need only the series: roughness, spectrum, autocorrelation."""
    out: Dict[str, Any] = {
        "roughness": roughness(x),
        "acorr": float(autocorrelation_time(np.asarray(x, dtype=np.float64))),
    }
    for bins in cfg.spectral_bins:
        out[f"specPR{bins}"] = spectral_participation_ratio(x, bins)
    return out


def companion_names(cfg: EstimatorConfig) -> Tuple[str, ...]:
    """The columns the companion statistics occupy, in the order they are written."""
    return (("PRdelay",) + tuple(f"specPR{b}" for b in cfg.spectral_bins)
            + ("roughness", "acorr"))

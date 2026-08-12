"""Three families with the same nominal rank and three different geometries.

Nothing about a network is asked of these. They exist to settle a question that comes
before any network: for a scalar observation of an ``r``-dimensional system, is ``r``
recoverable in principle, and what does the window have to look like for it to be? The
answer differs by family, and the three below are the three cases section 6.1 separates.

``quasiperiodic``
    deterministic motion on an ``r``-torus. Takens applies: the delay embedding of a
    generic scalar observation is a diffeomorphic copy of the torus, so the intrinsic
    dimension **is** ``r`` and the estimator has a right answer to find.
``ornstein_uhlenbeck``
    ``r`` independent one-pole processes. The delay vector is a function of the state
    *and* of the last ``E - 1`` innovations, so the cloud is full rank in ``R^E`` for
    every ``r``. There is no ``r``-manifold and no right answer.
``coloured``
    ``r`` band-limited processes, white noise through a cascade of one-pole filters.
    Between the two: the sample paths are smooth, and the state is still not
    finite-dimensional. It is the family that separates smoothness from geometry, which
    is what the roughness null exists to test.

Two things this module does not do, and one it fixes.

**It does not build its own drive.** The archived ``generators.py`` carried a second copy
of ``frequencies`` and ``resonance_margin``, and they were not the same functions as the
ones beside them. Its prime table stopped at 53, so ``frequencies`` would have raised
``IndexError`` above ``r = 16``; and its ``resonance_margin`` searched order 4 rather than
order 3, by full Cartesian enumeration over ``(2*order + 1) ** r`` integer vectors, which
is exponential in ``r`` and unusable past ``r = 8``. Both are replaced by
:mod:`actdim.systems.drive`, whose table runs to sixty primes, whose margin is exact at
order 3 over L1-bounded vectors, and whose frequency layout varies with the seed. The
margins reported here are therefore at order 3 and will not equal the archived ones.

**It registers nothing.** These are generators, not rungs of the ladder: no construction
here claims an active dimension that a network could have, so none belongs in
:data:`actdim.systems.spec.LADDER`. They are plain functions returning arrays.

**The observer no longer depends on the embedding dimension.** In the archived atlas one
generator drew the trajectory, the phases and the observer's projection from a single
generator seeded with ``10000*seed + 97*r + max_E + sum(map(ord, family))``. Because
``max_E`` entered that seed, the two halves of the identifiability ratio
``d(2E)/d(E)`` were computed on two *different realisations* of the system through two
different observers, when the ratio is defined as two embeddings of one window. Each draw
here comes from its own named stream keyed by the cell's seed alone, so a family, a rank
and a length fix the series and the embedding dimension is the only thing that varies.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from ..linalg import rank_report
from ..runtime.determinism import rng as stream_rng
from .drive import DEFAULT_BAND, centre_for_octave, frequencies, realised_band
from .drive import resonance_margin

#: The slowest drive frequency of the atlas, in cycles per sample. The torus arm is swept
#: over this and over the record length separately, so that "cycles of the slowest mode"
#: and "number of samples" can be told apart.
DEFAULT_F0 = 1.0 / 400.0

#: Correlation time of the two stochastic families, in samples.
DEFAULT_TAU_C = 200.0

#: One-pole stages in the coloured cascade. Sample paths are then ``order - 1`` times
#: differentiable, so the trajectory is smooth on the scale of ``tau_c`` while the state
#: stays infinite-dimensional -- which is the whole point of running it beside the torus.
DEFAULT_ORDER = 3

#: Relative singular-value threshold for the hard rank of a generated trajectory. The
#: archived ``state_rank`` used this value; a rank is a step function of its threshold, so
#: it is named here rather than inherited.
STATE_RANK_TOL = 1e-8


def quasiperiodic(r: int, n: int, seed: int, f0: float = DEFAULT_F0,
                  band: float = DEFAULT_BAND) -> Tuple[np.ndarray, Dict[str, Any]]:
    """An ``r``-torus: ``r`` rationally independent sinusoids filling one octave.

    ``f0`` is the slowest frequency in cycles per sample. The band does not widen with
    ``r``, which is requirement 6: if it did, the roughness of the observable would order
    the rank by itself and no embedding would be under test.
    """
    r, n = int(r), int(n)
    freqs = frequencies(r, centre_for_octave(f0, band), band=band, seed=seed)
    phases = stream_rng(seed, "drive_phases").uniform(0.0, 2.0 * np.pi, r)
    t = np.arange(n, dtype=float)
    X = np.sin(2.0 * np.pi * np.outer(t, freqs) + phases)
    meta = {"freqs": freqs, "margin": resonance_margin(freqs),
            "realised_band": realised_band(freqs), "f0": float(freqs.min()),
            "cycles": float(n * freqs.min()), "samples_per_cycle": float(1.0 / freqs.max())}
    return X, meta


def ornstein_uhlenbeck(r: int, n: int, seed: int, tau_c: float = DEFAULT_TAU_C
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """``r`` independent Ornstein-Uhlenbeck processes with unit stationary variance.

    ``innov_ratio = sqrt(1 - a^2)`` is the per-step innovation as a fraction of the
    stationary spread: the scale below which the delay cloud stops looking like a manifold
    and starts looking like a full-rank Gaussian.
    """
    from scipy.signal import lfilter

    r, n = int(r), int(n)
    burn = int(10 * tau_c)
    a = float(np.exp(-1.0 / tau_c))
    innovation = float(np.sqrt(1.0 - a * a))
    w = stream_rng(seed, "ou_innovations").standard_normal((n + burn, r)) * innovation
    X = lfilter([1.0], [1.0, -a], w, axis=0)[burn:]
    return X, {"tau_c": float(tau_c), "innov_ratio": innovation}


def coloured(r: int, n: int, seed: int, tau_c: float = DEFAULT_TAU_C,
             order: int = DEFAULT_ORDER) -> Tuple[np.ndarray, Dict[str, Any]]:
    """``r`` band-limited processes: white noise through ``order`` cascaded one-pole filters."""
    from scipy.signal import lfilter

    r, n = int(r), int(n)
    burn = int(10 * tau_c * order)
    a = float(np.exp(-1.0 / tau_c))
    X = stream_rng(seed, "coloured_innovations").standard_normal((n + burn, r))
    for _ in range(int(order)):
        X = lfilter([1.0 - a], [1.0, -a], X, axis=0)
    X = X[burn:]
    return X / (X.std(axis=0, keepdims=True) + 1e-12), {"tau_c": float(tau_c),
                                                        "order": int(order)}


#: The three families, under the names the atlas writes into its ``family`` column.
FAMILIES: Dict[str, Callable[..., Tuple[np.ndarray, Dict[str, Any]]]] = {
    "qp": quasiperiodic,
    "ou": ornstein_uhlenbeck,
    "colored": coloured,
}

#: How the scalar observers read a state, in the order the archived atlas defined them.
OBSERVERS: Tuple[str, ...] = ("linear", "generic", "norm", "normsq")


def observe(X: np.ndarray, seed: int, kind: str = "generic") -> np.ndarray:
    """A scalar function of the state, standardised.

    ``generic`` is a random linear functional plus a small quadratic term: generic in
    Takens' sense, and dominated by its linear part so that it does not manufacture the
    harmonics a pure square would. ``norm`` is the squared-norm observer the constructed
    systems use, kept so that the atlas and the ladder can be read on one axis.

    The projection is drawn from a stream named for the observer and keyed by the cell's
    seed alone, so that two embedding dimensions of one cell see the identical series.
    """
    X = np.asarray(X, dtype=float)
    generator = stream_rng(seed, "observer_projection")
    weights = generator.standard_normal(X.shape[1]) / np.sqrt(X.shape[1])
    z = X @ weights
    if kind == "linear":
        y = z
    elif kind == "generic":
        y = z + 0.2 * z ** 2
    elif kind == "norm":
        y = np.sqrt((X ** 2).sum(axis=1))
    elif kind == "normsq":
        y = (X ** 2).sum(axis=1)
    else:
        raise ValueError(f"unknown observer {kind!r}. Known: {', '.join(OBSERVERS)}")
    spread = float(y.std())
    return (y - y.mean()) / (spread + 1e-12)


def state_rank(X: np.ndarray) -> Tuple[int, float]:
    """Hard rank and effective rank of the state trajectory.

    The second is the measured active dimension, and it is what every estimate in the
    atlas is scored against -- never the nominal ``r``, which for the two stochastic
    families names an injected rank and not a property of the set the trajectory fills.
    """
    report = rank_report(X, center=True, tol=STATE_RANK_TOL)
    return report.rank, report.effective_rank


def generate(family: str, r: int, n: int, seed: int, **options: Any
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One cell of the atlas: the trajectory and what is known about it by construction."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}. Known: {', '.join(FAMILIES)}")
    return FAMILIES[family](r, n, seed, **options)

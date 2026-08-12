"""Three r-dimensional generators with the *same* nominal rank and different true geometry.

They exist to separate three things a delay embedding can be looking at:

``qp``      deterministic quasi-periodic motion on an r-torus.  Takens applies: the delay
            embedding of a generic scalar observation is a diffeomorphic copy of the torus,
            so the intrinsic dimension **is** r and MG has a right answer to find.
``ou``      r independent Ornstein-Uhlenbeck processes.  The delay vector is a function of
            the state *and of the last E-1 innovations*, so the cloud is full rank in R^E
            for every r >= 1.  There is no r-manifold and no right answer.
``colored`` r band-limited processes (white noise through a cascade of one-pole filters).
            Between the two: smooth sample paths, but still no finite-dimensional state.

Every frequency set is checked for low-order resonance, because a rational ratio silently
collapses the torus.  ``f0 * 2**linspace(0,1,r)`` -- the obvious matched-band construction --
puts the two extreme modes at an exact 2:1 ratio, which makes them one phase, not two, and
costs a whole dimension: measured MG for that construction is 1.4 at r=2 (truth 2) and 2.5 at
r=3 (truth 3, effective 2).
"""

from __future__ import annotations

import numpy as np
from itertools import product
from scipy.signal import lfilter

PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])


def frequencies(r, f0, band=2.0):
    """r rationally independent frequencies in [f0, f0*band].

    ``frac(sqrt(p))`` for distinct primes p is irrational and the ratios are irrational too,
    so no exact resonance exists; :func:`resonance_margin` reports how close the *low-order*
    ones come, which is what matters over a finite window.
    """
    a = np.sqrt(PRIMES[:r].astype(float)) % 1.0
    if r > 1:                                    # spread them over the band, keep irrational
        a = (a - a.min()) / (a.max() - a.min()) * 0.94 + 0.03
    else:
        a = np.array([0.5])
    return f0 * band ** a


def resonance_margin(f, order=4):
    """min over nonzero integer n with |n|_1 <= order of dist(sum n_j f_j, Z).

    Zero means the torus is really a lower-dimensional closed curve.  Small means the
    trajectory needs a window of order 1/margin before it looks r-dimensional.
    """
    r = len(f)
    best = np.inf
    for n in product(range(-order, order + 1), repeat=r):
        s = sum(abs(v) for v in n)
        if s == 0 or s > order:
            continue
        z = float(np.dot(n, f))
        best = min(best, abs(z - round(z)))
    return best


def qp(r, N, rng, f0=1 / 400.0, band=2.0):
    """r-torus.  ``f0`` is the slowest frequency in cycles per sample."""
    f = frequencies(r, f0, band)
    t = np.arange(N)
    ph = rng.uniform(0, 2 * np.pi, r)
    return np.sin(2 * np.pi * np.outer(t, f) + ph), dict(freqs=f, margin=resonance_margin(f))


def ou(r, N, rng, tau_c=200.0):
    """r independent OU processes with unit stationary variance.

    ``innov_ratio = sqrt(1 - a^2)`` is the per-step innovation as a fraction of the
    stationary spread: the scale below which the delay cloud stops looking like a manifold
    and starts looking like a full-rank Gaussian.
    """
    burn = int(10 * tau_c)
    a = np.exp(-1.0 / tau_c)
    s = np.sqrt(1 - a * a)
    w = rng.standard_normal((N + burn, r)) * s
    X = lfilter([1.0], [1.0, -a], w, axis=0)[burn:]
    return X, dict(tau_c=tau_c, innov_ratio=float(s))


def colored(r, N, rng, tau_c=200.0, order=3):
    """r band-limited processes: white noise through ``order`` cascaded one-pole filters.

    Sample paths are ``order - 1`` times differentiable, so the trajectory is smooth on the
    scale of tau_c -- but the state is still not finite-dimensional, which is the point of
    including it next to ``qp``.
    """
    burn = int(10 * tau_c * order)
    a = np.exp(-1.0 / tau_c)
    X = rng.standard_normal((N + burn, r))
    for _ in range(order):
        X = lfilter([1.0 - a], [1.0, -a], X, axis=0)
    X = X[burn:]
    return X / (X.std(0, keepdims=True) + 1e-12), dict(tau_c=tau_c, order=order)


FAMILIES = {"qp": qp, "ou": ou, "colored": colored}


# ------------------------------------------------------------------ scalar observers
def observe(X, rng, kind="generic"):
    """A scalar function of the state.  ``generic`` is a random linear functional plus a
    small quadratic term -- generic in Takens' sense, and dominated by the linear part so
    that it does not manufacture harmonics the way a pure square would."""
    w = rng.standard_normal(X.shape[1]) / np.sqrt(X.shape[1])
    z = X @ w
    if kind == "linear":
        y = z
    elif kind == "generic":
        y = z + 0.2 * z ** 2
    elif kind == "norm":                     # the "Frobenius norm" observer, sum of squares
        y = np.sqrt((X ** 2).sum(1))
    elif kind == "normsq":
        y = (X ** 2).sum(1)
    else:
        raise ValueError(kind)
    return (y - y.mean()) / (y.std() + 1e-12)


def state_rank(X, tol_ratio=1e-8):
    """Hard rank and participation ratio of the state trajectory: the *measured* active
    dimension, which is the quantity MG should be compared against."""
    C = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(C, compute_uv=False)
    s2 = s ** 2
    return int((s > s.max() * tol_ratio).sum()), float(s2.sum() ** 2 / (s2 ** 2).sum())

"""Scalar-record competitors of MG, all computed from the same one-dimensional window.

Every function takes a 1-D array and returns one number. Delay-space competitors use the
same delay construction as MG (z-scored window, lag from the autocorrelation time) so that
any difference comes from the statistic, not from the reconstruction.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from math import factorial
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

CFG_ACORR = EstimatorConfig(max_E=20, tau="acorr", k_neighbors=20, theiler="autocorr", theiler_cap=320)


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def acf_time(x, max_lag=2000):
    y = z(x)
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:min(max_lag, n)]
    a = a / a[0]
    below = np.where(a < 1 / np.e)[0]
    return int(below[0]) if len(below) else min(max_lag, n)


def delay(x, E=20, tau=None):
    y = z(x)
    if tau is None:
        tau = max(1, round(acf_time(y) / 4))
    n = len(y) - (E - 1) * tau
    return np.stack([y[i * tau:i * tau + n] for i in range(E)], 1), tau


def mg(x, cfg=CFG_ACORR):
    return estimate(np.asarray(x, float), cfg).MG


# ---- spectral / periodicity statistics (no delay space) ------------------------------

def spectral_entropy(x):
    y = z(x)
    p = np.abs(np.fft.rfft(y))[1:] ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(y) // 2))


def self_repeat(x, lo=20, hi=250):
    """Codex 'recurrence': best normalised self-repeat error over lags lo..hi."""
    y = np.asarray(x, float)
    v = y.var()
    hi = min(hi, len(y) // 2)
    return float(min(np.mean((y[p:] - y[:-p]) ** 2) / (2 * v) for p in range(lo, hi + 1)))


def roughness(x):
    y = np.asarray(x, float)
    return float(np.std(np.diff(y)) / np.std(y))


def peak_count(x, rel=0.05):
    y = z(x)
    P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    P = P / P[1:].sum()
    peaks = [i for i in range(2, len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    mass = sorted(((P[max(1, i - 3):i + 4].sum(), i) for i in peaks), reverse=True)
    kept = []
    for m, i in mass:
        if m < rel:
            break
        if all(abs(i - k) > 6 for k in kept):
            kept.append(i)
    return len(kept)


def perm_entropy(x, order=5, lag=None):
    y = np.asarray(x, float)
    if lag is None:
        lag = max(1, round(acf_time(y) / 4))
    n = len(y) - (order - 1) * lag
    pats = np.stack([y[i * lag:i * lag + n] for i in range(order)], 1).argsort(1)
    codes = (pats * (order ** np.arange(order))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(factorial(order)))


# ---- delay-space statistics -------------------------------------------------------------

def _sub(Y, m=2500, seed=0):
    if len(Y) <= m:
        return Y, np.arange(len(Y))
    idx = np.sort(np.random.default_rng(seed).choice(len(Y), m, replace=False))
    return Y[idx], idx


def _theiler_pairs(Y, idx, w):
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    t = np.abs(idx[:, None] - idx[None, :])
    iu = np.triu_indices(len(Y), 1)
    keep = t[iu] > w
    return d[iu][keep]


def recurrence_rate(x, E=20, tau=None, frac=0.1):
    """RQA recurrence rate: share of (Theiler-separated) pairs closer than frac x median distance."""
    Y, tau = delay(x, E, tau)
    Ys, idx = _sub(Y, 2000)
    d = _theiler_pairs(Ys, idx, (E - 1) * tau)
    return float(np.mean(d < frac * np.median(d)))


def corr_dim(x, E=20, tau=None, q=(0.01, 0.1)):
    """Grassberger-Procaccia slope of log C(r) between two quantiles of pair distances."""
    Y, tau = delay(x, E, tau)
    Ys, idx = _sub(Y, 2500)
    d = _theiler_pairs(Ys, idx, (E - 1) * tau)
    r = np.quantile(d, np.linspace(q[0], q[1], 8))
    C = np.array([np.mean(d < ri) for ri in r])
    return float(np.polyfit(np.log(r), np.log(C), 1)[0])


def twonn_fit(x, E=20, tau=None):
    """TwoNN (Facco 2017): slope through origin of -log(1-F) against log(mu)."""
    Y, tau = delay(x, E, tau)
    w = (E - 1) * tau
    tree = cKDTree(Y)
    dd, ii = tree.query(Y, k=min(2 * w + 6, len(Y)))
    mu = []
    for i in range(len(Y)):
        r = dd[i][np.abs(ii[i] - i) > w]
        if len(r) >= 2 and r[0] > 0:
            mu.append(r[1] / r[0])
    mu = np.sort(np.array(mu))
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = slice(0, int(0.9 * n))
    a, b = np.log(mu[keep]), -np.log(1 - F[keep])
    return float((a @ b) / (a @ a))


def linear_pr(x, E=20, tau=None):
    Y, _ = delay(x, E, tau)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Y.T)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


ALL = {"MG": mg, "spectral_entropy": spectral_entropy, "self_repeat": self_repeat,
       "roughness": roughness, "peak_count": peak_count, "perm_entropy": perm_entropy,
       "recurrence_rate": recurrence_rate, "corr_dim": corr_dim, "twonn": twonn_fit,
       "linear_pr": linear_pr}

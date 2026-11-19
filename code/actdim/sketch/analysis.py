"""Statistics for trajectories and trajectory sketches."""
from __future__ import annotations

from typing import Any

import numpy as np


def pr(X: Any, tol: float = 1e-12) -> float:
    """Participation ratio ``(sum lambda)^2 / sum lambda^2`` of the row cloud of ``X``.

    One for a cloud along a single direction, ``r`` for ``r`` equally weighted ones. Returns
    NaN below three rows, where a covariance has no shape to report, and on a cloud that
    does not move.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return float("nan")
    s = np.linalg.svd(X, compute_uv=False) ** 2
    total = s.sum()
    return float(total * total / (s * s).sum()) if total > tol else float("nan")


def detrend(X: Any) -> np.ndarray:
    """Remove a per-coordinate linear trend and the mean.

    A steady drift is a rank-one contribution that would otherwise dominate every window.
    """
    X = np.asarray(X, dtype=float)
    t = np.arange(len(X), dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    beta = (t[:, None] * X).sum(0) / (t @ t)
    return X - np.outer(t, beta) - X.mean(0, keepdims=True)


def block_mean(X: Any, m: int) -> np.ndarray:
    """Average consecutive blocks of ``m`` rows, dropping the incomplete tail."""
    X = np.asarray(X, dtype=float)
    n = (len(X) // m) * m
    return X[:n].reshape(-1, m, X.shape[1]).mean(1) if n >= m else X[:0]

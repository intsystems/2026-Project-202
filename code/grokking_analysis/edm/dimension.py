"""Effective / intrinsic dimension estimators for reconstructed phase spaces.

Four estimators, ranked in the paper (Sec. 3.2) by robustness to mini-batch noise:

* :func:`mle_intrinsic_dimension` -- Levina-Bickel maximum likelihood estimator.
  The primary tool of the paper: continuous-valued, local, no global recurrence
  required (Alg. 1 in the appendix).
* :func:`simplex_projection` -- Sugihara-May simplex projection.
* :func:`cao_method` -- Cao's averaged false-neighbour ratio.
* :func:`false_nearest_neighbors` -- classical FNN.

Every estimator dithers the input with negligible (1e-9 .. 1e-10) Gaussian noise
to break exact ties in the KD-tree; pass an explicit ``rng`` for reproducibility.
"""

import numpy as np
from sklearn.neighbors import KDTree

from .embedding import delay_embedding

DEGENERATE_STD = 1e-6
"""A window whose std falls below this is treated as a degenerate observable (E = 1)."""


def _dither(series, scale, rng):
    rng = np.random.default_rng() if rng is None else rng
    series = np.asarray(series, dtype=np.float64)
    return series + rng.normal(0.0, scale, size=len(series))


def mle_intrinsic_dimension(series, tau=1, max_E=15, k_neighbors=5, rng=None):
    """Levina-Bickel MLE of the intrinsic dimension of a delay-embedded series.

    The series is projected into a deliberately redundant space ``R^max_E`` and
    the local dimension at every point is estimated from the log-ratios of the
    distances to its ``k`` nearest neighbours::

        d_x = [ 1/(k-1) * sum_{j=1}^{k-1} log(r_k / r_j) ]^{-1}

    The returned value is the mean of the local estimates.
    """
    series = _dither(series, 1e-9, rng)

    try:
        embedded = delay_embedding(series, max_E, tau)
    except ValueError:
        return np.nan

    if len(embedded) < k_neighbors + 2:
        return np.nan

    tree = KDTree(embedded)
    distances, _ = tree.query(embedded, k=k_neighbors + 1)
    distances = np.maximum(distances[:, 1:], 1e-8)  # drop the point itself

    r_k = distances[:, -1:]
    sum_log_ratios = np.sum(np.log(r_k / distances[:, :-1]), axis=1)
    sum_log_ratios = np.maximum(sum_log_ratios, 1e-5)

    local_id = (k_neighbors - 1) / sum_log_ratios
    local_id = local_id[np.isfinite(local_id)]
    if len(local_id) == 0:
        return np.nan

    global_id = float(np.mean(local_id))
    return float(max_E) if global_id > max_E * 2 else global_id


def false_nearest_neighbors(series, tau=1, max_m=10, Rtol=10.0, Atol=2.0, rng=None):
    """Percentage of false nearest neighbours for embedding dimensions 1..max_m."""
    series = _dither(series, 1e-10, rng)
    sigma = np.std(series)
    fnn_percent = []

    for m in range(1, max_m + 1):
        try:
            embedded = delay_embedding(series, m, tau)
        except ValueError:
            fnn_percent.append(0.0)
            continue

        tree = KDTree(embedded)
        distances, indices = tree.query(embedded, k=2)
        r_d, nn_idx = distances[:, 1], indices[:, 1]

        max_idx = len(series) - m * tau
        valid = (np.arange(len(embedded)) < max_idx) & (nn_idx < max_idx)
        if not np.any(valid):
            fnn_percent.append(0.0)
            continue

        r_d, nn_idx = r_d[valid], nn_idx[valid]
        current = np.arange(len(embedded))[valid]

        dist_increase = np.abs(series[current + m * tau] - series[nn_idx + m * tau])
        r_d1 = np.sqrt(r_d ** 2 + dist_increase ** 2)

        eps = 1e-10
        is_false = ((dist_increase / (r_d + eps)) > Rtol) | ((r_d1 / (sigma + eps)) > Atol)
        fnn_percent.append(100.0 * np.sum(is_false) / len(is_false))

    return np.array(fnn_percent)


def cao_method(series, tau=1, max_E=15, plateau_tol=0.05, rng=None):
    """Cao's method. Returns ``(optimal_E, E1_curve)``."""
    series = _dither(series, 1e-10, rng)
    e_values = []

    for d in range(1, max_E + 1):
        try:
            emb_d_plus_1 = delay_embedding(series, d + 1, tau)
            length = len(emb_d_plus_1)
            emb_d = delay_embedding(series, d, tau)[:length]
        except ValueError:
            break
        if length < 10:
            break

        tree = KDTree(emb_d, metric="chebyshev")
        distances, indices = tree.query(emb_d, k=2)
        r_d = np.maximum(distances[:, 1], 1e-10)
        nn_indices = indices[:, 1]

        diff = np.abs(series[np.arange(length) + d * tau] - series[nn_indices + d * tau])
        e_values.append(np.mean(np.maximum(r_d, diff) / r_d))

    e1 = [e_values[i] / e_values[i - 1] for i in range(1, len(e_values))]
    if not e1:
        return 1, [1.0]

    plateau = np.where(np.abs(np.diff(e1)) < plateau_tol)[0]
    optimal_E = int(plateau[0]) + 2 if len(plateau) > 0 else max_E
    return optimal_E, [1.0] + e1


def simplex_projection(series, tau=1, max_E=15, Tp=1, train_fraction=0.7):
    """Simplex projection (Sugihara-May). Returns ``(optimal_E, rmse_per_E)``."""
    series = np.asarray(series, dtype=np.float64)
    rmse_per_E = []

    for E in range(1, max_E + 1):
        n_embed = len(series) - (E - 1) * tau - Tp
        if n_embed < 10:
            break

        X = np.array([series[i:i + E * tau:tau] for i in range(n_embed)])
        Y = np.array([series[i + (E - 1) * tau + Tp] for i in range(n_embed)])

        split = int(n_embed * train_fraction)
        X_lib, Y_lib = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]

        predictions = []
        for target in X_test:
            distances = np.linalg.norm(X_lib - target, axis=1)
            nearest = np.argsort(distances)[:E + 1]
            nearest_dists = distances[nearest]
            weights = np.exp(-nearest_dists / (nearest_dists[0] + 1e-8))
            weights /= np.sum(weights)
            predictions.append(np.sum(weights * Y_lib[nearest]))

        rmse_per_E.append(np.sqrt(np.mean((np.array(predictions) - Y_test) ** 2)))

    if not rmse_per_E:
        return 1, []
    return int(np.argmin(rmse_per_E)) + 1, rmse_per_E


# --- Uniform (series, tau) -> float adapters used by the sliding-window driver ---

def estimate_E_mle(series, tau, max_E=15, k_neighbors=5, rng=None):
    if np.std(series) < DEGENERATE_STD:
        return 1.0
    return mle_intrinsic_dimension(series, tau, max_E=max_E, k_neighbors=k_neighbors, rng=rng)


def estimate_E_fnn(series, tau, max_m=15, rng=None):
    if np.std(series) < DEGENERATE_STD:
        return 1.0
    fnn = false_nearest_neighbors(series, tau=tau, max_m=max_m, rng=rng)
    below = np.where(fnn < 1.0)[0]
    return float(below[0] + 1 if len(below) > 0 else np.argmin(fnn) + 1)


def estimate_E_cao(series, tau, max_E=15, rng=None):
    if np.std(series) < DEGENERATE_STD:
        return 1.0
    return float(cao_method(series, tau, max_E=max_E, rng=rng)[0])


def estimate_E_simplex(series, tau, max_E=15, rng=None):
    if np.std(series) < DEGENERATE_STD:
        return 1.0
    return float(simplex_projection(series, tau, max_E=max_E)[0])


ESTIMATORS = {
    "mle": ("MLE Intrinsic Dimension", estimate_E_mle),
    "fnn": ("Classic FNN", estimate_E_fnn),
    "cao": ("Cao Method E1", estimate_E_cao),
    "simplex": ("Simplex Projection", estimate_E_simplex),
}

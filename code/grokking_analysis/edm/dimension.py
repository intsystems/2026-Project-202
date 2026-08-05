"""Effective / intrinsic dimension estimators for reconstructed phase spaces.

Four estimators, ranked in the paper (Sec. 3.2) by robustness to mini-batch noise:

* :func:`mle_intrinsic_dimension` -- Levina-Bickel maximum likelihood estimator.
  The primary tool of the paper (Alg. 1 in the appendix). Supports the
  MacKay-Ghahramani pooling correction (see :data:`CORRECTIONS`) and a Theiler
  window, which excludes neighbours that are close in time rather than in phase
  space -- without it an oversampled trajectory reports the dimension of its own
  tangent line (1.227 for k = 5) instead of the dimension of the attractor.
* :func:`simplex_projection` -- Sugihara-May simplex projection.
* :func:`cao_method` -- Cao's averaged false-neighbour ratio.
* :func:`false_nearest_neighbors` -- classical FNN.

Every estimator dithers the input with negligible (1e-9 .. 1e-10) Gaussian noise
to break exact ties in the KD-tree; pass an explicit ``rng`` for reproducibility.
"""

from typing import Callable, NamedTuple

import numpy as np
from sklearn.neighbors import KDTree

from .embedding import autocorrelation_time, delay_embedding

DEGENERATE_STD = 1e-6
"""A window whose std falls below this is treated as a degenerate observable (E = 1)."""

CORRECTIONS = ("levina_bickel", "mackay_ghahramani")
"""How the per-point log-ratio sums are pooled into a single estimate.

Under the Poisson model both estimators share the same per-point statistic

    S_i = sum_{j=1}^{k-1} log( r_k(x_i) / r_j(x_i) ),   d * S_i ~ Gamma(k-1, 1),

so the local estimate is ``d_i = (k-1) / S_i``.

``levina_bickel``
    The original paper averages the local estimates, ``d = mean_i d_i``. Because
    ``E[1/S_i] = d/(k-2)`` rather than ``d/(k-1)``, every local estimate is biased
    upwards by ``(k-1)/(k-2)`` -- 33% at ``k = 5`` -- and the arithmetic mean is
    dominated by the few points with anomalously small ``S_i``.

``mackay_ghahramani``
    MacKay and Ghahramani (2005) point out that the points share one ``d``, so the
    likelihood should be pooled *before* inverting::

        d = ( n(k-1) - 1 ) / sum_i S_i

    which is the exact unbiased ML estimate for the pooled Gamma sample. For large
    ``n`` this equals the inverse of the average of ``1/d_i`` (a harmonic mean of
    the local estimates), so by AM-HM it never exceeds the Levina-Bickel value.
"""


class Estimator(NamedTuple):
    """One dimension estimator: a long name for titles, a short one for legends."""

    name: str
    label: str
    fn: Callable


def _dither(series, scale, rng):
    """Break exact KD-tree ties with negligible noise.

    ``scale=None`` skips it entirely: the estimators then drop degenerate points
    instead of perturbing the data, which is the numerically honest choice. The
    default reproduces the original notebooks.
    """
    series = np.asarray(series, dtype=np.float64)
    if scale is None:
        return series
    rng = np.random.default_rng() if rng is None else rng
    return series + rng.normal(0.0, scale, size=len(series))


def resolve_theiler_window(theiler_window, series, tau, max_E):
    """Turn ``0`` / ``"embedding"`` / ``"autocorr"`` into a concrete lag."""
    if theiler_window in (None, 0, "none"):
        return 0
    if theiler_window == "embedding":
        return (max_E - 1) * tau
    if theiler_window == "autocorr":
        return max((max_E - 1) * tau, autocorrelation_time(series))
    return int(theiler_window)


def _neighbor_distances(embedded, k_neighbors, theiler_window, floor=1e-8):
    """Distances to the ``k`` nearest neighbours, excluding temporal ones.

    With ``theiler_window = W`` every candidate ``j`` with ``|i - j| <= W`` is
    dropped, so the estimate is built from genuine recurrences of the trajectory
    rather than from the points immediately before and after ``i``.
    """
    n_points = len(embedded)
    tree = KDTree(embedded)

    if theiler_window <= 0:
        if n_points < k_neighbors + 2:
            return None
        distances, _ = tree.query(embedded, k=k_neighbors + 1)
        distances = distances[:, 1:]  # drop the point itself
        return distances if floor is None else np.maximum(distances, floor)

    # At most 2W+1 candidates per point are excluded, so this many always leaves k.
    excluded = 2 * theiler_window + 1
    if n_points - excluded < k_neighbors:
        return None

    distances, indices = tree.query(embedded, k=min(n_points, k_neighbors + excluded))
    valid = np.abs(indices - np.arange(n_points)[:, None]) > theiler_window

    # KDTree returns ascending distances, so a stable sort on ~valid keeps the
    # nearest valid neighbours first without re-sorting by distance.
    order = np.argsort(~valid, axis=1, kind="stable")
    distances = np.take_along_axis(distances, order, axis=1)[:, :k_neighbors]
    return distances if floor is None else np.maximum(distances, floor)


def mle_log_ratio_sums(
    series, tau=1, max_E=15, k_neighbors=5, theiler_window=0, dither=1e-9, rng=None
):
    """Per-point statistics ``S_i = sum_j log(r_k / r_j)`` of the Levina-Bickel model.

    Returns ``None`` when the window is too short to embed (or too short for the
    requested Theiler exclusion). This is the shared sufficient statistic of both
    poolings in :data:`CORRECTIONS`.

    With ``dither=None`` no noise is added and no distance floor is applied;
    points whose neighbours are exactly coincident are dropped instead, so the
    returned array may be shorter than the embedded cloud.
    """
    exact = dither is None
    series = _dither(series, dither, rng)
    theiler_window = resolve_theiler_window(theiler_window, series, tau, max_E)

    try:
        embedded = delay_embedding(series, max_E, tau)
    except ValueError:
        return None

    distances = _neighbor_distances(
        embedded, k_neighbors, theiler_window, floor=None if exact else 1e-8
    )
    if distances is None:
        return None

    if exact:
        distances = distances[np.all(distances > 0.0, axis=1)]
        if len(distances) == 0:
            return None

    r_k = distances[:, -1:]
    sum_log_ratios = np.sum(np.log(r_k / distances[:, :-1]), axis=1)
    if not exact:
        return np.maximum(sum_log_ratios, 1e-5)
    return sum_log_ratios[sum_log_ratios > 0.0]


def mle_intrinsic_dimension(
    series, tau=1, max_E=15, k_neighbors=5, correction="levina_bickel",
    theiler_window=0, dither=1e-9, clamp_to_max_E=True, rng=None,
):
    """Maximum-likelihood intrinsic dimension of a delay-embedded series.

    The series is projected into a deliberately redundant space ``R^max_E`` and the
    local dimension at every point is estimated from the log-ratios of the distances
    to its ``k`` nearest neighbours::

        d_x = [ 1/(k-1) * sum_{j=1}^{k-1} log(r_k / r_j) ]^{-1}

    ``correction`` selects how those local estimates are pooled -- see
    :data:`CORRECTIONS` for the difference between the original Levina-Bickel
    average and the MacKay-Ghahramani correction.

    ``theiler_window`` (Theiler 1986) excludes neighbours that are close in *time*
    rather than in phase space: ``0`` disables it, ``"embedding"`` uses the span of
    one delay vector ``(max_E - 1) * tau``, ``"autocorr"`` widens that to the
    autocorrelation time of the window, and an integer sets the lag directly.
    Without it, an oversampled trajectory returns the dimension of its own tangent
    line instead of the dimension of the attractor.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"unknown correction '{correction}'. Expected one of {CORRECTIONS}")

    sum_log_ratios = mle_log_ratio_sums(
        series, tau, max_E, k_neighbors, theiler_window=theiler_window,
        dither=dither, rng=rng,
    )
    if sum_log_ratios is None or len(sum_log_ratios) == 0:
        return np.nan

    if correction == "levina_bickel":
        local_id = (k_neighbors - 1) / sum_log_ratios
        local_id = local_id[np.isfinite(local_id)]
        if len(local_id) == 0:
            return np.nan
        global_id = float(np.mean(local_id))
    else:
        total = float(np.sum(sum_log_ratios))
        n_points = len(sum_log_ratios)
        if not np.isfinite(total) or total <= 0:
            return np.nan
        global_id = (n_points * (k_neighbors - 1) - 1) / total

    if clamp_to_max_E and global_id > max_E * 2:
        return float(max_E)
    return global_id


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


def local_svd_dimension(segment, tau=1, embedding=None, measure="participation_ratio",
                        center=True, **_ignored):
    """Effective number of active directions in a *short* trajectory segment.

    This answers a different question from :func:`mle_intrinsic_dimension`. The MLE
    estimates the dimension of the invariant set the trajectory fills, which needs
    the trajectory to return near its past states. This measures how many
    independent directions the optimizer actually explores over the segment itself
    -- Broomhead and King's singular-system analysis of a delay embedding (1986),
    the classical tool for short and transient records.

    The delay-embedded segment is centred and its singular spectrum taken; the
    normalised squared spectrum ``p_i`` is summarised as either the participation
    ratio ``1 / sum p_i^2`` or the entropy rank ``exp(-sum p_i log p_i)``. Both are
    continuous and need no recurrence, so they are defined on ~10 samples where a
    neighbour-based estimate is not.

    Reference values (exact, not empirical): a straight ramp gives 1, a pure
    sinusoid 2, and white noise the full embedding dimension. Because a redundant
    embedding is *harmless* here -- the SVD sorts the directions out -- ``tau=1``
    with ``embedding = len(segment) // 2`` is the right default, the opposite of
    what the neighbour-based estimators want.

    Three limits, all of which matter for how the number may be described:

    * **It is a linear measure.** The SVD gives the dimension of the smallest
      linear subspace containing the segment, not of the manifold it lies on. A
      circle is a 1-manifold but scores 2 -- which is exactly why a sinusoid gives
      2 above. Read it as an effective rank, never as a fractal dimension.
    * **It is energy-weighted, so it is not even the rank.** Two sinusoids of
      unequal amplitude have rank 4 but score ~2.4.
    * **The value depends on the embedding.** Mean PR on the ``s5_wd1`` weight norm
      moves 1.27 -> 1.35 -> 1.46 as ``embedding`` goes W/4 -> W/3 -> W/2, so only
      changes in it are meaningful, not the level.

    On these logs it is also ~0.95 Spearman-correlated with :func:`local_roughness`,
    which detects the same transition slightly earlier -- see that function.
    """
    segment = np.asarray(segment, dtype=np.float64)
    E = max(2, len(segment) // 2) if embedding is None else embedding

    try:
        embedded = delay_embedding(segment, E, tau)
    except ValueError:
        return np.nan
    if len(embedded) < 2:
        return np.nan

    if center:
        embedded = embedded - embedded.mean(axis=0, keepdims=True)

    spectrum = np.linalg.svd(embedded, compute_uv=False) ** 2
    total = spectrum.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan

    p = spectrum / total
    if measure == "participation_ratio":
        return float(1.0 / np.sum(p ** 2))
    if measure == "entropy":
        p = p[p > 0]
        return float(np.exp(-np.sum(p * np.log(p))))
    raise ValueError(f"unknown measure '{measure}'. Expected 'participation_ratio' or 'entropy'")


def local_roughness(segment, **_ignored):
    """Fraction of a segment's variation that a straight line does *not* explain.

    The null model for :func:`local_svd_dimension`. It uses no embedding and no
    SVD -- just ``std(residual after a linear fit) / std(segment)`` -- yet on the
    weight-norm logs it is 0.79-0.95 Spearman-correlated with the participation
    ratio and fires 250-450 steps *earlier* with the same silence on the WD=0
    controls. Keep it in any comparison: if a singular spectrum cannot beat this,
    the phenomenon being measured is departure from local linearity, and should be
    described that way rather than as a dimension.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if len(segment) < 3:
        return np.nan
    spread = segment.std()
    if not np.isfinite(spread) or spread <= 0:
        return np.nan
    t = np.arange(len(segment), dtype=np.float64)
    residual = segment - np.polyval(np.polyfit(t, segment, 1), t)
    return float(residual.std() / spread)


def embedding_dimension_scan(
    series, max_E_values=(5, 10, 15, 20, 25, 30), tau=1, k_neighbors=5,
    correction="mackay_ghahramani", theiler_window="embedding", seed=0,
):
    """Estimate the dimension at several embedding dimensions ``max_E``.

    An intrinsic dimension only exists if the estimate is insensitive to the size
    of the space it is measured in. On a genuine low-dimensional attractor the
    scan is flat; on a point cloud with no resolvable manifold the estimate simply
    tracks ``max_E``. See :func:`identifiability_ratio`.
    """
    return {
        max_E: mle_intrinsic_dimension(
            series, tau=tau, max_E=max_E, k_neighbors=k_neighbors,
            correction=correction, theiler_window=theiler_window,
            rng=np.random.default_rng(seed),
        )
        for max_E in max_E_values
    }


def identifiability_ratio(series, max_E=10, **kwargs):
    """``E(2 * max_E) / E(max_E)`` -- how much doubling the embedding space moves the answer.

    A ratio near 1 means the estimate is a property of the data; a ratio near 2
    means it is a property of the embedding, i.e. no dimension is identifiable at
    this sample size. Lorenz-63 over 2400+ samples gives ~1.0; the weight-norm
    logs of this project give ~2.0 at every available window length.
    """
    scan = embedding_dimension_scan(series, max_E_values=(max_E, 2 * max_E), **kwargs)
    low, high = scan[max_E], scan[2 * max_E]
    if not np.isfinite(low) or low <= 0 or not np.isfinite(high):
        return np.nan
    return high / low


# --- Uniform (series, tau) -> float adapters used by the sliding-window driver ---

def estimate_E_mle(
    series, tau, max_E=15, k_neighbors=5, correction="levina_bickel", theiler_window=0,
    dither=1e-9, clamp_to_max_E=True, degenerate=1.0, rng=None,
):
    """``degenerate`` is what a variance-free window reports; pass ``nan`` to
    leave it out of the plot rather than drawing a fabricated ``E = 1``."""
    if np.std(series) < DEGENERATE_STD:
        return degenerate
    return mle_intrinsic_dimension(
        series, tau, max_E=max_E, k_neighbors=k_neighbors, correction=correction,
        theiler_window=theiler_window, dither=dither, clamp_to_max_E=clamp_to_max_E, rng=rng,
    )


def estimate_E_mle_mg(series, tau, **kwargs):
    kwargs.setdefault("correction", "mackay_ghahramani")
    return estimate_E_mle(series, tau, **kwargs)


def estimate_E_mle_mg_theiler(series, tau, **kwargs):
    """MacKay-Ghahramani pooling with temporal neighbours excluded."""
    kwargs.setdefault("theiler_window", "embedding")
    return estimate_E_mle_mg(series, tau, **kwargs)


def estimate_E_mle_mg_theiler_acf(series, tau, **kwargs):
    """As above, but the exclusion widens to the window's autocorrelation time."""
    kwargs["theiler_window"] = "autocorr"
    return estimate_E_mle_mg(series, tau, **kwargs)


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


def estimate_E_svd(series, tau, measure="participation_ratio", **kwargs):
    """Local singular-spectrum dimension; ``tau`` is ignored (a redundant embedding
    is the correct choice here, see :func:`local_svd_dimension`)."""
    if np.std(series) < DEGENERATE_STD:
        return kwargs.get("degenerate", 1.0)
    return local_svd_dimension(series, tau=1, measure=measure,
                               embedding=kwargs.get("embedding"))


def estimate_E_svd_entropy(series, tau, **kwargs):
    return estimate_E_svd(series, tau, measure="entropy", **kwargs)


def estimate_E_roughness(series, tau, **kwargs):
    """Null-model baseline; ``tau`` is ignored (no embedding is used)."""
    return local_roughness(series)


def estimate_E_simplex(series, tau, max_E=15, rng=None):
    if np.std(series) < DEGENERATE_STD:
        return 1.0
    return float(simplex_projection(series, tau, max_E=max_E)[0])


ESTIMATORS = {
    "mle": Estimator("MLE Intrinsic Dimension", "MLE, Levina-Bickel", estimate_E_mle),
    "mle_mg": Estimator(
        "MLE Intrinsic Dimension (MacKay-Ghahramani)", "MLE, MacKay-Ghahramani", estimate_E_mle_mg
    ),
    "mle_mg_theiler": Estimator(
        "MLE ID (MacKay-Ghahramani + Theiler)", "MLE, MG + Theiler", estimate_E_mle_mg_theiler
    ),
    "mle_mg_theiler_acf": Estimator(
        "MLE ID (MacKay-Ghahramani + Theiler, ACF)", "MLE, MG + Theiler (ACF)",
        estimate_E_mle_mg_theiler_acf,
    ),
    "fnn": Estimator("Classic FNN", "FNN", estimate_E_fnn),
    "cao": Estimator("Cao Method E1", "Cao", estimate_E_cao),
    "simplex": Estimator("Simplex Projection", "Simplex", estimate_E_simplex),
    "svd": Estimator("Local SVD participation ratio", "local SVD (PR)", estimate_E_svd),
    "svd_entropy": Estimator("Local SVD entropy rank", "local SVD (entropy)",
                             estimate_E_svd_entropy),
    "roughness": Estimator("Departure from local linearity", "roughness (baseline)",
                           estimate_E_roughness),
}

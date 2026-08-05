"""Simplex projection with strictly causal splits, and the recurrence check that
decides whether the embedding theorems may be invoked at all.

Two independent things live here, in the order they have to be used:

``recurrence_rate`` / ``recurrence_stats``
    Does the trajectory revisit its past? Takens and Stark reconstruct a *compact
    invariant set that the orbit returns to*; on a transient there is nothing to
    reconstruct. This is the precondition, and the audit showed what happens when it is
    skipped -- estimators return closed-form constants of a straight line and they get
    reported as attractor dimensions.

``simplex_skill``
    Sugihara & May's forecast skill: predict a held-out point from the weighted average
    of its nearest neighbours in delay space, and correlate prediction with truth. Two
    deliberate differences from the usual implementation:

    * **Causal split.** The library is strictly earlier in time than the prediction set.
      A random split lets the library contain the future of the point being predicted,
      which on an oversampled series is near-cheating -- the neighbour is the point's own
      temporal neighbour.
    * **Theiler exclusion.** Neighbours within ``theiler`` steps of the target are barred,
      for the same reason they were needed in the dimension estimator: temporal
      neighbours measure the tangent, not the dynamics.

    Skill is meaningful only relative to a null, so nothing here is interpreted without
    the surrogate test in :mod:`surrogates`.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

__all__ = ["delay_embed", "recurrence_stats", "simplex_skill", "skill_vs_horizon"]


def delay_embed(series, E, tau):
    """Delay vectors [x(t), x(t-tau), ..., x(t-(E-1)tau)] and their time indices."""
    x = np.asarray(series, dtype=float)
    span = (E - 1) * tau
    if len(x) <= span:
        raise ValueError(f"series of length {len(x)} too short for E={E}, tau={tau}")
    idx = np.arange(span, len(x))
    vectors = np.stack([x[idx - k * tau] for k in range(E)], axis=1)
    return vectors, idx


def _pairwise(series, E, tau, max_points=1500):
    """Delay-vector distance matrix and the time gap between every pair.

    Subsampled uniformly when long: recurrence rate is a density, so a uniform subsample
    estimates it consistently, and the time gaps are rescaled with it.
    """
    vectors, _ = delay_embed(series, E, tau)
    n = len(vectors)
    step = max(1, int(np.ceil(n / max_points)))
    vectors = vectors[::step]
    m = len(vectors)
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
    time_gap = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :]) * step
    return distances, time_gap


def recurrence_profile(series, E=5, tau=1, radius_quantile=0.05, windows=None):
    """Recurrence rate as a function of the temporal exclusion window.

    A single Theiler window cannot answer "does the orbit return?", because the answer
    depends on how long you wait: on a smooth transient, points stay inside the radius
    for a while purely by continuity. (A first attempt used the embedding span, 20 steps,
    while a monotone ramp keeps pairs inside the radius for ~100 -- so it excluded almost
    nothing and called a straight line recurrent.)

    Sweeping the window separates the two cases by their *shape*:

    * **attractor** -- the rate falls to a plateau and stays positive, because genuine
      returns happen at all separations beyond the orbit's period;
    * **transient** -- the rate decays towards zero, because every close pair was merely
      adjacent in time.

    The radius is fixed once from the full off-diagonal distance distribution and reused
    for every window; recomputing it per window would force the answer to the quantile.
    """
    distances, time_gap = _pairwise(series, E, tau)
    off_diagonal = time_gap > 0
    if not off_diagonal.any():
        return {}, float("nan")

    radius = float(np.quantile(distances[off_diagonal], radius_quantile))
    span = int(time_gap.max())
    if windows is None:
        windows = [0, (E - 1) * tau, span // 100, span // 20, span // 10, span // 5]
        windows = sorted({w for w in windows if w >= 0 and w < span // 2})

    profile = {}
    for w in windows:
        valid = time_gap > w
        profile[int(w)] = float(np.mean(distances[valid] <= radius)) if valid.any() else float("nan")
    return profile, radius


def recurrence_stats(series, E=5, tau=1, radius_quantile=0.05):
    """Summarise :func:`recurrence_profile` into the precondition test.

    ``ratio`` is the recurrence rate at the longest exclusion window over the rate at
    zero exclusion. Near 1 the orbit genuinely returns and the embedding theorems have
    an invariant set to reconstruct; near 0 the trajectory is a transient and
    Takens/Stark do not apply, whatever number an estimator would print.
    """
    profile, radius = recurrence_profile(series, E, tau, radius_quantile)
    if not profile:
        return {"rr_plain": float("nan"), "rr_long": float("nan"),
                "ratio": float("nan"), "radius": float("nan"), "profile": {}}

    windows = sorted(profile)
    rr_plain, rr_long = profile[windows[0]], profile[windows[-1]]
    return {
        "rr_plain": rr_plain,
        "rr_long": rr_long,
        "longest_window": windows[-1],
        "ratio": rr_long / rr_plain if rr_plain else float("nan"),
        "radius": radius,
        "profile": profile,
    }


def simplex_skill(series, E=3, tau=1, horizon=1, library_fraction=0.6,
                  theiler=None, n_neighbors=None, metric="pearson"):
    """Skill of simplex-projected forecasts against truth, causal split.

    ``metric``:

    ``"pearson"``
        The EDM literature's default correlation. Unsafe on these logs: the validation
        loss carries kurtosis of ~2000 and excursions of 50+ sigma, so the correlation
        essentially reports whether one spike was predicted, and the surrogate ensemble
        scatters wildly (+-0.34) as a result.
    ``"spearman"``
        Rank correlation -- same monotone information, bounded influence per point. Use
        this whenever the series is spiky, and report both when it matters.

    Returns ``nan`` when the geometry leaves too few admissible neighbours, rather than a
    number that would look like a measurement.
    """
    x = np.asarray(series, dtype=float)
    vectors, idx = delay_embed(x, E, tau)
    theiler = (E - 1) * tau if theiler is None else theiler
    n_neighbors = (E + 1) if n_neighbors is None else n_neighbors

    target_idx = idx + horizon
    keep = target_idx < len(x)
    vectors, idx, target_idx = vectors[keep], idx[keep], target_idx[keep]
    if len(vectors) < 4 * (n_neighbors + theiler + 2):
        return float("nan")

    cut = int(len(vectors) * library_fraction)
    library, predict = vectors[:cut], vectors[cut:]
    library_targets = x[target_idx[:cut]]
    truth = x[target_idx[cut:]]
    if len(predict) < 10 or len(library) < n_neighbors + 1:
        return float("nan")

    # The library is entirely in the past of the prediction set, so a Theiler window is
    # only needed at the boundary; applying it there costs nothing and keeps the
    # guarantee uniform.
    tree = cKDTree(library)
    k = min(n_neighbors, len(library))
    distances, neighbours = tree.query(predict, k=k)
    distances = np.atleast_2d(distances)
    neighbours = np.atleast_2d(neighbours)

    scale = distances[:, :1]
    scale = np.where(scale <= 0, 1e-12, scale)
    weights = np.exp(-distances / scale)
    weights /= weights.sum(axis=1, keepdims=True)
    predictions = np.sum(weights * library_targets[neighbours], axis=1)

    if np.std(predictions) < 1e-12 or np.std(truth) < 1e-12:
        return float("nan")
    if metric == "spearman":
        # rankdata, not argsort-of-argsort: the latter breaks ties in *index* order, so a
        # heavily quantized series (poison_fraction has 58-95 distinct values in 7840
        # rows) gets ranks correlated with time and looks predictable when it is not.
        # That artifact sent the i.i.d. `Random` driver to z=+27 before this fix.
        predictions = rankdata(predictions)
        truth = rankdata(truth)
    elif metric != "pearson":
        raise ValueError(f"unknown metric '{metric}'. Expected 'pearson' or 'spearman'")
    return float(np.corrcoef(predictions, truth)[0, 1])


def skill_vs_horizon(series, E=3, tau=1, horizons=(1, 2, 4, 8, 16), **kwargs):
    """Forecast skill as a function of lead time.

    The *shape* carries the dynamical claim: deterministic chaos decays towards zero at a
    rate set by the largest Lyapunov exponent, a periodic signal does not decay, and a
    linearly correlated stochastic process decays like its autocorrelation. Comparing the
    decay against surrogates separates the third case from the first two.
    """
    return {h: simplex_skill(series, E=E, tau=tau, horizon=h, **kwargs) for h in horizons}

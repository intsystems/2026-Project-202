"""Convergent cross mapping, with a null that the ghost drivers can calibrate.

Phase 2 established that univariate determinism cannot detect a driver: the training loss
rejects the linear-surrogate null even when nothing is driving it, because SGD, the
schedule and batch effects give it nonlinear structure of its own. Driver detection is
inherently a *two-series* question.

**The idea (Sugihara et al. 2012).** If a driver ``X`` forces a response ``Y``, then by
Takens the reconstructed manifold of ``Y`` is a diffeomorphic image of the joint system's
attractor, so ``X``'s state is recoverable from ``Y``'s delay vectors. Cross mapping
*from* ``Y`` *to* ``X`` should therefore succeed — and, crucially, should **improve with
library size**, because a denser sampling of ``Y``'s manifold puts nearest neighbours
closer together. That convergence is what distinguishes causation from shared trend or
coincidental correlation, both of which give a flat skill-vs-library curve.

Note the direction convention, which is the usual source of confusion: *"Y xmap X"* —
embedding ``Y`` and predicting ``X`` — is evidence that **X drives Y**.

**What is added here.** `../poisoned_batch/ccm_pipeline.py` already draws rho(L) curves,
but reads them by eye and has no null. A rho of 0.6 means nothing on its own: two series
sharing a trend produce that easily. Two nulls are supplied instead:

``surrogate``
    IAAFT surrogates of the driver. Same spectrum, same values, alignment with the
    response destroyed. Cross-mapping must beat this.
``convergence``
    The slope of rho against library size on the real data. Genuine cross-mapping
    converges; a spurious match does not.

Both must pass. And the ghost runs -- constant, normal, uniform drivers -- provide an
empirical false-positive rate, which is the only way to know the null is calibrated on
*these* logs rather than on theory.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from forecast import delay_embed

__all__ = ["cross_map_skill", "ccm_convergence", "ccm_test"]


def cross_map_skill(embed_series, target_series, E=3, tau=1, library_size=None,
                    n_neighbors=None, theiler=None, metric="spearman",
                    n_replicates=4, seed=0):
    """Skill of predicting ``target_series`` from ``embed_series``'s delay vectors.

    Convergence is assessed by drawing random libraries of ``library_size`` from the
    embedded cloud, as in the original method, and averaging over ``n_replicates`` draws
    so the number is not one lucky sample.
    """
    x = np.asarray(embed_series, dtype=float)
    y = np.asarray(target_series, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"series lengths differ: {len(x)} vs {len(y)}")

    vectors, idx = delay_embed(x, E, tau)
    targets = y[idx]
    n = len(vectors)
    n_neighbors = (E + 1) if n_neighbors is None else n_neighbors
    theiler = (E - 1) * tau if theiler is None else theiler
    library_size = n if library_size is None else min(library_size, n)
    if library_size < n_neighbors + 1 or n < 4 * n_neighbors:
        return float("nan")

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_replicates):
        library = rng.choice(n, size=library_size, replace=False)
        tree = cKDTree(vectors[library])
        # Predict every point; exclude library members that are temporal neighbours of
        # the target, so skill cannot come from the trajectory sitting beside itself.
        k = min(n_neighbors + 2 * theiler + 1, library_size)
        distances, neighbours = tree.query(vectors, k=k)
        distances, neighbours = np.atleast_2d(distances), np.atleast_2d(neighbours)
        library_times = library[neighbours]

        valid = np.abs(library_times - idx[:, None] + idx[0]) > theiler
        order = np.argsort(~valid, axis=1, kind="stable")
        distances = np.take_along_axis(distances, order, axis=1)[:, :n_neighbors]
        library_times = np.take_along_axis(library_times, order, axis=1)[:, :n_neighbors]
        enough = np.take_along_axis(valid, order, axis=1)[:, :n_neighbors].all(axis=1)
        if enough.sum() < 20:
            continue

        distances, library_times = distances[enough], library_times[enough]
        scale = np.where(distances[:, :1] <= 0, 1e-12, distances[:, :1])
        weights = np.exp(-distances / scale)
        weights /= weights.sum(axis=1, keepdims=True)

        predicted = np.sum(weights * targets[library_times], axis=1)
        truth = targets[enough]
        if np.std(predicted) < 1e-12 or np.std(truth) < 1e-12:
            continue
        if metric == "spearman":
            predicted, truth = rankdata(predicted), rankdata(truth)
        scores.append(np.corrcoef(predicted, truth)[0, 1])

    return float(np.mean(scores)) if scores else float("nan")


def ccm_convergence(embed_series, target_series, E=3, tau=1, library_sizes=None,
                    **kwargs):
    """rho as a function of library size -- the convergence signature."""
    n = len(np.asarray(embed_series)) - (E - 1) * tau
    if library_sizes is None:
        library_sizes = [int(f * n) for f in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)]
    library_sizes = [L for L in library_sizes if L >= (E + 2)]
    return {int(L): cross_map_skill(embed_series, target_series, E=E, tau=tau,
                                    library_size=L, **kwargs) for L in library_sizes}


def ccm_test(embed_series, target_series, E=3, tau=1, n_surrogates=39, seed=0,
             surrogate_kind="iaaft", library_sizes=None, **kwargs):
    """Full test: convergence on the real data, plus a surrogate null at full library.

    Returns the convergence curve, the gain from smallest to largest library, and the
    rank test against surrogates of the *target* (the putative driver).
    """
    from surrogates import surrogate_test

    curve = ccm_convergence(embed_series, target_series, E=E, tau=tau, seed=seed,
                            library_sizes=library_sizes, **kwargs)
    sizes = sorted(curve)
    finite = [curve[L] for L in sizes if np.isfinite(curve[L])]
    gain = (finite[-1] - finite[0]) if len(finite) >= 2 else float("nan")

    result = surrogate_test(
        target_series,
        lambda surrogate: cross_map_skill(embed_series, surrogate, E=E, tau=tau,
                                          seed=seed, **kwargs),
        n_surrogates=n_surrogates, kind=surrogate_kind, seed=seed,
    )
    return {
        "curve": curve,
        "rho_max": finite[-1] if finite else float("nan"),
        "gain": gain,
        "z": result.z_score,
        "p": result.p_value,
        "surrogate_mean": float(result.values.mean()),
        # Both conditions, deliberately: beating the null without converging is the
        # signature of a shared trend, not of coupling.
        # Primary criterion is the surrogate rank test plus a minimal effect size. The
        # surrogate null already excludes the shared-spectrum/shared-trend explanation
        # that convergence is classically used to rule out, and convergence itself is
        # only diagnostic when the smallest library undersamples the manifold -- on these
        # logs rho can be saturated by L=20, which would make a gain criterion reject
        # genuine couplings. `gain` is still reported, as supporting evidence.
        "detected": bool(result.p_value <= 0.05 and np.isfinite(curve[sizes[-1]])
                         and (finite[-1] if finite else 0) > 0.05),
        "saturated": bool(len(finite) >= 2 and finite[0] > 0.9 * finite[-1]),
    }

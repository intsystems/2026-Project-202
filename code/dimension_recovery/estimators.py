"""The dimension estimators under test, behind one signature.

**SUPERSEDED by ``../active_dimension/mg.py``.  Do not use for new experiments.**

This module is kept only because ``exp1``--``exp15`` in this directory import it and their
committed result files are cited by the paper; re-pointing them at ``mg.py`` would change
nothing numerically (see below) but would invalidate every committed CSV's provenance.
``active_dimension/mg.py`` is the canonical estimator -- it is Algorithm 1 of the paper.

*The two implementations are numerically identical at matched settings.*  Verified:
``mg.all_estimators(x, MGConfig(max_E=25, tau=4, k_neighbors=30, theiler=0))["MG"]``
reproduces ``estimators.mg(x, max_E=25, k=30, tau=4)`` to 0.0e+00 on all 30 cells of
``results/exp9_frobenius_k10/stationary_validation.csv`` and on twelve further random and
quasiperiodic series across ``max_E`` in {15,21,25,31,41}, ``k`` in {5,20,30,50}, ``tau`` in
{1,2,4}.  Both call the same ``edm`` kernel.  What differs is the **defaults**, and only two
of them matter:

* **Theiler exclusion.**  Here the default is ``theiler=0`` -- no exclusion.  ``mg.py``
  defaults to ``"embedding"``, the delay span ``(max_E - 1) * tau``, capped at 150.  Every
  experiment in this directory except ``exp15_real_digits_functional_subspace_v3``
  (which sets ``theiler = (max_E - 1) * tau`` explicitly, so it already matches ``mg.py``)
  ran at 0.  Measured cost on ``exp9``: turning the exclusion on -- 96 samples at that
  configuration -- moves the held-out MAE from 0.302 to 0.335 on seed 1 and from 0.693 to
  0.677 on seed 2, and costs seed 1 its perfect rank correlation (rho +1.00 -> +0.99, one
  inversion).  Small, but it is not zero and it is not signed.
* **Clamping.**  ``edm.mle_intrinsic_dimension`` has ``clamp_to_max_E=True`` and nothing here
  overrides it, so a raw estimate above ``2 * max_E`` is returned as exactly ``max_E``.
  ``mg.py`` does not clamp; it returns the raw value and sets ``degenerate=True``.  The clamp
  **never fired in any committed result file**: the largest estimate anywhere in exp9 and
  exp11--exp15 is 22.34 against the lowest clamp threshold of 62, and no cell equals its
  ``max_E``.  It is a latent hazard, not a correction that was applied.  Where it does fire
  it is silent and severe: on an exactly periodic series at ``max_E=25`` this module returns
  25.0 while ``mg.py`` returns 399974.85 with ``degenerate=True``; on a constant window this
  module returns 25.0 and ``mg.py`` returns ``nan``.

``mg.py`` also reports the nulls (linear PR of the delay matrix, spectral PR, roughness,
autocorrelation time) on every call and flags degenerate windows, neither of which this
module does.

LB and MG come from the project's own package, so what is validated here is the code the
report actually runs, not a reimplementation. PR, TwoNN and correlation dimension are
added because agreement across estimators with different failure modes is worth more than
any single number, and because the audit's Tier 4 asks for them.

Every estimator takes a 1-D series and returns a float. ``nan`` means the window was too
short or degenerate, and is propagated rather than silently replaced.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension, local_svd_dimension       # noqa: E402
from edm.embedding import delay_embedding                          # noqa: E402

MAX_E, K, TAU = 15, 5, 1


def _embed(series, max_E=MAX_E, tau=TAU):
    try:
        return delay_embedding(np.asarray(series, dtype=float), max_E, tau)
    except ValueError:
        return None


def lb(series, max_E=MAX_E, k=K, theiler=0, tau=TAU):
    return mle_intrinsic_dimension(series, tau=tau, max_E=max_E, k_neighbors=k,
                                   correction="levina_bickel", theiler_window=theiler,
                                   rng=np.random.default_rng(0))


def mg(series, max_E=MAX_E, k=K, theiler=0, tau=TAU):
    return mle_intrinsic_dimension(series, tau=tau, max_E=max_E, k_neighbors=k,
                                   correction="mackay_ghahramani", theiler_window=theiler,
                                   rng=np.random.default_rng(0))


def pr(series, max_E=MAX_E, tau=TAU, **_):
    """Participation ratio of the delay matrix's singular values.

    Exactly 1.0 on a straight line, so unlike LB it has no tangent constant to be
    confused with a dimension. Linear, so it cannot see a curved manifold's true
    dimension -- it reports the dimension of the linear span, which for a k-torus
    observed through a scalar is at least 2k. That bias is deterministic and therefore
    correctable by calibration, which is the case for including it.
    """
    return local_svd_dimension(np.asarray(series, dtype=float), tau=tau, embedding=max_E,
                               measure="participation_ratio")


def twonn(series, max_E=MAX_E, theiler=0, tau=TAU, discard=0.1, **_):
    """Facco et al. two-nearest-neighbour estimator.

    Uses only the ratio mu = r2/r1, so it is far less sensitive to the choice of k than
    LB, and it is fitted on the empirical CDF rather than pooled, so a few degenerate
    points cannot dominate.
    """
    emb = _embed(series, max_E, tau)
    if emb is None or len(emb) < 10:
        return np.nan
    n = len(emb)
    if theiler <= 0:
        dist, _ = KDTree(emb).query(emb, k=3)
        r1, r2 = dist[:, 1], dist[:, 2]
    else:
        want = min(n, 2 * theiler + 4)
        dist, idx = KDTree(emb).query(emb, k=want)
        keep = np.abs(idx - np.arange(n)[:, None]) > theiler
        # stable sort on ~keep pulls the valid neighbours to the front, distances
        # within each group staying in the ascending order KDTree returned them in
        order = np.argsort(~keep, axis=1, kind="stable")
        dist = np.take_along_axis(dist, order, axis=1)
        enough = keep.sum(axis=1) >= 2
        r1 = np.where(enough, dist[:, 0], np.nan)
        r2 = np.where(enough, dist[:, 1], np.nan)
    ok = np.isfinite(r1) & (r1 > 0)
    if ok.sum() < 10:
        return np.nan
    mu = np.sort(r2[ok] / r1[ok])
    m = len(mu)
    cut = int(m * (1 - discard))
    mu, f = mu[:cut], (np.arange(1, m + 1) / m)[:cut]
    good = (mu > 1) & (f < 1)
    if good.sum() < 5:
        return np.nan
    x, y = np.log(mu[good]), -np.log1p(-f[good])
    return float((x @ y) / (x @ x))                          # fit through the origin


def corrdim(series, max_E=MAX_E, theiler=0, tau=TAU, cap=800, **_):
    """Grassberger-Procaccia slope over the middle decade of the correlation sum.

    The pair count is quadratic, so long windows are thinned to ``cap`` points first.
    Thinning is a stride, not a random subsample, so the Theiler exclusion still means
    what it says in units of the thinned index.
    """
    emb = _embed(series, max_E, tau)
    if emb is None or len(emb) < 20:
        return np.nan
    if len(emb) > cap:
        emb = emb[:: int(np.ceil(len(emb) / cap))]
    n = len(emb)
    i, j = np.triu_indices(n, k=1 + theiler)
    d = np.linalg.norm(emb[i] - emb[j], axis=1)
    d = d[d > 0]
    if len(d) < 50:
        return np.nan
    lo, hi = np.percentile(d, [10, 50])
    if not (hi > lo > 0):
        return np.nan
    radii = np.logspace(np.log10(lo), np.log10(hi), 12)
    counts = np.array([(d < r).mean() for r in radii])
    ok = counts > 0
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(radii[ok]), np.log(counts[ok]), 1)[0]
    return float(slope)


def roughness(series, **_):
    """std(diff x) / std(x): the null. It must NOT track a change in dimension.

    ../prediction_improved/report_0708_experiments.md section 5 finds Spearman +0.934
    between this and the LB estimate across the project's real logs. If it also tracks k
    here, then nothing in this experiment separates geometry from smoothness.
    """
    x = np.asarray(series, dtype=float)
    s = np.std(x)
    return float(np.std(np.diff(x)) / s) if s > 0 else np.nan


ESTIMATORS = {"LB": lb, "MG": mg, "PR": pr, "TwoNN": twonn, "CorrDim": corrdim,
              "roughness": roughness}


def evaluate(series, window, stride, names=None, **kw):
    """Median of each estimator over sliding windows. Returns {name: float}."""
    names = names or list(ESTIMATORS)
    x = np.asarray(series, dtype=float)
    starts = range(0, len(x) - window + 1, stride)
    out = {}
    for name in names:
        vals = [ESTIMATORS[name](x[s:s + window], **kw) for s in starts]
        vals = np.asarray(vals, dtype=float)
        out[name] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        out[name + "_sd"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan
    return out


def trace(series, window, stride, name="MG", **kw):
    """Sliding estimate labelled by the window's right edge."""
    x = np.asarray(series, dtype=float)
    right, vals = [], []
    for s in range(0, len(x) - window + 1, stride):
        vals.append(ESTIMATORS[name](x[s:s + window], **kw))
        right.append(s + window - 1)
    return np.asarray(right), np.asarray(vals, dtype=float)

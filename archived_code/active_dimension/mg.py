"""The dimension estimators under test, behind one frozen configuration object.

MG is the MacKay-Ghahramani-pooled Levina-Bickel maximum-likelihood intrinsic dimension of a
delay embedding.  The neighbour search, the dither and the distance floor are imported from
the project's own package (``grokking_analysis/edm``) rather than reimplemented, so the
numbers here are bit-identical to the ones the earlier reports produce -- but every
estimator that can share a KD-tree does, which is a ~3x speedup over calling them one by one.

Four deliberate deviations from ``dimension_recovery/estimators.py``:

* ``clamp_to_max_E`` is **off**.  Upstream returns ``max_E`` whenever the raw estimate
  exceeds ``2 * max_E``, turning a divergent estimate into a plausible-looking number.
* the Theiler window is **on** by default (``"embedding"``), not 0.  Every headline
  experiment in ``dimension_recovery`` left it at 0.
* ``degenerate`` inputs are reported as ``nan``, not silently floored.  ``edm.dimension``
  floors the per-point log-ratio sum at 1e-5 and the neighbour distance at 1e-8, so an
  exactly recurrent series (a rational frequency ratio, a constant window) returns 0.08 or
  ``n(k-1)-1 = 399975`` instead of failing.  :func:`all_estimators` detects both and flags
  them in ``degenerate``.
* the nulls -- ``roughness``, the autocorrelation time, and the **linear** participation
  ratio of the delay matrix -- are returned alongside MG on every call, so no experiment can
  report MG without them.  The audit of ``exp10-12`` found that the linear PR recovers k
  *better* than MG on the same data (MAE 1.03-1.23 vs 1.29-1.46); an experiment that does
  not report it cannot claim that nonlinear manifold geometry is doing any work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grokking_analysis"))

from edm.dimension import _dither, _neighbor_distances, resolve_theiler_window   # noqa: E402
from edm.embedding import delay_embedding, autocorrelation_time                  # noqa: E402

FLOOR_DIST = 1e-8          # edm.dimension's distance floor
FLOOR_SUM = 1e-5           # edm.dimension's log-ratio-sum floor
THEILER_CAP = 150          # see all_estimators: the neighbour query is linear in this


@dataclass(frozen=True)
class MGConfig:
    """Everything that can change an MG number.  Frozen after calibration (E1).

    Only *estimator* parameters live here.  The audit of ``exp10-12`` found that their
    calibration grid also contained system parameters (``cycles_per_window``, ``eta``), so
    selecting a configuration changed the data, not just the measurement -- and since the
    objective was error against the known k, the absolute level of the estimate was tuned on
    the answer (MG at k=20 moves 5.3 -> 16.3 across that grid).  Keeping the two kinds of
    parameter in different objects makes that mistake impossible to repeat here.
    """

    max_E: int = 20
    tau: object = 1                   # int, or "acorr" for acf_time/4 measured per window
    k_neighbors: int = 5
    theiler: object = "embedding"     # 0 | int | "embedding" | "autocorr"
    window: int = 6000
    stride: int = 500
    dither: float = 1e-9

    def as_dict(self):
        return asdict(self)

    def tag(self):
        return f"E{self.max_E}_t{self.tau}_k{self.k_neighbors}_th{self.theiler}_W{self.window}"


DEFAULT = MGConfig()


def resolve_tau(cfg, x):
    """Turn ``tau="acorr"`` into a concrete lag, measured from the window itself.

    The delay window spans ``(max_E - 1) * tau`` samples, and a torus is only unfolded when
    that span covers a real fraction of the oscillation period.  With a fixed integer tau,
    a claim that "MG saturates" on a slowly oscillating log is a claim about that tau, not
    about MG: measured on a period-400 torus, MG at r = 2/4/6 reads 1.84/2.26/2.31 at tau=1
    and 3.45/18.1/20.3 at tau=20.  ``"acorr"`` sets ``tau ~ acf_time / 4``, the textbook
    choice, so the estimator adapts to the signal's own timescale and the comparison across
    regimes is fair.  The span is capped at an eighth of the window so the embedding still
    has points to work with.
    """
    if cfg.tau != "acorr":
        return int(cfg.tau)
    a = autocorrelation_time(np.asarray(x, float))
    tau = max(1, int(round(a / 4.0)))
    return int(min(tau, max(1, len(x) // (8 * max(1, cfg.max_E - 1)))))

SPEC_NBINS = (64, 256, 1024, 0)   # 0 = no binning, native FFT resolution
ESTIMATOR_NAMES = (("MG", "LB", "TwoNN", "PRdelay")
                   + tuple(f"specPR{n}" for n in SPEC_NBINS) + ("roughness", "acorr"))
NULLS = ("PRdelay",) + tuple(f"specPR{n}" for n in SPEC_NBINS) + ("roughness", "acorr")


def spectral_pr(x, nbin=256):
    """Participation ratio of the (binned) periodogram: an effective count of spectral lines.

    The sharpest null available for a quasi-periodic system, and the one that a
    phase-randomised surrogate cannot provide.  Randomising the phases of an r-torus leaves
    an r-torus -- every line stays in the module generated by f_1..f_r -- so an IAAFT
    surrogate tests "nonlinear determinism vs a linear Gaussian process", not "r directions
    vs a broader spectrum".  This statistic tests the second: it is linear, needs no
    embedding and no neighbours, and counts modes directly.  If it tracks r as well as MG
    does, then MG's correlation with r is a spectral fact, not a geometric one.

    ``nbin`` is a free parameter and is therefore **calibrated on the same split, by the same
    objective, as MG's own parameters** -- otherwise the head-to-head is rigged.  It matters:
    256 bands span [0, 0.5] in steps of 0.00195, which resolves a drive band at f0=1/16 and
    does not resolve one at f0=1/400 (measured: specPR256 = 1.03, 2.08, 2.12, 2.14, 2.11 for
    r = 1, 2, 4, 6, 8 on the slow torus -- saturating at 2).  ``nbin=0`` keeps the native FFT
    resolution.
    """
    x = np.asarray(x, float)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    if p.sum() <= 0 or (nbin and len(p) < nbin):
        return np.nan
    if nbin:
        e = np.linspace(0, len(p), nbin + 1).astype(int)
        p = np.array([p[a:c].sum() for a, c in zip(e[:-1], e[1:])])
    b = p / p.sum()
    return float(1.0 / np.sum(b ** 2))


def all_estimators(x, cfg: MGConfig = DEFAULT, seed=0):
    """Every estimator on one window, sharing one KD-tree.  Returns a dict.

    ``degenerate`` is True when the window contains delay vectors closer together than the
    1e-8 floor, or when the pooled log-ratio sum hits the 1e-5 floor.  Both make the returned
    MG meaningless; the flag exists so that a run cannot average over them unnoticed.
    """
    x = np.asarray(x, float)
    out = {n: np.nan for n in ESTIMATOR_NAMES}
    out["degenerate"] = True
    if len(x) < cfg.max_E * (1 if cfg.tau == "acorr" else int(cfg.tau)) + 20             or not np.isfinite(x).all():
        return out
    sd = x.std()
    out["roughness"] = float(np.diff(x).std() / sd) if sd > 0 else np.nan
    out["acorr"] = float(autocorrelation_time(x))
    for nb in SPEC_NBINS:
        out[f"specPR{nb}"] = spectral_pr(x, nb)
    if sd <= 0:
        return out

    xd = _dither(x, cfg.dither, np.random.default_rng(seed))
    tau = resolve_tau(cfg, x)
    out["tau_used"] = tau
    # `edm._neighbor_distances` guarantees enough valid neighbours by querying
    # k + 2*theiler + 1 of them, so the cost is linear in the Theiler window.  With
    # tau="acorr" on a smooth observer that reaches ~1000 neighbours per point and a single
    # window takes a minute.  Capped, and the cap is recorded.
    th = min(resolve_theiler_window(cfg.theiler, xd, tau, cfg.max_E), THEILER_CAP)
    out["theiler_used"] = th
    try:
        emb = delay_embedding(xd, cfg.max_E, tau)
    except ValueError:
        return out

    # linear null: participation ratio of the delay matrix's singular spectrum
    c = emb - emb.mean(0, keepdims=True)
    s2 = np.linalg.svd(c, compute_uv=False) ** 2
    if s2.sum() > 0:
        out["PRdelay"] = float(s2.sum() ** 2 / (s2 ** 2).sum())

    d = _neighbor_distances(emb, cfg.k_neighbors, th, floor=FLOOR_DIST)
    if d is None:
        return out
    hit_floor = float((d <= FLOOR_DIST * 1.000001).mean())

    r_k = d[:, -1:]
    S = np.sum(np.log(r_k / d[:, :-1]), axis=1)
    n = len(S)
    hit_sum = float((S <= FLOOR_SUM).mean())
    S = np.maximum(S, FLOOR_SUM)
    total = float(S.sum())
    if np.isfinite(total) and total > 0:
        out["MG"] = (n * (cfg.k_neighbors - 1) - 1) / total
        loc = (cfg.k_neighbors - 1) / S
        out["LB"] = float(np.mean(loc[np.isfinite(loc)]))

    # TwoNN on the same neighbour distances (r1, r2 are already Theiler-filtered)
    r1, r2 = d[:, 0], d[:, 1]
    good = np.isfinite(r1) & (r1 > FLOOR_DIST)
    if good.sum() >= 20:
        mu = np.sort(r2[good] / r1[good])
        m = len(mu)
        cut = int(m * 0.9)
        mu, f = mu[:cut], (np.arange(1, m + 1) / m)[:cut]
        g = (mu > 1) & (f < 1)
        if g.sum() >= 5:
            a, b = np.log(mu[g]), -np.log1p(-f[g])
            out["TwoNN"] = float((a @ b) / (a @ a))

    out["degenerate"] = bool(hit_floor > 0.01 or hit_sum > 0.01)
    out["frac_floor"] = hit_floor
    out["frac_sumfloor"] = hit_sum
    return out


def mg(x, cfg: MGConfig = DEFAULT, seed=0):
    return all_estimators(x, cfg, seed)["MG"]


# --------------------------------------------------------------------- sliding windows
def window_starts(n, cfg: MGConfig):
    return list(range(0, n - cfg.window + 1, cfg.stride))


def sliding(x, cfg: MGConfig = DEFAULT, seed=0):
    """Right-edge-labelled trace: the value at index t uses only samples t-W+1 .. t.

    Returns (right_edges, {name: array}).  Labelling by the right edge is what makes a
    detection lag meaningful; ``exp10`` labels windows by the right edge but takes the
    *ground truth* there too, which mislabels every window straddling a switch.
    """
    x = np.asarray(x, float)
    st = window_starts(len(x), cfg)
    recs = [all_estimators(x[s:s + cfg.window], cfg, seed) for s in st]
    right = np.array([s + cfg.window - 1 for s in st])
    keys = list(ESTIMATOR_NAMES) + ["degenerate"]
    return right, {k: np.array([r.get(k, np.nan) for r in recs], float) for k in keys}


def summarise(x, cfg: MGConfig = DEFAULT, seed=0):
    """Median / spread of each estimator over the sliding windows of one stationary series."""
    st = window_starts(len(np.asarray(x, float)), cfg)
    recs = [all_estimators(np.asarray(x, float)[a:a + cfg.window], cfg, seed) for a in st]
    tr = {k: np.array([r.get(k, np.nan) for r in recs], float)
          for k in list(ESTIMATOR_NAMES) + ["degenerate"]}
    bad = tr["degenerate"] > 0.5
    out = {"frac_degenerate": float(bad.mean()), "n_windows": int(len(bad))}
    # propagate the resolved lag and Theiler window: without these a capped run is
    # indistinguishable from an uncapped one in the saved CSV
    for k in ("tau_used", "theiler_used"):
        v = [r[k] for r in recs if k in r]
        out[k] = float(np.median(v)) if v else np.nan
    for k in ESTIMATOR_NAMES:
        v = tr[k][~bad]
        v = v[np.isfinite(v)]
        out[k] = float(np.median(v)) if len(v) else np.nan
        out[k + "_sd"] = float(np.std(v)) if len(v) else np.nan
    return out


# --------------------------------------------------------------------- surrogates
def match_endpoints(x, min_frac=0.85):
    """Trim to the sub-series whose ends match best in value and slope.

    A Fourier surrogate implicitly treats the window as one period of a periodic signal.
    If the ends do not match, the implied jump leaks broadband power across the whole
    spectrum, and randomising *those* phases manufactures noise that was never in the data.
    Measured: on a pure sinusoid (whose surrogate must still be a 1-torus, MG ~ 2) the
    unmatched surrogate reads MG = 4.3.  Theiler et al. (Physica D 58, 1992) sec. 3.3.
    """
    x = np.asarray(x, float)
    n = len(x)
    lo = int(n * min_frac)
    if lo >= n - 2:
        return x
    d = np.diff(x)
    s = np.std(x) + 1e-30
    ds = np.std(d) + 1e-30
    cost = np.abs(x[:n - lo] - x[lo - 1:n - 1]) / s + np.abs(d[:n - lo] - d[lo - 2:n - 2]) / ds
    return x[:lo + int(np.argmin(cost))]


def iaaft(x, iters=12, rng=None, match=True):
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996).

    Preserves the power spectrum -- and therefore the autocorrelation, the roughness and
    every smoothness statistic -- and the amplitude distribution, while randomising the
    phase relations that make a signal deterministic.  For an r-torus the surrogate is not
    noise: it is a signal with the same spectral *lines* but independent phases on the
    intermodulation lines, so its delay dimension is the number of resolvable lines rather
    than r.  That is exactly the comparison wanted: if MG on the data equals MG on the
    surrogate, the estimate is a function of the power spectrum and carries no information
    about geometry, however well it correlates with r.  Theiler et al. (Physica D 58, 1992);
    Osborne & Provenzale (Physica D 35, 1989) for why coloured noise alone yields a finite,
    reproducible dimension.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    if match:
        x = match_endpoints(x)
    n = len(x)
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=n)
        y = sorted_x[np.argsort(np.argsort(y))]      # re-impose the amplitude distribution
    return y


def surrogate_summary(x, cfg: MGConfig = DEFAULT, n=3, seed=0):
    """Median MG over ``n`` IAAFT surrogates of ``x``, plus the spread across them."""
    vals = []
    for i in range(n):
        s = iaaft(x, rng=np.random.default_rng(1000 * seed + i))
        vals.append(summarise(s, cfg, seed=seed)["MG"])
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(MG_surr=np.nan, MG_surr_sd=np.nan, n_surr=0)
    return dict(MG_surr=float(np.median(v)), MG_surr_sd=float(v.std()), n_surr=int(len(v)))


# --------------------------------------------------------------------- calibration
class Calibration:
    """A monotone map estimate -> r, fitted on calibration seeds only and then frozen.

    Fitted on a split that is disjoint in **both** seed and r, because the exp10-12 audit
    found that "held out" there meant held-out seeds while the frequency geometry was
    bit-identical across seeds, so the geometry MG responds to was never held out at all.
    """

    def __init__(self, kind="isotonic"):
        self.kind, self.fitted = kind, False

    def fit(self, est, r):
        m, y = np.asarray(est, float), np.asarray(r, float)
        ok = np.isfinite(m) & np.isfinite(y)
        m, y = m[ok], y[ok]
        if len(m) < 3:
            raise ValueError("not enough calibration points")
        if self.kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            f = IsotonicRegression(out_of_bounds="clip").fit(m, y)
            self._p = f.predict
        elif self.kind == "affine":
            a, b = np.polyfit(m, y, 1)
            self.coef = (float(a), float(b))
            self._p = lambda z: a * z + b
        elif self.kind == "identity":
            self._p = lambda z: z
        else:
            raise ValueError(self.kind)
        self.fitted, self.n_points = True, int(len(m))
        return self

    def predict(self, est):
        if not self.fitted:
            raise RuntimeError("calibration used before it was fitted")
        z = np.atleast_1d(np.asarray(est, float))
        out = np.full(len(z), np.nan)
        ok = np.isfinite(z)
        if ok.any():
            out[ok] = self._p(z[ok])
        return out if np.ndim(est) else float(out[0])

"""All per-window statistics: MG, every scalar competitor, and the domain monitors.

Scalar statistics see ONLY the observed one-dimensional log window. Delay-space competitors
use the shared construction of `research_mg_wins/baselines.py` (z-scored window, E=20,
lag = acf_time/4). `recurrence_rate` and `corr_dim` are re-implemented here with `pdist`
(identical numbers, checked in `_selfcheck`) because the shared versions build an
n x n x 20 float64 array (~1 GB at n=2500), above this agent's RAM budget.
Domain statistics use internals (full strategy state, cross-play of checkpoints).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "code"))
import baselines as BL  # noqa: E402
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402

CFG_MG = EstimatorConfig(max_E=20, tau="acorr", k_neighbors=20, theiler="autocorr", theiler_cap=320)
CFG_MG_TAU1 = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
RELS = (0.01, 0.02, 0.05, 0.1)


# ---------------- memory-safe copies of shared delay statistics -------------------------
def _theiler_pairs_lowmem(Y, idx, w):
    d = pdist(Y)
    t = pdist(idx[:, None].astype(float), "cityblock")
    return d[t > w]


def recurrence_rate(x, E=20, tau=None, frac=0.1):
    Y, tau = BL.delay(x, E, tau)
    Ys, idx = BL._sub(Y, 2000)
    d = _theiler_pairs_lowmem(Ys, idx, (E - 1) * tau)
    return float(np.mean(d < frac * np.median(d)))


def corr_dim(x, E=20, tau=None, q=(0.01, 0.1)):
    Y, tau = BL.delay(x, E, tau)
    Ys, idx = BL._sub(Y, 2500)
    d = _theiler_pairs_lowmem(Ys, idx, (E - 1) * tau)
    r = np.quantile(d, np.linspace(q[0], q[1], 8))
    C = np.array([np.mean(d < ri) for ri in r])
    return float(np.polyfit(np.log(r), np.log(C), 1)[0])


# ---------------- spectral competitors ----------------------------------------------------
def _spectrum(x):
    y = BL.z(x)
    P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return P / P[1:].sum()


def peak_list(x, rel, min_bin=2):
    """Same peak definition as BL.peak_count; returns kept peak bins in descending power."""
    P = _spectrum(x)
    peaks = [i for i in range(max(2, min_bin), len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    mass = sorted(((P[max(1, i - 3):i + 4].sum(), i) for i in peaks), reverse=True)
    kept = []
    for m, i in mass:
        if m < rel:
            break
        if all(abs(i - k) > 6 for k in kept):
            kept.append(i)
    return kept, P


def harmonic_count(x, rel=0.02, max_order=4, max_total=6, min_bin=4):
    """Number of fundamentals needed to explain all strong spectral peaks as integer
    combinations (harmonics and intermodulation tones) of the fundamentals.
    Strong spectral competitor written for this setting (anharmonic cycles)."""
    kept, P = peak_list(x, rel, min_bin)
    freqs = []
    for i in kept:          # parabolic refinement of the bin
        if 1 <= i < len(P) - 1:
            a, b, c = np.log(P[i - 1] + 1e-30), np.log(P[i] + 1e-30), np.log(P[i + 1] + 1e-30)
            den = a - 2 * b + c
            freqs.append(i + (0.5 * (a - c) / den if den != 0 else 0.0))
        else:
            freqs.append(float(i))
    F = []

    def explained(f):
        tol = max(2.0, 0.01 * f)
        if not F:
            return False
        rng = range(-max_order, max_order + 1)
        for ns in itertools.product(rng, repeat=len(F)):
            tot = sum(abs(n) for n in ns)
            if tot == 0 or tot > max_total:
                continue
            if abs(abs(sum(n * g for n, g in zip(ns, F))) - f) <= tol:
                return True
        return False

    for f in freqs:
        if explained(f):
            continue
        sub = [j for j, g in enumerate(F) if any(abs(g - n * f) <= max(2.0, 0.01 * g) for n in range(2, 7))]
        if sub:
            F[sub[0]] = f
            continue
        F.append(f)
        if len(F) >= 6:
            break
    return len(F)


def simple(seg):
    t = np.arange(len(seg))
    r = seg - np.polyval(np.polyfit(t, seg, 1), t)
    return {"crossings": trend_crossings(seg), "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            "det_std": float(r.std() / (abs(seg.mean()) + 1e-12))}


def scalar_stats(seg, timing=None):
    import time
    out = {}

    def rec(name, fn):
        t0 = time.perf_counter()
        try:
            out[name] = float(fn())
        except Exception:  # noqa: BLE001
            out[name] = float("nan")
        if timing is not None:
            timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0

    rec("MG", lambda: estimate(seg, CFG_MG).MG)
    rec("MG_tau1", lambda: estimate(seg, CFG_MG_TAU1).MG)
    rec("spectral_entropy", lambda: BL.spectral_entropy(seg))
    rec("self_repeat", lambda: BL.self_repeat(seg))
    rec("self_repeat_long", lambda: BL.self_repeat(seg, 20, 2000))
    rec("roughness", lambda: BL.roughness(seg))
    for r in RELS:
        rec(f"peak_count_{r}", lambda r=r: BL.peak_count(seg, rel=r))
        rec(f"harmonic_count_{r}", lambda r=r: harmonic_count(seg, rel=r))
    rec("perm_entropy", lambda: BL.perm_entropy(seg))
    rec("recurrence_rate", lambda: recurrence_rate(seg))
    rec("corr_dim", lambda: corr_dim(seg))
    rec("twonn", lambda: BL.twonn_fit(seg))
    rec("linear_pr", lambda: BL.linear_pr(seg))
    t0 = __import__("time").perf_counter()
    out.update(simple(seg))
    if timing is not None:
        timing["simple"] = timing.get("simple", 0.0) + __import__("time").perf_counter() - t0
    return out


# ---------------- domain monitors (internals) --------------------------------------------
def domain_stats(game, simres, a0, a1, B, rng, n_ckpt=24):
    import sim as G
    M, h, g, c = game.matrices()
    X = simres["x"][a0:a1]; Y = simres["y"][a0:a1]
    gx = (Y @ M.T + h) * X * (1 - X)
    gy = (X @ M + g) * Y * (1 - Y)
    xi2 = float(np.mean(game.S ** 2 * (np.sum(gx ** 2, 1) + np.sum(gy ** 2, 1))))
    Z = np.concatenate([X, Y], 1)          # strategy state (probabilities; logits of
    # transitive subgames grow without bound and would dominate a logit-space PCA)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)[::-1]
    out = {"xi2": xi2, "state_pr": float(ev.sum() ** 2 / (ev ** 2).sum())}
    for r in (0.01, 0.03, 0.1):
        out[f"state_count_{r}"] = int(np.sum(ev > r * ev[0]))
    # cross-play matrix of n checkpoints (player-1 ckpt i vs player-2 ckpt j), B plays per entry
    idx = np.linspace(0, len(X) - 1, n_ckpt).astype(int)
    Xi = X[idx]; Yj = Y[idx]
    U = game.S * (Xi @ M @ Yj.T + (Xi @ h)[:, None] + (Yj @ g)[None, :] + c)
    if np.isfinite(B):
        XX = np.repeat(Xi, n_ckpt, 0); YY = np.tile(Yj, (n_ckpt, 1))
        sd = np.sqrt(G.play_variance(game, XX, YY) / B).reshape(n_ckpt, n_ckpt)
        U = U + rng.normal(0, 1, U.shape) * sd
    U = U - U.mean(0, keepdims=True) - U.mean(1, keepdims=True) + U.mean()
    s = np.linalg.svd(U, compute_uv=False)
    out["xplay_pr"] = float((s ** 2).sum() ** 2 / (s ** 4).sum())
    for r in (0.03, 0.1, 0.3):
        out[f"xplay_count_{r}"] = int(np.sum(s > r * s[0]))
    return out


def _selfcheck():
    rng = np.random.default_rng(0)
    t = np.arange(900)
    x = np.sin(0.05 * t) + 0.5 * np.sin(0.0731 * t) + 0.01 * rng.normal(size=len(t))
    a, b = recurrence_rate(x), BL.recurrence_rate(x)
    c, d = corr_dim(x), BL.corr_dim(x)
    assert abs(a - b) < 1e-12 and abs(c - d) < 1e-9, (a, b, c, d)
    print("selfcheck ok", a, b, c, d)


if __name__ == "__main__":
    _selfcheck()

"""Real data, a frozen nonlinear backbone, and a trainable adapter confined to k directions.

The package exists to keep three numbers apart, so they are kept apart here:

``available``   the number of directions the optimiser is *allowed* to move in, k.  Set by
                construction: ``theta = theta0 + V^T c`` with ``V`` a fixed (k, P) orthonormal
                frame.
``functional``  the rank of the Jacobian of the model's outputs with respect to those k
                directions, on held-out data.  Measured -- two allowed directions can produce
                the same function change.
``active``      the number of directions the optimiser actually excites over the analysis
                window.  Measured from the trajectory and update covariances.

The reason the third is not the first, even when the noise is deliberately confined to r
directions: near a minimum ``c_{t+1} = c_t - eta (H c_t + xi_t)``, and the stationary
covariance is ``sum_j A^j eta^2 Sigma A^{jT}`` with ``A = I - eta H``, whose range is the
smallest ``A``-invariant subspace containing ``range(Sigma)``.  For a generic ``H`` that
Krylov space is all of R^k however small ``rank(Sigma)`` is.  Only after preconditioning by
``H^{-1}`` (which makes the linearised dynamics isotropic) does rank-r forcing give a rank-r
trajectory.  Both are run, precisely because the difference is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Enough mutually incommensurate frequencies for the k=20 calibration arm.  The
# original atlas stopped at 53 because it only swept r<=8; keeping the list here
# avoids a silent IndexError when the controlled experiment is extended to r=20.
PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
])


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ data + backbone
def digits_split(seed=0, n_train=1024, n_probe=384):
    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    itr = rng.choice(len(Xtr), min(n_train, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), min(n_probe, len(Xte)), replace=False)
    return Xtr[itr], ytr[itr], Xte[ite], yte[ite]


def frozen_backbone(Xtr, ytr, seed=0, hidden=(96, 96), steps=2000, lr=0.08):
    """A small tanh MLP trained briefly on the data, then frozen.  Returns ``phi(X)``.

    Trained rather than random, so the features are of the data; briefly, so the adapter
    still has work to do.  Numpy, because at this size a framework costs more than it saves.
    """
    rng = np.random.default_rng(seed)
    sizes = (Xtr.shape[1],) + tuple(hidden)
    W = [rng.standard_normal((sizes[i + 1], sizes[i])) / np.sqrt(sizes[i]) for i in range(len(hidden))]
    b = [np.zeros(sizes[i + 1]) for i in range(len(hidden))]
    Wo = rng.standard_normal((10, hidden[-1])) / np.sqrt(hidden[-1])
    bo = np.zeros(10)
    Y = np.eye(10)[ytr]
    for _ in range(steps):
        idx = rng.choice(len(Xtr), 256, replace=False)
        a, acts = Xtr[idx], [Xtr[idx]]
        for Wi, bi in zip(W, b):
            a = np.tanh(a @ Wi.T + bi)
            acts.append(a)
        g = (_softmax(a @ Wo.T + bo) - Y[idx]) / len(idx)
        gWo, gbo, da = g.T @ a, g.sum(0), g @ Wo
        for i in range(len(W) - 1, -1, -1):
            dz = da * (1 - acts[i + 1] ** 2)
            W[i] -= lr * (dz.T @ acts[i]); b[i] -= lr * dz.sum(0)
            da = dz @ W[i]
        Wo -= lr * gWo; bo -= lr * gbo

    def phi(Z):
        a = Z
        for Wi, bi in zip(W, b):
            a = np.tanh(a @ Wi.T + bi)
        return np.hstack([a, np.ones((len(a), 1))])
    return phi


# ------------------------------------------------------------------ the adapter system
@dataclass
class Adapter:
    """A linear head on frozen nonlinear features, confined to a k-dimensional subspace.

    ``logits(c) = L0 + sum_j c_j M_j`` with ``M_j = Phi V_j^T`` precomputed, so a step costs
    two (k, n*10) contractions and a 10^5-step run costs seconds.  The loss is softmax
    cross-entropy, so ``c -> loss`` is nonlinear and the curvature ``H(c)`` is a real,
    data-dependent, non-isotropic object -- which is what makes the Krylov point above bite.
    """

    Phi: np.ndarray
    Y: np.ndarray
    Phi_p: np.ndarray
    Yp: np.ndarray
    k: int
    seed: int = 0
    V: np.ndarray = field(default=None)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed + 7919)
        n, H = self.Phi.shape
        self.W0 = rng.standard_normal((10, H)) / np.sqrt(H)
        if self.V is None:
            self.V = np.linalg.qr(rng.standard_normal((10 * H, self.k)))[0].T   # (k, P)
        self.L0, self.L0p = self.Phi @ self.W0.T, self.Phi_p @ self.W0.T
        Vw = self.V.reshape(self.k, 10, H)
        self.M = np.einsum("nh,jch->jnc", self.Phi, Vw)                          # (k, n, 10)
        self.Mp = np.einsum("mh,jch->jmc", self.Phi_p, Vw)
        self.Yoh, self.Yoh_p = np.eye(10)[self.Y], np.eye(10)[self.Yp]
        self.n = n

    # -- forward / backward ---------------------------------------------------
    def logits(self, c):
        return self.L0 + np.tensordot(c, self.M, axes=(0, 0))

    def logits_probe(self, c):
        return self.L0p + np.tensordot(c, self.Mp, axes=(0, 0))

    def loss_grad(self, c, idx=None, w=None):
        """(loss, grad_c).  ``idx`` selects a mini-batch, ``w`` gives per-example weights."""
        if idx is None:
            L, Yoh, M, Yl, nn = self.logits(c), self.Yoh, self.M, self.Y, self.n
        else:
            L = self.L0[idx] + np.tensordot(c, self.M[:, idx], axes=(0, 0))
            Yoh, M, Yl, nn = self.Yoh[idx], self.M[:, idx], self.Y[idx], len(idx)
            w = None if w is None else w[idx]
        p = _softmax(L)
        ll = -np.log(np.clip(p[np.arange(nn), Yl], 1e-12, None))
        if w is None:
            loss, Gm = float(ll.mean()), (p - Yoh) / nn
        else:
            s = w.sum()
            loss, Gm = float(w @ ll / s), (w[:, None] * (p - Yoh)) / s
        return loss, np.tensordot(M, Gm, axes=([1, 2], [0, 1]))

    def loss_probe(self, c):
        p = _softmax(self.logits_probe(c))
        return float(-np.log(np.clip(p[np.arange(len(self.Yp)), self.Yp], 1e-12, None)).mean())

    # -- curvature and the three dimensions -----------------------------------
    def hessian(self, c, eps=1e-4):
        """(k, k) Hessian of the full-batch loss by central differences of the gradient."""
        Hm = np.zeros((self.k, self.k))
        for j in range(self.k):
            e = np.zeros(self.k); e[j] = eps
            Hm[:, j] = (self.loss_grad(c + e)[1] - self.loss_grad(c - e)[1]) / (2 * eps)
        return 0.5 * (Hm + Hm.T)

    def functional_dim(self, tol=1e-8):
        """Rank and participation ratio of d(probe logits)/dc.  ``logits`` is affine in c,
        so the Jacobian is the constant matrix ``Mp``; the signature is kept general."""
        J = self.Mp.reshape(self.k, -1).T
        s = np.linalg.svd(J, compute_uv=False)
        s2 = s ** 2
        return int((s > s.max() * tol).sum()), float(s2.sum() ** 2 / (s2 ** 2).sum())

    def solve(self, steps=4000, eta=0.5, c0=None):
        """Full-batch descent to the interior minimum used as the operating point."""
        c = np.zeros(self.k) if c0 is None else c0.copy()
        for _ in range(steps):
            c -= eta * self.loss_grad(c)[1]
        return c


def build(seed=0, k=10, n_train=1024, n_probe=384, hidden=(96, 96), backbone_steps=2000):
    Xtr, ytr, Xp, yp = digits_split(seed, n_train, n_probe)
    phi = frozen_backbone(Xtr, ytr, seed=seed, hidden=hidden, steps=backbone_steps)
    return Adapter(phi(Xtr), ytr, phi(Xp), yp, k=k, seed=seed)


# ------------------------------------------------------------------ helpers
def frequencies(r, f0, band=2.0):
    """r rationally independent frequencies filling [f0, f0*band].

    ``frac(sqrt(p))`` for distinct primes is irrational, and the band does not widen with r,
    which is the fix ``dimension_recovery/README.md`` sec. 1.1 identified: if the band widened
    with r, the smoothness null alone would recover the ordering and the experiment would
    prove nothing.  ``f0 * band**linspace(0,1,r)`` looks like it does the same job and does
    not -- it puts the extreme modes at an exact 2:1 ratio, which is one phase, not two.
    """
    if r <= 1:
        return np.array([f0 * np.sqrt(band)])
    a = np.sqrt(PRIMES[:r].astype(float)) % 1.0
    a = (a - a.min()) / (a.max() - a.min()) * 0.94 + 0.03
    return f0 * band ** a


_RESCACHE = {}


def resonance_margin(f, order=3):
    """min over nonzero integer n, |n|_1 <= order, of dist(sum n_j f_j, Z).

    The window must be longer than 1/margin for the torus to look r-dimensional.
    """
    # Enumerating ``range(-order, order + 1) ** r`` is exponential in r and
    # made the k=20 validation effectively impossible.  Only vectors with
    # L1 norm <= ``order`` are needed.  Generate those as sums of at most
    # ``order`` signed unit vectors: O((2r)^order), exact for this definition.
    from itertools import product
    key = (tuple(np.round(f, 12)), order)
    if key in _RESCACHE:
        return _RESCACHE[key]
    best = np.inf
    r = len(f)
    seen = set()
    for degree in range(1, order + 1):
        for inds in product(range(r), repeat=degree):
            for signs in product((-1, 1), repeat=degree):
                n = np.zeros(r, dtype=np.int8)
                for i, sign in zip(inds, signs):
                    n[i] += sign
                key_n = tuple(int(v) for v in n)
                if key_n in seen or not any(n):
                    continue
                seen.add(key_n)
                if np.abs(n).sum() <= order:
                    z = float(np.dot(n, f))
                    best = min(best, abs(z - round(z)))
    _RESCACHE[key] = best
    return best


def rank_pr(A, tol=1e-10):
    """Hard rank and participation ratio of the row cloud of ``A`` (mean removed)."""
    C = np.asarray(A, float)
    C = C - C.mean(0, keepdims=True)
    if C.size == 0:
        return 0, np.nan
    s = np.linalg.svd(C, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0, np.nan
    return int((s > s.max() * np.sqrt(tol)).sum()), float(s2.sum() ** 2 / (s2 ** 2).sum())

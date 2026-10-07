"""Learning dynamics in a zero-sum game built from weakly coupled 2x2 subgames.

Game (player 1 maximises, player 2 minimises), K_tot subgames, x_k = P1's prob. of action 0
in subgame k (logit theta_k), y_k = P2's (logit phi_k):

    u(x, y) = S * [ sum_k 4 a_k (x_k - xs_k)(y_k - ys_k)                    (cyclic subgames)
                  + sum_k a_k (beta_k x_k - gamma_k y_k)                     (transitive subgames)
                  + eps * sum_{k != j} 4 C_kj (x_k - 1/2)(y_j - 1/2) ]       (weak coupling)

A cyclic subgame is a generalised matching-pennies game with interior Nash (xs, ys) in
(0.25, 0.75): MWU/FTRL cycles around it (Mertikopoulos et al. 2018). A transitive subgame
has dominant actions: learning converges monotonically (the transitive axis of a
"spinning top", Czarnecki et al. 2020). The number of LIVE CYCLES (ground truth) is the
number of cyclic subgames: the linearisation at the interior rest point has one pair of
imaginary eigenvalues per cyclic subgame (checked numerically in `linear_cycles`).

Learners (steps scaled by 1/S so that the payoff unit S does not change the dynamics):
  alt   alternating MWU (player 2 sees player 1's new strategy) -- bounded cycles
  sim   simultaneous MWU -- slow outward spiral (Bailey & Piliouras 2018)
  pg    alternating softmax policy gradient (theta += eta * x(1-x) * grad)
  decay alternating MWU with step eta_t = eta0 / sqrt(1 + t / 20000) (chirping cycles)

Observation, one scalar per step:
  payoff  expected payoff of player 1 + exogenous slow transitive drift D(t), estimated
          from B sampled plays (Gaussian with the exact per-play variance / B); B = inf: exact
  winrate P(realised payoff > 0) ~ Phi(mu / sigma_play), estimated from B plays (binomial)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, ndtr

BURN, W, NWIN = 4000, 8192, 2
T_TOTAL = BURN + W * NWIN


@dataclass
class Game:
    K_live: int
    K_tr: int
    a: np.ndarray            # scale per subgame
    xs: np.ndarray           # interior Nash (cyclic) or unused
    ys: np.ndarray
    beta: np.ndarray         # transitive gradients (0 for cyclic)
    gamma: np.ndarray
    cyclic: np.ndarray       # bool
    C: np.ndarray            # coupling matrix (zero diagonal)
    eps: float
    S: float                 # payoff unit
    theta0: np.ndarray
    phi0: np.ndarray
    drift_amp: float
    drift_T: float
    drift_tc: float
    info: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.a)

    # bilinear form u = S*(x^T M y + h^T x + g^T y + c)
    def matrices(self):
        K = self.K
        M = np.zeros((K, K)); h = np.zeros(K); g = np.zeros(K); c = 0.0
        for k in range(K):
            if self.cyclic[k]:
                M[k, k] += 4 * self.a[k]
                h[k] += -4 * self.a[k] * self.ys[k]
                g[k] += -4 * self.a[k] * self.xs[k]
                c += 4 * self.a[k] * self.xs[k] * self.ys[k]
            else:
                h[k] += self.a[k] * self.beta[k]
                g[k] += -self.a[k] * self.gamma[k]
        E = 4 * self.eps * self.C
        M += E
        h += -0.5 * E.sum(1)
        g += -0.5 * E.sum(0)
        c += 0.25 * E.sum()
        return M, h, g, c


def make_game(rng, K_live, K_tr, scale_range=(0.5, 2.0), eps_max=0.15, drift_max=1.0):
    K = K_live + K_tr
    cyclic = np.array([True] * K_live + [False] * K_tr)
    perm = rng.permutation(K)
    cyclic = cyclic[perm]
    a = np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]), K))
    xs = rng.uniform(0.25, 0.75, K)
    ys = rng.uniform(0.25, 0.75, K)
    beta = np.where(cyclic, 0.0, rng.uniform(0.3, 1.0, K))
    gamma = np.where(cyclic, 0.0, rng.uniform(0.3, 1.0, K))
    C = rng.normal(0, 1, (K, K)) * np.sqrt(np.outer(a, a))
    np.fill_diagonal(C, 0)
    eps = rng.uniform(0, eps_max)
    S = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    amp = rng.uniform(0.3, 3.0, K)
    psi = rng.uniform(0, 2 * np.pi, K)
    lx, ly = np.log(xs / (1 - xs)), np.log(ys / (1 - ys))
    theta0 = np.where(cyclic, lx + amp * np.cos(psi), rng.uniform(-6, 0, K))
    phi0 = np.where(cyclic, ly + amp * np.sin(psi), rng.uniform(-6, 0, K))
    # exogenous transitive drift of the payoff (skill difference), in units of the cycles
    drift_amp = rng.uniform(0, drift_max) * float(np.mean(a[cyclic])) if K_live else 0.0
    drift_T = float(np.exp(rng.uniform(np.log(5000), np.log(40000))))
    drift_tc = rng.uniform(0, T_TOTAL)
    return Game(K_live, K_tr, a, xs, ys, beta, gamma, cyclic, C, eps, S, theta0, phi0,
                drift_amp, drift_T, drift_tc)


def simulate(game: Game, learner="alt", eta=0.05, T=T_TOTAL):
    """Return dict with states x, y (T x K) and noiseless expected payoff (T)."""
    M, h, g, c = game.matrices()
    th = game.theta0.astype(float).copy()
    ph = game.phi0.astype(float).copy()
    K = game.K
    X = np.empty((T, K)); Y = np.empty((T, K))
    th_prev_grad = None; ph_prev_grad = None
    for t in range(T):
        x = expit(th); y = expit(ph)
        X[t] = x; Y[t] = y
        e = eta / np.sqrt(1 + t / 20000) if learner == "decay" else eta
        if learner in ("alt", "decay", "pg"):
            gx = M @ y + h
            if learner == "pg":
                th = th + e * x * (1 - x) * gx * 4
            else:
                th = th + e * gx
            x = expit(th)
            gy = M.T @ x + g
            if learner == "pg":
                ph = ph - e * y * (1 - y) * gy * 4
            else:
                ph = ph - e * gy
        elif learner == "sim":
            gx = M @ y + h; gy = M.T @ x + g
            th = th + e * gx; ph = ph - e * gy
        elif learner == "omwu":
            gx = M @ y + h; gy = M.T @ x + g
            if th_prev_grad is None:
                th_prev_grad, ph_prev_grad = gx, gy
            th = th + e * (2 * gx - th_prev_grad); ph = ph - e * (2 * gy - ph_prev_grad)
            th_prev_grad, ph_prev_grad = gx, gy
        else:
            raise ValueError(learner)
    u = game.S * (np.einsum("tk,kj,tj->t", X, M, Y) + X @ h + Y @ g + c)
    return {"x": X, "y": Y, "u": u}


def drift(game: Game, T=T_TOTAL):
    t = np.arange(T)
    return game.S * game.drift_amp * np.tanh((t - game.drift_tc) / game.drift_T)


def play_variance(game: Game, X, Y):
    """Exact variance of one realised play's payoff, ignoring the weak coupling (eps*C)."""
    var = np.zeros(len(X))
    for k in range(game.K):
        x, y = X[:, k], Y[:, k]
        if game.cyclic[k]:
            al = 4 * game.a[k]
            vals = {(s, r): al * (s - game.xs[k]) * (r - game.ys[k]) for s in (0, 1) for r in (0, 1)}
        else:
            vals = {(s, r): game.a[k] * (game.beta[k] * s - game.gamma[k] * r) for s in (0, 1) for r in (0, 1)}
        m1 = np.zeros(len(X)); m2 = np.zeros(len(X))
        for (s, r), v in vals.items():
            p = (x if s else 1 - x) * (y if r else 1 - y)
            m1 += p * v; m2 += p * v * v
        var += m2 - m1 ** 2
    return game.S ** 2 * var


def observe(game: Game, sim: dict, obs="payoff", B=np.inf, rng=None):
    mu = sim["u"] + drift(game, len(sim["u"]))
    if obs == "payoff":
        if np.isinf(B):
            return mu
        sd = np.sqrt(play_variance(game, sim["x"], sim["y"]) / B)
        return mu + rng.normal(0, 1, len(mu)) * sd
    if obs == "winrate":
        sig = np.sqrt(play_variance(game, sim["x"], sim["y"])) + 1e-12
        p = ndtr(mu / sig)
        if np.isinf(B):
            return p
        return rng.binomial(int(B), p) / B
    raise ValueError(obs)


def linear_cycles(game: Game, eta=0.05):
    """Pairs of (near-)imaginary eigenvalues of the continuous-time MWU field at the interior
    rest point of the cyclic block (transitive subgames frozen at their pure limit)."""
    M, h, g, c = game.matrices()
    cy = np.where(game.cyclic)[0]; tr = np.where(~game.cyclic)[0]
    if len(cy) == 0:
        return 0
    x_tr = np.ones(len(tr)); y_tr = np.ones(len(tr))  # dominant actions are action 0 -> x,y -> 1
    # rest point of cyclic block: M_cc y_c + M_ct y_t + h_c = 0 ; M_cc^T x_c + M_tc^T x_t + g_c = 0
    Mcc = M[np.ix_(cy, cy)]
    yc = np.linalg.solve(Mcc, -(M[np.ix_(cy, tr)] @ y_tr + h[cy]))
    xc = np.linalg.solve(Mcc.T, -(M[np.ix_(tr, cy)].T @ x_tr + g[cy]))
    if np.any((xc <= 0) | (xc >= 1) | (yc <= 0) | (yc >= 1)):
        return -1
    n = len(cy)
    J = np.zeros((2 * n, 2 * n))
    J[:n, n:] = Mcc * (yc * (1 - yc))[None, :]
    J[n:, :n] = -Mcc.T * (xc * (1 - xc))[None, :]
    ev = np.linalg.eigvals(J)
    return int(np.sum((np.abs(ev.real) < 1e-8 * (1 + np.abs(ev).max())) & (ev.imag > 1e-9)))


def subgame_freqs(sim: dict, game: Game, a0=BURN, a1=T_TOTAL):
    """Mean frequency (cycles/step) of each cyclic subgame in the monitored span (state-based)."""
    out = []
    for k in np.where(game.cyclic)[0]:
        z = sim["x"][a0:a1, k] - np.median(sim["x"][a0:a1, k])
        s = np.signbit(z)
        out.append(np.count_nonzero(s[1:] != s[:-1]) / 2 / (a1 - a0))
    return np.array(out)


def resonant(freqs, tol=0.03, ratios=(1.0, 2.0, 3.0)):
    f = np.sort(freqs)
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            r = f[j] / max(f[i], 1e-12)
            if any(abs(r - q) < tol * q for q in ratios):
                return True
    return False

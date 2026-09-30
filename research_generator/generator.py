"""E9: a trained recurrent generator whose number of active components is known.

A rate network (the FORCE setting of research_force_motion, Sussillo & Abbott 2009)
is trained by recursive least squares to generate, autonomously, a scalar signal
made of q sinusoids with rationally independent frequencies. A network that has
learned it runs on a q-torus, so the active dimension of its autonomous regime is q
-- known in advance, but realised by training, not written down by hand.

The control that makes the test sharp: four sinusoids that are HARMONICS of one
base frequency. Four spectral lines, but a periodic signal, so the dimension is 1.
A peak counter says 4; a dimension estimator must say 1.

Independent check: the Lyapunov spectrum of the trained autonomous map. A q-torus
has q exponents at zero and the rest negative.
"""
from __future__ import annotations

import time

import numpy as np

DT = 0.1
T0 = 10 * np.pi                        # base period, time units (314 steps)
RATIOS = (1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5), np.sqrt(7))   # rationally independent


def target_spec(kind: str, q: int, seed: int):
    """(frequencies in cycles per time unit, amplitudes, phases)."""
    rng = np.random.default_rng(10_000 + seed)
    if kind == "torus":
        f = np.array(RATIOS[:q]) / T0
    elif kind == "harmonic":           # q harmonics of one base: periodic, dimension 1
        f = np.arange(1, q + 1) / T0
    elif kind == "mixed":              # 2 independent bases, q/2 harmonics each: dimension 2
        f = np.concatenate([np.arange(1, q // 2 + 1) * b for b in (1.0, np.sqrt(2))]) / T0
    else:
        raise ValueError(kind)
    amp = np.full(len(f), 1.2 / np.sqrt(len(f)))
    ph = rng.uniform(0, 2 * np.pi, len(f))
    return f, amp, ph


def target(t, spec):
    f, amp, ph = spec
    return (amp[None, :] * np.sin(2 * np.pi * np.outer(t, f) + ph[None, :])).sum(1)


def init(seed, n, gain=1.5):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(n, n)) * gain / np.sqrt(n)
    u = rng.uniform(-1, 1, size=n)
    x = rng.normal(size=n) * 0.5
    return j, u, x


def train(seed, n, spec, steps, gain=1.5, every=2):
    """FORCE / RLS on the single read-out w; the output is fed back through u."""
    j, u, x = init(seed, n, gain)
    w = np.zeros(n)
    p = np.eye(n)
    t0 = time.perf_counter()
    for it in range(steps):
        r = np.tanh(x)
        x = (1 - DT) * x + DT * (j @ r + u * (w @ r))
        r = np.tanh(x)
        if (it + 1) % every == 0:
            err = w @ r - target(np.array([(it + 1) * DT]), spec)[0]
            pr = p @ r
            c = 1.0 / (1.0 + r @ pr)
            w -= c * err * pr
            p -= c * np.outer(pr, pr)
    return j, u, w, x, time.perf_counter() - t0


def rollout(j, u, w, x, burn, length, observed=(0, 1, 2), n_lyap=12, lyap_len=16384, qr_every=10,
            keep_states=False, state_every=4):
    """Autonomous run with frozen weights. Returns observers, output, top Lyapunov exponents."""
    a = j + (u @ w.T if w.ndim == 2 else np.outer(u, w))
    x = x.copy()
    zs = np.empty((length, w.shape[1])) if w.ndim == 2 else None
    for _ in range(burn):
        x = (1 - DT) * x + DT * (a @ np.tanh(x))
    obs = np.empty((length, len(observed)))
    z = np.empty(length)
    states = np.empty((length // state_every, len(x))) if keep_states else None
    V = np.linalg.qr(np.random.default_rng(7).normal(size=(len(x), n_lyap)))[0]
    logs = np.zeros(n_lyap)
    nq = 0
    t0 = time.perf_counter()
    t_lyap = 0.0
    for i in range(length):
        r = np.tanh(x)
        obs[i] = r[list(observed)]
        if zs is not None:
            zs[i] = r @ w
            z[i] = zs[i].sum()
        else:
            z[i] = w @ r
        if keep_states and i % state_every == 0:
            states[i // state_every] = r
        if i < lyap_len:
            tl = time.perf_counter()
            V = (1 - DT) * V + DT * (a @ ((1 - r ** 2)[:, None] * V))
            if (i + 1) % qr_every == 0:
                V, R = np.linalg.qr(V)
                if i >= 1000:                     # transient of the tangent basis
                    logs += np.log(np.abs(np.diag(R)))
                    nq += 1
            t_lyap += time.perf_counter() - tl
        x = (1 - DT) * x + DT * (a @ r)
    lyap = logs / (nq * qr_every * DT)
    info = {"t_rollout": time.perf_counter() - t0, "t_lyap": t_lyap, "zs": zs}
    if keep_states:
        return obs, z, np.sort(lyap)[::-1], info, states
    return obs, z, np.sort(lyap)[::-1], info


def spectral_fidelity(z, spec, tol=0.03):
    """Share of the output's power within +-tol (relative) of the target lines, and the
    smallest share any single target line carries. A learned network puts nearly all
    power on the target lines, and every line gets its share."""
    zz = z - z.mean()
    P = np.abs(np.fft.rfft(zz * np.hanning(len(zz)))) ** 2
    fr = np.fft.rfftfreq(len(zz), d=DT)
    total = P[1:].sum()
    shares = []
    for f in spec[0]:
        m = np.abs(fr - f) <= tol * f
        shares.append(P[m].sum() / total)
    return float(sum(shares)), float(min(shares))


def peak_count(z, rel=0.05):
    """The naive competitor: how many spectral peaks carry at least `rel` of the power."""
    zz = z - z.mean()
    P = np.abs(np.fft.rfft(zz * np.hanning(len(zz)))) ** 2
    P = P / P[1:].sum()
    peaks = [i for i in range(2, len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    # merge a peak with its immediate leakage: sum a +-3 bin neighbourhood
    mass = sorted((P[max(1, i - 3):i + 4].sum(), i) for i in peaks)[::-1]
    kept = []
    for m, i in mass:
        if m < rel:
            break
        if all(abs(i - k) > 6 for k in kept):
            kept.append(i)
    return len(kept)


# --- multi-output variant: q read-outs, one sinusoid each, each fed back -------------

def init_multi(seed, n, q, gain=1.5):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(n, n)) * gain / np.sqrt(n)
    u = rng.uniform(-1, 1, size=(n, q))
    x = rng.normal(size=n) * 0.5
    return j, u, x


def targets_multi(t, spec):
    """(len(t), q): output i is amp_i sin(2 pi f_i t + phi_i)."""
    f, amp, ph = spec
    return np.sin(2 * np.pi * np.outer(t, f) + ph[None, :])


def train_multi(seed, n, spec, steps, gain=1.5, every=2, alpha=1.0):
    q = len(spec[0])
    j, u, x = init_multi(seed, n, q, gain)
    w = np.zeros((n, q))
    p = np.eye(n) / alpha
    t0 = time.perf_counter()
    for it in range(steps):
        r = np.tanh(x)
        x = (1 - DT) * x + DT * (j @ r + u @ (r @ w))
        r = np.tanh(x)
        if (it + 1) % every == 0:
            err = r @ w - targets_multi(np.array([(it + 1) * DT]), spec)[0]
            pr = p @ r
            c = 1.0 / (1.0 + r @ pr)
            w -= c * np.outer(pr, err)
            p -= c * np.outer(pr, pr)
    return j, u, w, x, time.perf_counter() - t0


def fidelity_multi(zs, spec, tol=0.03):
    """Per output: share of its power within +-tol of its own target line. Returns the minimum."""
    fr = np.fft.rfftfreq(len(zs), d=DT)
    shares = []
    for i, f in enumerate(spec[0]):
        zz = zs[:, i] - zs[:, i].mean()
        P = np.abs(np.fft.rfft(zz * np.hanning(len(zz)))) ** 2
        shares.append(P[np.abs(fr - f) <= tol * f].sum() / P[1:].sum())
    return float(min(shares)), [round(float(s), 3) for s in shares]

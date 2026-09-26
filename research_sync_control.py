"""Learned synchronization of a heterogeneous oscillator network.

The experiment is deliberately a monitoring benchmark: a controller is trained
to synchronize a distributed recurrent system, while the proposed estimator sees
only one local scalar sensor.  Full-state diagnostics are kept as an external
check, not as an input to the estimator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.windows import score
from actdim.estimator.diagnostics import diagnose

OUT = ROOT / "research_sync_control_results"
OUT.mkdir(exist_ok=True)


def graph(n: int):
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for d in (1, 2, 3):
            a[i, (i + d) % n] = 1.0
            a[i, (i - d) % n] = 1.0
    return a / a.sum(1, keepdims=True)


def simulate(theta0, omega, gain, *, steps, dt, A, noise=0.0):
    """First-order inertial-free Kuramoto dynamics with learned extra coupling."""
    theta = np.array(theta0, dtype=np.float64, copy=True)
    n = len(theta)
    rows = np.empty((steps, n + 3), dtype=np.float64)
    for t in range(steps):
        phase = theta[None, :] - theta[:, None]
        coupling = (A * np.sin(phase)).sum(1)
        velocity = omega + (0.10 + gain) * coupling
        if noise:
            velocity = velocity + noise * np.random.default_rng(t + 17).normal(size=n)
        theta += dt * velocity
        order = np.abs(np.exp(1j * theta).mean())
        rows[t, :n] = theta
        rows[t, n:] = (np.sin(theta[0]), np.cos(theta[0]), order)
    return rows


def train_controller(seed=0, n=32, steps=420, dt=0.03):
    """Learn a single local coupling gain by differentiating through rollouts."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A = torch.tensor(graph(n), dtype=torch.float64)
    omega = torch.tensor(1.0 + rng.normal(0, 0.20, size=n), dtype=torch.float64)
    phases = torch.tensor(rng.uniform(-np.pi, np.pi, size=(10, n)), dtype=torch.float64)
    log_gain = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_gain], lr=0.08)
    hist = []
    for it in range(steps):
        theta = phases.clone()
        gain = torch.nn.functional.softplus(log_gain)
        for _ in range(180):
            diff = theta[:, None, :] - theta[:, :, None]
            coupling = (A[None] * torch.sin(diff)).sum(2)
            theta = theta + dt * (omega[None] + (0.10 + gain) * coupling)
        z = torch.exp(1j * theta)
        coherence = torch.abs(z.mean(1))
        # The task is synchronization; the second term prevents an unlimited gain.
        loss = ((1.0 - coherence) ** 2).mean() + 0.006 * gain**2
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 10 == 0 or it == steps - 1:
            hist.append((it, float(gain.detach()), float(loss.detach()),
                         float(coherence.mean().detach())))
    return float(torch.nn.functional.softplus(log_gain).detach()), np.asarray(hist), omega.numpy()


def pr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False) ** 2
    return float(s.sum() ** 2 / np.sum(s ** 2))


def lyap_proxy(theta, omega, gain, A, dt=0.03, eps=1e-7):
    """Finite-difference largest exponent proxy along a trajectory."""
    x = np.asarray(theta, dtype=np.float64).copy()
    v = np.random.default_rng(123).normal(size=len(x)); v /= np.linalg.norm(v)
    logs = []
    for _ in range(len(x) - 1):
        def f(y):
            return y + dt * (omega + (0.10 + gain) * (A * np.sin(y[None, :] - y[:, None])).sum(1))
        y = f(x)
        y2 = f(x + eps * v)
        v = (y2 - y) / eps
        norm = np.linalg.norm(v)
        if norm > 0:
            logs.append(np.log(norm) / dt); v /= norm
        x = y
    return float(np.mean(logs[-max(100, len(logs)//2):]))


def measure(stage, gain, omega, A, sensor_a, sensor_b, seed=0, steps=18000, dt=0.03):
    rng = np.random.default_rng(seed + 900)
    theta0 = rng.uniform(-np.pi, np.pi, size=len(omega))
    rows = simulate(theta0, omega, gain, steps=steps, dt=dt, A=A)
    # Discard the initial transient.  The estimator receives one fixed scalar
    # aggregate sensor; it never sees the full state used by the audit.
    cut = 2000
    sensor = (np.sin(rows[cut:, :len(omega)]) @ sensor_a
              + np.cos(rows[cut:, :len(omega)]) @ sensor_b) / np.sqrt(len(omega))
    cfg = EstimatorConfig(max_E=12, tau=20, k_neighbors=8,
                          theiler=700, theiler_cap=700,
                          window=8192, stride=4096)
    try:
        est = score(sensor[-8192:], cfg, seed=seed)
        diag = diagnose(sensor[-8192:], cfg, seed=seed)
        mg, lb, prd = est["MG"], est["LB"], est["PRdelay"]
        deg = est["degenerate"]
    except Exception as exc:
        mg = lb = prd = np.nan; deg = True; diag = None; exc_name = type(exc).__name__
    else:
        exc_name = ""
    x = np.column_stack([np.sin(rows[cut:, :len(omega)]),
                         np.cos(rows[cut:, :len(omega)]),
                         np.gradient(rows[cut:, :len(omega)], dt, axis=0)])
    order = rows[cut:, -1]
    out = {
        "stage": stage, "gain": gain, "sensor_MG": mg, "sensor_LB": lb,
        "sensor_PRdelay": prd, "degenerate": deg, "order_mean": order.mean(),
        "order_sd": order.std(), "state_PR": pr(x[::4]),
        "phase_sd": np.std(np.angle(np.exp(1j * rows[-1, :len(omega)]))),
        "sensor_std": sensor[-8192:].std(), "error": exc_name,
    }
    if diag is not None:
        out["ident_ratio"] = diag.identifiability_ratio
        out["trend_crossings"] = diag.trend_crossings
    np.savez_compressed(OUT / f"trace_{stage}_s{seed}.npz", sensor=sensor,
                        order=order, state=rows[cut:, :len(omega)])
    return out


def main():
    n, dt = 32, 0.03
    A = graph(n)
    gains = []
    histories = []
    omegas = []
    for seed in (0, 1, 2):
        g, h, om = train_controller(seed=seed, n=n)
        gains.append(g); histories.append(h); omegas.append(om)
    # Every seed defines one plant and one fixed sensor.  The same sensor is used
    # before, during, and after training for that plant.
    measurements = []
    for seed, gain, omega in zip((0, 1, 2), gains, omegas):
        rng = np.random.default_rng(seed + 1700)
        a = rng.normal(size=n); b = rng.normal(size=n)
        a /= np.linalg.norm(a); b /= np.linalg.norm(b)
        stages = [("before", 0.0), ("mid", gain * 0.45), ("after", gain)]
        measurements.extend(measure(name, g, omega, A, a, b, seed=seed, dt=dt)
                            for name, g in stages)
    pd.DataFrame(measurements).to_csv(OUT / "summary.csv", index=False)
    np.savetxt(OUT / "training_history.csv", histories[0], delimiter=",")
    np.save(OUT / "learned_gains.npy", np.asarray(gains))
    np.save(OUT / "omega.npy", omega)
    (OUT / "metadata.json").write_text(json.dumps({
        "system": "N=32 heterogeneous first-order oscillator network",
        "objective": "learn a distributed coupling gain to synchronize phases",
        "sensor": "one fixed scalar aggregate a^T sin(theta)+b^T cos(theta)",
        "seeds": [0, 1, 2], "learned_gains": gains,
        "dt": dt, "steps": 18000, "window": 8192, "delay": 20,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(measurements).to_string(index=False))
    print("learned gains", gains)


if __name__ == "__main__":
    main()

"""Cost benchmark: scalar-log versus full-state Lyapunov analysis.

The benchmark uses the same dense Kuramoto-like recurrent system and the same
stored trajectories for all methods.  The scalar method sees one fixed scalar
projection.  Lyapunov methods see the full N-dimensional phase trajectory and
propagate tangent vectors or a tangent basis.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.windows import score

OUT = ROOT / "research_sync_control_results"
OUT.mkdir(exist_ok=True)


def complete_graph(n: int) -> np.ndarray:
    a = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    return a / (n - 1)


def rollout(n: int, gain: float, steps: int, seed: int, dt: float = 0.03):
    rng = np.random.default_rng(seed)
    a = complete_graph(n)
    omega = 1.0 + 0.20 * rng.normal(size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    state = np.empty((steps, n), dtype=np.float64)
    sensor_a = rng.normal(size=n); sensor_b = rng.normal(size=n)
    sensor_a /= np.linalg.norm(sensor_a); sensor_b /= np.linalg.norm(sensor_b)
    sensor = np.empty(steps, dtype=np.float64)
    kappa = 0.10 + gain
    for t in range(steps):
        diff = theta[None, :] - theta[:, None]
        theta += dt * (omega + kappa * (a * np.sin(diff)).sum(1))
        state[t] = theta
        sensor[t] = (sensor_a @ np.sin(theta) + sensor_b @ np.cos(theta)) / np.sqrt(n)
    return state, sensor, omega, a, dt


def discrete_jacobian(theta, gain, a, dt):
    """Jacobian of one Euler step for the phase dynamics."""
    c = np.cos(theta[None, :] - theta[:, None])
    kappa = 0.10 + gain
    d = kappa * a * c
    j = d.copy()
    np.fill_diagonal(j, -d.sum(axis=1))
    return np.eye(len(theta)) + dt * j


def largest_lyapunov(state, gain, a, dt, burn=256):
    n = state.shape[1]
    v = np.random.default_rng(123).normal(size=n)
    v /= np.linalg.norm(v)
    logs = []
    for t in range(state.shape[0] - 1):
        v = discrete_jacobian(state[t], gain, a, dt) @ v
        norm = np.linalg.norm(v)
        v /= norm
        if t >= burn:
            logs.append(np.log(norm) / dt)
    return float(np.mean(logs))


def full_spectrum(state, gain, a, dt, burn=256, qr_stride=4):
    """All exponents using QR re-orthogonalisation of the full tangent basis."""
    n = state.shape[1]
    q = np.eye(n)
    sums = np.zeros(n)
    count = 0
    for t in range(state.shape[0] - 1):
        q = discrete_jacobian(state[t], gain, a, dt) @ q
        if (t + 1) % qr_stride == 0:
            q, r = np.linalg.qr(q, mode="reduced")
            if t >= burn:
                sums += np.log(np.maximum(np.abs(np.diag(r)), 1e-300))
                count += qr_stride
    return sums / (count * dt)


def scalar_log(sensor, seed, window=1536):
    cfg = EstimatorConfig(max_E=8, tau=8, k_neighbors=8,
                          theiler=160, theiler_cap=160,
                          window=window, stride=window)
    out = score(sensor[-window:], cfg, seed=seed)
    return float(out["MG"]), float(out["LB"]), bool(out["degenerate"])


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t0


def main():
    rows = []
    spectrum_rows = []
    sizes = (16, 32, 64, 96)
    steps = 2048
    for n in sizes:
        for stage, gain in (("before", 0.0), ("after", 1.8)):
            state, sensor, omega, a, dt = rollout(n, gain, steps, seed=41 + n)
            mg, scalar_seconds = timed(lambda: scalar_log(sensor, seed=41 + n))
            lyap, lyap_seconds = timed(lambda: largest_lyapunov(state, gain, a, dt))
            stored_scalar = sensor.nbytes
            stored_full = state.nbytes
            rows.append({
                "n": n, "stage": stage, "steps": steps,
                "MG": mg[0], "LB": mg[1], "degenerate": mg[2],
                "lambda_max": lyap,
                "scalar_seconds": scalar_seconds,
                "largest_lyap_seconds": lyap_seconds,
                "scalar_bytes": stored_scalar,
                "full_state_bytes": stored_full,
                "storage_ratio": stored_full / stored_scalar,
            })
            if n <= 64:
                spectrum, spectrum_seconds = timed(
                    lambda: full_spectrum(state, gain, a, dt))
                spectrum_rows.append({
                    "n": n, "stage": stage,
                    "spectrum_seconds": spectrum_seconds,
                    "spectrum_max": float(np.max(spectrum)),
                    "spectrum_min": float(np.min(spectrum)),
                    "spectrum_sum": float(np.sum(spectrum)),
                })
            print(rows[-1])
    df = pd.DataFrame(rows)
    sf = pd.DataFrame(spectrum_rows)
    df.to_csv(OUT / "computation_benchmark.csv", index=False)
    sf.to_csv(OUT / "lyapunov_spectrum_benchmark.csv", index=False)
    (OUT / "computation_benchmark_metadata.json").write_text(json.dumps({
        "system": "dense complete-graph heterogeneous phase oscillator network",
        "sizes": sizes, "steps": steps, "scalar_window": 1536,
        "methods": ["scalar MG/LB", "largest Lyapunov", "full Lyapunov spectrum"],
        "note": "timings are analysis-only after the same state trajectory was generated",
    }, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for stage, color in (("before", "#b2182b"), ("after", "#2166ac")):
        g = df[df.stage == stage]
        ax[0].plot(g.n, g.scalar_seconds, "o-", color=color, label=f"MG/LB {stage}")
        ax[0].plot(g.n, g.largest_lyap_seconds, "s--", color=color,
                   alpha=.75, label=f"largest Lyapunov {stage}")
        ax[1].plot(g.n, g.storage_ratio, "o-", color=color, label=stage)
        ax[2].plot(g.n, g.MG, "o-", color=color, label=f"MG {stage}")
        ax[2].plot(g.n, g.lambda_max, "s--", color=color, label=f"$\\lambda_{{max}}$ {stage}")
    ax[0].set_ylabel("analysis time (s)"); ax[0].set_xlabel("N")
    ax[0].set_title("Analysis time")
    ax[1].set_ylabel("full-state / scalar storage"); ax[1].set_xlabel("N")
    ax[1].set_title("Storage overhead")
    ax[2].set_ylabel("value"); ax[2].set_xlabel("N")
    ax[2].set_title("Signals of simplification")
    for axy in ax:
        axy.grid(alpha=.25); axy.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "computation_benchmark.pdf", bbox_inches="tight")
    fig.savefig(OUT / "computation_benchmark.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()

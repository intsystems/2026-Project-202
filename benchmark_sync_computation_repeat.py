"""Repeated timing benchmark for scalar-log and Lyapunov diagnostics."""
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
from benchmark_sync_computation import (complete_graph, rollout,
                                         largest_lyapunov, full_spectrum)

OUT = ROOT / "research_sync_control_results"


def scalar_log(sensor, seed, window=1536):
    cfg = EstimatorConfig(max_E=8, tau=8, k_neighbors=8,
                          theiler=160, theiler_cap=160,
                          window=window, stride=window)
    out = score(sensor[-window:], cfg, seed=seed)
    return float(out["MG"]), float(out["LB"]), bool(out["degenerate"])


def median_time(fn, repeats=5):
    fn()  # warm-up
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter(); value = fn(); values.append(time.perf_counter() - t0)
    return value, float(np.median(values)), values


def main():
    rows, spectra = [], []
    sizes = (32, 64, 96, 128)
    steps = 1536
    for n in sizes:
        for stage, gain in (("before", 0.0), ("after", 1.8)):
            state, sensor, omega, a, dt = rollout(n, gain, steps, seed=410 + n)
            mg, scalar_s, scalar_reps = median_time(
                lambda: scalar_log(sensor, seed=410 + n))
            ly, lyap_s, lyap_reps = median_time(
                lambda: largest_lyapunov(state, gain, a, dt), repeats=3)
            spectrum, spectrum_s, spectrum_reps = median_time(
                lambda: full_spectrum(state, gain, a, dt), repeats=3)
            row = {
                "n": n, "stage": stage, "steps": steps,
                "MG": mg[0], "LB": mg[1], "degenerate": mg[2],
                "lambda_max": ly, "full_spectrum_max": float(np.max(spectrum)),
                "scalar_seconds": scalar_s, "largest_lyap_seconds": lyap_s,
                "full_spectrum_seconds": spectrum_s,
                "scalar_bytes": sensor.nbytes, "full_state_bytes": state.nbytes,
                "storage_ratio": state.nbytes / sensor.nbytes,
                "scalar_repeats": scalar_reps, "largest_repeats": lyap_reps,
                "spectrum_repeats": spectrum_reps,
            }
            rows.append(row)
            print(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "computation_benchmark_repeated.csv", index=False)
    (OUT / "computation_benchmark_repeated_metadata.json").write_text(json.dumps({
        "system": "dense complete-graph heterogeneous phase oscillator network",
        "sizes": sizes, "steps": steps, "repeats": 5,
        "timing": "analysis only; trajectories are generated once and reused",
        "methods": ["scalar MG/LB", "largest Lyapunov exponent",
                    "full Lyapunov spectrum"],
    }, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for stage, color in (("before", "#b2182b"), ("after", "#2166ac")):
        g = df[df.stage == stage]
        ax[0].plot(g.n, g.scalar_seconds, "o-", color=color, label=f"scalar-log, {stage}")
        ax[0].plot(g.n, g.full_spectrum_seconds, "s--", color=color,
                   label=f"full Lyapunov spectrum, {stage}")
        ax[1].plot(g.n, g.storage_ratio, "o-", color=color, label=stage)
        ax[2].plot(g.n, g.MG, "o-", color=color, label=f"MG, {stage}")
        ax[2].plot(g.n, g.full_spectrum_max, "s--", color=color,
                   label=f"max spectrum, {stage}")
    ax[0].set(xlabel="N", ylabel="median analysis time (s)", title="Analysis cost")
    ax[1].set(xlabel="N", ylabel="full-state / scalar storage", title="Storage cost")
    ax[2].set(xlabel="N", ylabel="diagnostic value", title="Simplification signal")
    for axy in ax:
        axy.grid(alpha=.25); axy.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "computation_benchmark_repeated.pdf", bbox_inches="tight")
    fig.savefig(OUT / "computation_benchmark_repeated.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()

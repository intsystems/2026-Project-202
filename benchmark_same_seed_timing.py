"""Fair same-seed timing: scalar-log MG versus full Lyapunov spectrum."""
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
from benchmark_sync_computation import complete_graph, full_spectrum

OUT = ROOT / "research_sync_control_results"


def make_trajectory(n: int, gain: float, steps: int, seed: int, dt: float = 0.03):
    rng = np.random.default_rng(seed)
    omega = 1.0 + 0.20 * rng.normal(size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    a = rng.normal(size=n); b = rng.normal(size=n)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    graph = complete_graph(n)
    state = np.empty((steps, n), dtype=np.float64)
    sensor = np.empty(steps, dtype=np.float64)
    for t in range(steps):
        diff = theta[None, :] - theta[:, None]
        theta += dt * (omega + (0.10 + gain) * (graph * np.sin(diff)).sum(1))
        state[t] = theta
        sensor[t] = (a @ np.sin(theta) + b @ np.cos(theta)) / np.sqrt(n)
    return state, sensor, graph, dt


def run_mg(sensor, seed, window=4096):
    cfg = EstimatorConfig(max_E=12, tau=20, k_neighbors=8,
                          theiler=700, theiler_cap=700,
                          window=window, stride=window)
    out = score(sensor[-window:], cfg, seed=seed)
    return float(out['MG']), bool(out['degenerate'])


def timed(fn, repeats):
    fn()  # warm-up
    values, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, float(np.median(values)), values


def main():
    n = 64
    seed = 474
    steps = 8192
    window = 4096
    gains = (0.0, 0.3, 0.5, 1.8)
    rows = []
    for gain in gains:
        state, sensor, graph, dt = make_trajectory(n, gain, steps, seed)
        mg, mg_seconds, mg_repeats = timed(
            lambda: run_mg(sensor, seed, window), repeats=7)
        spectrum, lyap_seconds, lyap_repeats = timed(
            lambda: full_spectrum(state[-window:], gain, graph, dt,
                                  burn=512, qr_stride=4), repeats=5)
        rows.append({
            'seed': seed, 'n': n, 'gain': gain, 'window': window,
            'MG': mg[0], 'MG_degenerate': mg[1],
            'lambda_max': float(np.max(spectrum)),
            'n_positive': int(np.sum(spectrum > 1e-4)),
            'n_weak': int(np.sum(spectrum > -1e-4)),
            'MG_seconds': mg_seconds,
            'Lyapunov_spectrum_seconds': lyap_seconds,
            'speedup_full_spectrum_over_MG': lyap_seconds / mg_seconds,
            'scalar_bytes': sensor[-window:].nbytes,
            'full_state_bytes': state[-window:].nbytes,
            'storage_ratio': state[-window:].nbytes / sensor[-window:].nbytes,
            'MG_repeats': mg_repeats,
            'Lyapunov_repeats': lyap_repeats,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'same_seed_timing.csv', index=False)
    (OUT / 'same_seed_timing_metadata.json').write_text(json.dumps({
        'seed': seed, 'n': n, 'steps_generated': steps,
        'window_analyzed': window, 'fixed_across_gains': True,
        'timing': 'analysis only; one trajectory generated and reused per gain',
        'MG_repeats': 7, 'Lyapunov_repeats': 5,
    }, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(frame))
    width = .34
    ax.bar(x - width / 2, frame.MG_seconds, width, label='MG / scalar log', color='#2166ac')
    ax.bar(x + width / 2, frame.Lyapunov_spectrum_seconds, width,
           label='full Lyapunov spectrum', color='#b2182b')
    ax.set_xticks(x, [f'g={g:g}' for g in frame.gain])
    ax.set_ylabel('median analysis time (s)')
    ax.set_xlabel('same fixed seed and same plant')
    ax.set_title('Analysis time on the same trajectories')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    for i, row in frame.iterrows():
        ax.text(i + width / 2, row.Lyapunov_spectrum_seconds,
                f"{row.speedup_full_spectrum_over_MG:.1f}x",
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'same_seed_timing.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'same_seed_timing.png', dpi=240, bbox_inches='tight')
    print(frame[['gain', 'MG', 'n_positive', 'n_weak', 'MG_seconds',
                 'Lyapunov_spectrum_seconds', 'speedup_full_spectrum_over_MG',
                 'storage_ratio']].to_string(index=False))


if __name__ == '__main__':
    main()

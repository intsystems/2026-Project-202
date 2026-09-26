"""Scaling benchmark on identical trajectory windows.

This benchmark is deliberately separate from the transition plot. It checks
whether the scalar estimator becomes cheaper than the full Lyapunov spectrum
once the hidden state is large enough, without hiding the small-N crossover.
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
from benchmark_sync_computation import complete_graph, full_spectrum

OUT = ROOT / "research_sync_control_results"


def make_trajectory(n, gain, steps, seed, dt=.03):
    rng = np.random.default_rng(seed)
    omega = 1.0 + .20 * rng.normal(size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    a = rng.normal(size=n); b = rng.normal(size=n)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    A = complete_graph(n)
    state = np.empty((steps, n)); sensor = np.empty(steps)
    for t in range(steps):
        diff = theta[None, :] - theta[:, None]
        theta += dt * (omega + (.10 + gain) * (A * np.sin(diff)).sum(1))
        state[t] = theta
        sensor[t] = (a @ np.sin(theta) + b @ np.cos(theta)) / np.sqrt(n)
    return state, sensor, A, dt


def mg(sensor, seed, window):
    cfg = EstimatorConfig(max_E=12, tau=20, k_neighbors=8,
                          theiler=700, theiler_cap=700,
                          window=window, stride=window)
    r = score(sensor[-window:], cfg, seed=seed)
    return float(r['MG']), bool(r['degenerate'])


def measure(fn, repeats):
    fn()
    ts, value = [], None
    for _ in range(repeats):
        t0 = time.perf_counter(); value = fn(); ts.append(time.perf_counter() - t0)
    return value, float(np.median(ts))


def main():
    window = steps = 2048
    seed = 474
    gain = .5
    rows = []
    for n in (64, 96, 128, 160):
        state, sensor, A, dt = make_trajectory(n, gain, steps, seed + n)
        (mg_value, mg_degenerate), mg_time = measure(
            lambda: mg(sensor, seed + n, window), repeats=5)
        spectrum, lyap_time = measure(
            lambda: full_spectrum(state, gain, A, dt, burn=256, qr_stride=4), repeats=3)
        rows.append({
            'n': n, 'window': window, 'gain': gain,
            'MG': mg_value, 'MG_degenerate': mg_degenerate,
            'lambda_max': float(np.max(spectrum)),
            'n_positive': int(np.sum(spectrum > 1e-4)),
            'MG_seconds': mg_time,
            'Lyapunov_spectrum_seconds': lyap_time,
            'Lyapunov_over_MG': lyap_time / mg_time,
            'storage_ratio': state.nbytes / sensor.nbytes,
        })
        print(rows[-1])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'same_seed_timing_scaling.csv', index=False)
    (OUT / 'same_seed_timing_scaling_metadata.json').write_text(json.dumps({
        'seed_base': seed, 'gain': gain, 'window': window,
        'same_window_and_analysis': True,
        'note': 'small-N crossover is reported; no universal speedup is assumed',
    }, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(frame.n, frame.MG_seconds, 'o-', label='MG / scalar log', color='#2166ac')
    ax.plot(frame.n, frame.Lyapunov_spectrum_seconds, 's-',
            label='full Lyapunov spectrum', color='#b2182b')
    ax.set(xlabel='system size N', ylabel='median analysis time (s)',
           title='Scaling on identical trajectory windows')
    ax.grid(alpha=.25); ax.legend()
    for _, r in frame.iterrows():
        ax.annotate(f'{r.Lyapunov_over_MG:.1f}x',
                    (r.n, r.Lyapunov_spectrum_seconds), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'same_seed_timing_scaling.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'same_seed_timing_scaling.png', dpi=240, bbox_inches='tight')


if __name__ == '__main__':
    main()

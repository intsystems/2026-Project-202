"""Dense full-rank supervised regression with analytically countable HB phases.

Run from anywhere: python research_known_modes.py
No existing experiments or article files are modified.
"""
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
from actdim.estimator.diagnostics import trend_crossings

OUT = ROOT / 'research_known_modes_results'
PRIMES = np.array([2, 3, 5, 7, 11, 13])
ETA = 1.0
E = 16
K = 10
TAU = 1
THEILER = 100


def simulate(r, seed, steps=12288, regime='stationary'):
    rng = np.random.default_rng(seed)
    n, d = 256, 32
    q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    a = np.linalg.qr(rng.normal(size=(n, d)))[0]
    groups = np.array_split(np.arange(d), r)
    omega = 2 * np.pi * np.sqrt(PRIMES[:r]) / 32
    lam = 2 * (1 - np.cos(omega)) / ETA
    spectrum = np.empty(d)
    teacher_coords = np.empty(d)
    directions = np.zeros((d, r))
    amplitudes = np.empty(r)
    for j, g in enumerate(groups):
        spectrum[g] = lam[j]
        direction = rng.normal(size=len(g))
        direction /= np.linalg.norm(direction)
        directions[g, j] = direction
        # Equal initial loss per frequency; changes no estimator settings.
        amplitudes[j] = 1 / np.sqrt(r * lam[j])
        teacher_coords[g] = direction * amplitudes[j]
    x = np.sqrt(n) * (a * np.sqrt(spectrum)) @ q.T
    teacher = q @ teacher_coords
    y = x @ teacher
    h = x.T @ x / n
    b = x.T @ y / n
    # An independent labelled test set from the same feature covariance.
    xtest = (rng.normal(size=(256, d)) * np.sqrt(spectrum)) @ q.T
    ytest = xtest @ teacher
    probe = xtest[0]
    w = np.zeros(d)
    velocity = np.zeros(d)
    mode_basis = q @ directions
    rows, states, errors, velocities = [], [], [], []
    switch = 8192
    keep = min(2, r)
    suppressed = np.concatenate(groups[keep:]) if r > keep else np.array([], int)
    projector = q[:, suppressed] @ q[:, suppressed].T
    direct_gradient_error = 0.0
    for t in range(steps):
        residual = w - teacher
        e = residual @ mode_basis
        v = velocity @ mode_basis
        # Loss, parameters, momentum and mode coordinates at the SAME time.
        loss = 0.5 * residual @ h @ residual
        rows.append((t, loss, probe @ residual, np.linalg.norm(w),
                     np.linalg.norm(velocity),
                     0.5 * np.mean((xtest @ w - ytest)**2)))
        states.append(w.copy())
        errors.append(e)
        velocities.append(v)
        grad = h @ w - b
        if t in (0, 17, steps - 1):
            direct_gradient_error = max(direct_gradient_error,
                float(np.max(np.abs(grad - x.T @ (x @ w - y) / n))))
        if regime == 'damped':
            velocity *= 0.99
        elif regime == 'switch' and t >= switch:
            # Explicit controlled intervention, NOT spontaneous simplification.
            velocity -= 0.02 * (projector @ velocity)
        velocity += grad
        w -= ETA * velocity
    rows, states = np.array(rows), np.array(states)
    errors, velocities = np.array(errors), np.array(velocities)
    prev = errors + ETA * velocities
    # Exact invariant for beta=1: q_t^2+q_{t-1}^2-2 cos(omega)q_t q_{t-1}.
    invariant = errors**2 + prev**2 - 2 * np.cos(omega) * errors * prev
    coordinate_residual = (states - teacher) - errors @ mode_basis.T
    audited = slice(None) if regime == 'stationary' else slice(0, switch)
    audit = dict(r=r, seed=seed, regime=regime, data_rank=int(np.linalg.matrix_rank(x)),
                 parameter_count=d, steps=steps, initial_loss=rows[0, 1],
                 final_loss=rows[-1, 1], initial_test_loss=rows[0, 5],
                 final_test_loss=rows[-1, 5],
                 gradient_max_error=direct_gradient_error,
                 off_mode_max_error=float(np.abs(coordinate_residual).max()),
                 invariant_relative_drift=float(np.max(np.abs(
                     invariant[audited] / invariant[0] - 1))),
                 min_parameter_sd=float(states.std(axis=0).min()),
                 hessian_error=float(np.max(np.abs(h - (q * spectrum) @ q.T))))
    if regime == 'stationary':
        t = np.arange(steps)[:, None]
        analytical = -amplitudes * np.cos((t + 0.5) * omega) / np.cos(omega / 2)
        audit['analytic_trajectory_max_error'] = float(np.abs(errors-analytical).max())
    return rows, states, errors, velocities, invariant, audit


def measure(series, dither_seed, **tags):
    # score/estimate consumes its complete argument; slice windows explicitly.
    cfg = EstimatorConfig(max_E=E, tau=TAU, k_neighbors=K,
                          theiler=THEILER, theiler_cap=THEILER)
    low = estimate(series, cfg, dither_seed)
    high = estimate(series, cfg.replace(max_E=2*E), dither_seed)
    ratio = high.MG / low.MG if np.isfinite(low.MG) and low.MG > 0 else np.nan
    return dict(**tags, window=len(series), MG=low.MG, MG_2E=high.MG,
                ident_ratio=ratio, degenerate=low.degenerate,
                crossings=trend_crossings(series), sd=float(series.std()))


def main():
    OUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    records, audits = [], []
    for r in (1, 2, 4, 6):
        for seed in (0, 1, 2):
            rows, states, errors, velocities, inv, audit = simulate(r, seed)
            audits.append(audit)
            np.savez_compressed(OUT/f'stationary_r{r}_s{seed}.npz',
                logs=rows, weights=states, mode_error=errors,
                mode_velocity=velocities, invariant=inv)
            for window in (2048, 8192):
                for obs, col in [('loss', 1), ('probe_error', 2)]:
                    records.append(measure(rows[-window:, col], seed,
                        regime='stationary', r=r, seed=seed, observer=obs))
            print(f'stationary r={r} seed={seed}', flush=True)
    for regime in ('switch', 'damped'):
        for seed in (0, 1, 2):
            rows, states, errors, velocities, inv, audit = simulate(4, seed, 20480, regime)
            audits.append(audit)
            np.savez_compressed(OUT/f'{regime}_r4_s{seed}.npz',
                logs=rows, weights=states, mode_error=errors,
                mode_velocity=velocities, invariant=inv)
            segments = [('before', 0, 8192), ('after', 12288, 20480)]
            for phase, a, b in segments:
                for obs, col in [('loss', 1), ('probe_error', 2)]:
                    records.append(measure(rows[a:b, col], seed,
                        regime=regime, r=4, seed=seed, phase=phase, observer=obs))
            if regime == 'switch' and seed == 0:
                for end in range(4096, 20481, 2048):
                    records.append(measure(rows[end-4096:end, 1], seed,
                        regime='switch_trace', r=4, seed=seed,
                        end=end, observer='loss'))
            print(f'{regime} seed={seed}', flush=True)
    df, aud = pd.DataFrame(records), pd.DataFrame(audits)
    df.to_csv(OUT/'measurements.csv', index=False)
    aud.to_csv(OUT/'audits.csv', index=False)
    # An amplitude-only control; an offline transformation, not another training run.
    base = np.load(OUT/'stationary_r4_s0.npz')['logs'][-8192:, 1]
    scales = [measure(base * scale, 0, regime='amplitude_only',
                      scale=scale, observer='loss') for scale in (1., 0.01)]
    pd.DataFrame(scales).to_csv(OUT/'amplitude_control.csv', index=False)
    stationary = df[df.regime == 'stationary']
    summary = stationary.groupby(['r','window','observer']).agg(
        median=('MG','median'), minimum=('MG','min'), maximum=('MG','max'),
        ident=('ident_ratio','median'), degenerate_fraction=('degenerate','mean'))
    summary.to_csv(OUT/'stationary_summary.csv')
    metadata = dict(eta=ETA, E=E, doubled_E=2*E, k=K, tau=TAU,
        theiler=THEILER, seeds=[0,1,2], dimension=32, samples=256,
        spectral_primes=PRIMES.tolist(), frequency_denominator=32,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stationary_steps=12288, stationary_windows=[2048,8192],
        training_log_columns=['step','loss','probe_error','parameter_norm',
                              'momentum_norm','test_loss'],
        seed_scope='dense coordinates and test probe; identical scalar loss dynamics',
        reference='independent irrational HB phases; not target matrix rank',
        switch_step=8192, switch_beta_suppressed=0.98, switch_kept_modes=2,
        seconds=time.perf_counter()-start)
    (OUT/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for window in (2048,8192):
        g=summary.xs((window,'loss'),level=('window','observer'))
        axs[0].plot(g.index,g['median'], 'o-',label=f'W={window}')
        axs[0].fill_between(g.index,g['minimum'],g['maximum'],alpha=.12)
    axs[0].plot([1,6],[1,6],'k--',label='Analytical dimension')
    axs[0].set(xlabel='Independent phases',ylabel='Scalar-loss MG')
    axs[0].legend(fontsize=8)
    tr=df[df.regime=='switch_trace']
    axs[1].plot(tr.end,tr.MG,'o-',label='MG (W=4096)')
    axs[1].axvline(8192,color='k',ls='--',label='Selective damping starts')
    axs[1].set(xlabel='Right edge (training step)',ylabel='Scalar-loss MG')
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'overview.png',dpi=170)
    fig.savefig(OUT/'overview.pdf')
    print(summary.to_string(),flush=True)
    print(df[df.regime.isin(['switch','damped'])].to_string(index=False),flush=True)
    print(f'Total {metadata["seconds"]:.1f}s',flush=True)


if __name__ == '__main__':
    main()

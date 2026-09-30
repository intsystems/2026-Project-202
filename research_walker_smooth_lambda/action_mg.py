"""Frozen scalar settings from the previous experiment; paired eligible tests.

Report invalid/fallen records separately. Aggregate resets within each seed,
then seeds; resets are not independent training replications.
"""
from pathlib import Path
import concurrent.futures
import time
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'research_walker_phase_wide'))
from features import measure
SEEDS = [231, 232, 233, 234, 235]
COEFS = [0, .25, 1, 4]
W, TAU = 2048, 8
ENDS = (2048, 3072, 4096)

def label(seed, coef):
    return f'seed{seed}_lambda{coef:g}'

def analyze(job):
    seed, coef, reset = job
    root = H / label(seed, coef) / 'step1048576' / f'reset{reset}'
    cache = root / 'action_MG_windows.csv'
    if not cache.exists():
        with np.load(root / 'trajectory.npz') as data:
            a = np.asarray(data['actions'], dtype=float)
        assert a.shape == (4096, 6), (root, a.shape)
        signals = dict(action_norm=np.linalg.norm(a, axis=1),
                       delta_action_norm=np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=1),
                       mean_action=a.mean(axis=1))
        rows = []
        with threadpool_limits(limits=1):
            for signal, x in signals.items():
                for end in ENDS:
                    start = time.perf_counter()
                    result = measure(x[end-W:end], W, TAU)
                    rows.append(dict(signal=signal, end=end, window=W, tau=TAU,
                                     total_seconds=time.perf_counter()-start, **result))
        pd.DataFrame(rows).to_csv(cache, index=False)
    d = pd.read_csv(cache)
    rows = []
    for signal, g in d.groupby('signal'):
        ok = g.MG.notna() & np.isfinite(g.MG) & ~g.degenerate
        rows.append(dict(seed=seed, coef=coef, reset=reset, signal=signal,
                         valid_windows=int(ok.sum()), all_valid=bool(ok.all()),
                         MG=float(g.loc[ok, 'MG'].median()) if ok.all() else np.nan,
                         MG_seconds=float(g.MG_seconds.sum()),
                         total_seconds=float(g.total_seconds.sum()),
                         ident_min=g.ident.min(), ident_max=g.ident.max()))
    return rows

def main():
    jobs = []
    for seed in SEEDS:
        for coef in COEFS:
            d = pd.read_csv(H / label(seed, coef) / 'test.csv')
            jobs += [(seed, coef, int(r)) for r in d.loc[d.eligible, 'reset']]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(analyze, jobs), 1):
            rows.extend(result)
            if i % 10 == 0 or i == len(jobs):
                print(f'MG {i}/{len(jobs)} eligible trajectories', flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(H / 'action_mg_raw.csv', index=False)
    base = raw[raw.coef == 0][['seed', 'reset', 'signal', 'MG']].rename(columns={'MG':'MG_control'})
    frame = raw.merge(base, on=['seed', 'reset', 'signal'], how='inner', validate='many_to_one')
    frame['MG_ratio'] = frame.MG / frame.MG_control
    frame.to_csv(H / 'action_mg.csv', index=False)
    seeds = frame.groupby(['seed', 'coef', 'signal']).agg(n=('MG_ratio','count'), ratio=('MG_ratio','median')).reset_index()
    seeds.to_csv(H / 'action_mg_by_seed.csv', index=False)
    summary = seeds.groupby(['coef', 'signal']).agg(seeds=('ratio','count'), ratio=('ratio','median'),
                          min_ratio=('ratio','min'), max_ratio=('ratio','max'),
                          seeds_decrease=('ratio',lambda x: int((x.dropna()<1).sum()))).reset_index()
    summary.to_csv(H / 'action_mg_summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()

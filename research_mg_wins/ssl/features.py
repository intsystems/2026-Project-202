"""Window statistics (MG and every scalar competitor) on the saved per-step logs.

For each run file runs_*/<name>_s<seed>.npz and each log in LOGS, windows of W=1000 steps
ending at 1000, 1500, ..., 4000 (stride 500) are scored by every statistic in STATS.
Output: one CSV per run in <featdir>/ (incremental, skips existing), with wall-clock per stat.

usage: python features.py <rundir> <featdir>
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "research_trajectory_reference"))

W, S = 1000, 500
LOGS = ("loss", "param_norm", "grad_norm", "probe_z_norm", "probe_h_norm")


def _stats():
    from actdim.estimator.config import EstimatorConfig
    from actdim.estimator.mle import estimate
    import baselines as BL
    from cifar_events import simple
    cfg = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
    cfg4 = EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")
    st = {"MG": lambda x: estimate(np.asarray(x, float), cfg).MG,
          "MG_t4k50": lambda x: estimate(np.asarray(x, float), cfg4).MG,
          "spectral_entropy": BL.spectral_entropy,
          "self_repeat": BL.self_repeat,
          "roughness": BL.roughness,
          "perm_entropy": partial(BL.perm_entropy, lag=1),
          "recurrence_rate": partial(BL.recurrence_rate, tau=1),
          "corr_dim": partial(BL.corr_dim, tau=1),
          "twonn": partial(BL.twonn_fit, tau=1),
          "linear_pr": partial(BL.linear_pr, tau=1),
          "simple": simple,
          # level statistics of the same window (practitioner rules)
          "mean": lambda x: float(np.mean(x)),
          "rel_std": lambda x: float(np.std(x) / (abs(np.mean(x)) + 1e-12)),
          "rel_change": lambda x: float((np.mean(x[-100:]) - np.mean(x[:100])) / (abs(np.mean(x[:100])) + 1e-12))}
    return st


STATS = None


def _init():
    global STATS
    STATS = _stats()


def _one(args):
    path, featdir = args
    out = Path(featdir) / (Path(path).stem + ".csv")
    if out.exists():
        return path, 0.0
    t0 = time.time()
    d = np.load(path)
    rows = []
    T = len(d["loss"])
    for log in LOGS:
        x = d[log].astype(float)
        for end in range(W, T + 1, S):
            seg = x[end - W:end]
            row = {"run": Path(path).stem, "log": log, "end": end}
            if not np.all(np.isfinite(seg)) or seg.std() == 0:
                rows.append(row)
                continue
            for k, f in STATS.items():
                t = time.perf_counter()
                try:
                    v = f(seg)
                except Exception:
                    v = float("nan")
                dt = time.perf_counter() - t
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        row[kk] = vv
                    row["t_" + k] = dt
                else:
                    row[k] = v
                    row["t_" + k] = dt
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return path, time.time() - t0


def main():
    rundir, featdir = Path(sys.argv[1]), Path(sys.argv[2])
    featdir.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in rundir.glob("*.npz"))
    with Pool(int(os.environ.get("NWORK", 3)), initializer=_init) as p:
        for path, dt in p.imap_unordered(_one, [(f, str(featdir)) for f in files]):
            print(f"{time.strftime('%H:%M:%S')} {Path(path).stem} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()

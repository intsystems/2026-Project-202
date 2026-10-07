"""Window statistics of the scalar trunk logs: MG and every scalar competitor, same windows."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "research_trajectory_reference"))
sys.path.insert(0, str(HERE.parents[1] / "code"))
import baselines as BL  # noqa: E402
from cifar_events import simple  # noqa: E402
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

W, S = 1000, 250
CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")      # primary MG
CFG4 = EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")     # secondary variant

STATS = {
    "MG": lambda x: estimate(x, CFG).MG,
    "MG_tau4": lambda x: estimate(x, CFG4).MG,
    "self_repeat": BL.self_repeat,
    "spectral_entropy": BL.spectral_entropy,
    "roughness": BL.roughness,
    "perm_entropy": partial(BL.perm_entropy, lag=1),
    "recurrence_rate": partial(BL.recurrence_rate, tau=1),
    "corr_dim": partial(BL.corr_dim, tau=1),
    "twonn": partial(BL.twonn_fit, tau=1),
    "linear_pr": partial(BL.linear_pr, tau=1),
}
SIMPLE = ("crossings", "lag1", "det_std")
LOGS = ("param_norm", "batch_loss")


def windows(logs, T):
    rows, cost = [], {}
    for name in LOGS:
        x = np.asarray(logs[name], float)
        for a in range(0, T - W + 1, S):
            seg = x[a:a + W]
            r = {"log": name, "start": a, "end": a + W}
            if not np.isfinite(seg).all():
                rows.append(r)
                continue
            for k, f in STATS.items():
                t0 = time.perf_counter()
                try:
                    r[k] = float(f(seg))
                except Exception:  # noqa: BLE001
                    r[k] = np.nan
                cost[k] = cost.get(k, 0.0) + time.perf_counter() - t0
            t0 = time.perf_counter()
            r.update(simple(seg))
            cost["simple3"] = cost.get("simple3", 0.0) + time.perf_counter() - t0
            rows.append(r)
    return pd.DataFrame(rows), cost

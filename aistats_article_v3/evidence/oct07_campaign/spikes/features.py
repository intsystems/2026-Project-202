"""Per-run window statistics for the S4 warning rules (same windows for every statistic).

At every evaluation time t = W, W+S, ... (only data before t is used) each scalar log
(loss, grad_norm, update_norm in log10, param_norm raw) is cut to the trailing window
[t-W, t) and scored by MG and by every scalar competitor; internals (attn_max, attn_ent,
logz) and the scalar logs themselves also enter as levels (mean / extreme of the last S steps).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import baselines as BL  # noqa: E402
from baselines import EstimatorConfig, estimate  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402

CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
CFG2 = EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")
W, S = 500, 50
SCALAR_LOGS = ("loss", "grad_norm", "update_norm", "param_norm")
INTERNAL_LOGS = ("attn_max", "attn_ent", "logz")
LOG10 = ("loss", "grad_norm", "update_norm")


def simple(seg):            # cifar_events.simple (crossings, detrended lag-1, relative det. std)
    t = np.arange(len(seg))
    r = seg - np.polyval(np.polyfit(t, seg, 1), t)
    return {"crossings": trend_crossings(seg), "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            "det_std": float(r.std() / (abs(seg.mean()) + 1e-12)), "var": float(r.std())}


WSTATS = {"MG": lambda x: estimate(x, CFG).MG, "MG_t4k50": lambda x: estimate(x, CFG2).MG,
          "self_repeat": BL.self_repeat, "spectral_entropy": BL.spectral_entropy,
          "roughness": BL.roughness, "perm_entropy": partial(BL.perm_entropy, lag=1),
          "recurrence_rate": partial(BL.recurrence_rate, tau=1), "corr_dim": partial(BL.corr_dim, tau=1),
          "twonn": partial(BL.twonn_fit, tau=1), "linear_pr": partial(BL.linear_pr, tau=1)}


def transform(name, x):
    x = np.asarray(x, float)
    return np.log10(np.maximum(x, 1e-12)) if name in LOG10 else x


def run_features(logs):
    """logs: dict of per-step arrays. Returns (rows, seconds per statistic)."""
    n = int(np.isfinite(logs["loss"]).sum())
    series = {k: transform(k, logs[k][:n]) for k in SCALAR_LOGS + INTERNAL_LOGS}
    cost = {}
    rows = []
    for t in range(W, n + 1, S):
        row = {"t": t}
        for k in SCALAR_LOGS:
            seg = series[k][t - W:t]
            if not np.isfinite(seg).all() or seg.std() == 0:
                continue
            for name, f in WSTATS.items():
                t0 = time.perf_counter()
                try:
                    row[f"{name}|{k}"] = float(f(seg))
                except Exception:
                    row[f"{name}|{k}"] = np.nan
                cost[name] = cost.get(name, 0.0) + time.perf_counter() - t0
            t0 = time.perf_counter()
            for name, v in simple(seg).items():
                row[f"{name}|{k}"] = v
            cost["simple"] = cost.get("simple", 0.0) + time.perf_counter() - t0
        for k in SCALAR_LOGS + INTERNAL_LOGS:
            last = series[k][t - S:t]
            row[f"level|{k}"] = float(np.mean(last))
            row[f"max|{k}"] = float(np.max(last))
            row[f"min|{k}"] = float(np.min(last))
        rows.append(row)
    return rows, cost

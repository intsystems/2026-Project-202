"""E6 -- how much of "MG saturates" is MG, and how much is the delay lag?

The delay window spans ``(max_E - 1) * tau`` samples.  A torus is only unfolded when that
span covers a real fraction of the oscillation period, so any claim that MG fails on a slowly
oscillating log is a claim about the tau it was measured with, until tau has been swept.
E1 cannot answer this: it calibrates at ``f0 = 1/16``, where tau <= 4 already spans more than
a period, and a grid whose winner is chosen in one regime cannot license a claim about
another.

So this file sweeps tau explicitly, in both regimes, on the deterministic torus -- the one
family where an answer of r exists -- and reports the spread of MG across tau beside the
truth.  The number that matters is not any single MG value but how far the estimate moves
across the estimator's own free parameter while the system does not change at all.

    python e6_tau_sensitivity.py         # ~10 min on 12 cores
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import generators as G
from mg import MGConfig, all_estimators

OUT = Path(__file__).resolve().parent / "results" / "e6_tau"
OUT.mkdir(parents=True, exist_ok=True)

R_VALUES = (1, 2, 3, 4, 6, 8)
TAUS = (1, 2, 4, 8, 16, 32, "acorr")
PERIODS = (16, 400)          # the E2 'qp' and 'qp_slow' regimes
N = 24_000


def one(period, r, tau, max_E, seed):
    rng = np.random.default_rng(1009 * seed + 31 * r + period)
    X, meta = G.qp(r, N, rng, f0=1.0 / period)
    y = G.observe(X, rng, "generic")
    cfg = MGConfig(max_E=max_E, tau=tau, k_neighbors=5, theiler="embedding",
                   window=N, stride=N)
    e = all_estimators(y, cfg, seed=seed)
    return dict(period=period, r=r, tau=tau, max_E=max_E, seed=seed,
                tau_used=e.get("tau_used"), theiler_used=e.get("theiler_used"),
                span=(max_E - 1) * (e.get("tau_used") or 1),
                span_periods=(max_E - 1) * (e.get("tau_used") or 1) / period,
                MG=e["MG"], PRdelay=e["PRdelay"], specPR0=e["specPR0"],
                specPR256=e["specPR256"], roughness=e["roughness"],
                degenerate=bool(e["degenerate"]), margin=float(meta["margin"]))


def main():
    t0 = time.time()
    jobs = list(itertools.product(PERIODS, R_VALUES, TAUS, (10, 20), (0, 1)))
    print(f"{len(jobs)} cells", flush=True)
    rows = Parallel(n_jobs=12, verbose=2, batch_size=4)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tau_sensitivity.csv", index=False)

    print(f"\ndone in {time.time()-t0:.0f}s")
    for per in PERIODS:
        s = df[(df.period == per) & (df.max_E == 20)]
        print(f"\n=== oscillation period {per} samples, max_E = 20: MG by tau ===")
        print(s.pivot_table(index="tau", columns="r", values="MG",
                            sort=False).round(2).to_string())
        print("--- delay span, in periods of the slowest mode ---")
        print(s.pivot_table(index="tau", columns="r", values="span_periods",
                            sort=False).round(2).to_string())
        sp = s.groupby("r").agg(lo=("MG", "min"), hi=("MG", "max"))
        print("--- how far MG moves across tau alone, with the system unchanged ---")
        print((sp.hi - sp.lo).round(2).to_string())
        print("--- the linear spectral null on the same signals (native resolution) ---")
        print(s.groupby("r").specPR0.median().round(2).to_string())


if __name__ == "__main__":
    main()

"""E0b -- does the winner of "MG vs the linear spectral null" depend on the observer?

Report section 0.2 claims it does.  That claim was first seen in a scratch probe, which is
not good enough to print, so it is shipped here: the same r-torus seen through two scalar
observers that differ only in how evenly the r modes are weighted.

``equal``    w_j = +/-1 / sqrt(r).  Every mode contributes the same power.
``generic``  w_j ~ N(0, 1/r).  Mode amplitudes differ by an order of magnitude, so the weak
             ones fall below the neighbour scale -- which is what a random projection of a
             network's parameters actually looks like.

Ground truth needs no separate measurement here: for r equal-amplitude sinusoids the state
participation ratio is r by construction, and ``e0_atlas`` measures it at 1.0000, 2.0000,
2.9988, 3.9972, 4.9911, 5.9862, 7.9849 for r = 1..8.

    python e0b_observer_amplitude.py      # ~4 min on 12 cores
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

OUT = Path(__file__).resolve().parent / "results" / "e0b_observer"
OUT.mkdir(parents=True, exist_ok=True)

R_VALUES = (1, 2, 3, 4, 6, 8)
N = 24_000
PERIOD = 20              # samples per cycle of the fastest mode


def observe(X, rng, kind):
    w = (rng.choice([-1.0, 1.0], X.shape[1]) if kind == "equal"
         else rng.standard_normal(X.shape[1])) / np.sqrt(X.shape[1])
    z = X @ w
    y = z + 0.2 * z ** 2
    return (y - y.mean()) / (y.std() + 1e-12)


def one(r, kind, tau, seed):
    rng = np.random.default_rng(4242 * seed + 7 * r)
    X, meta = G.qp(r, N, rng, f0=1.0 / PERIOD)
    y = observe(X, rng, kind)
    hard, pr = G.state_rank(X)
    cfg = MGConfig(max_E=20, tau=tau, k_neighbors=5, theiler="embedding",
                   window=N, stride=N)
    e = all_estimators(y, cfg, seed=seed)
    return dict(r=r, observer=kind, tau=tau, seed=seed, state_PR=pr,
                MG=e["MG"], specPR0=e["specPR0"], specPR256=e["specPR256"],
                PRdelay=e["PRdelay"], roughness=e["roughness"],
                degenerate=bool(e["degenerate"]))


def main():
    t0 = time.time()
    jobs = list(itertools.product(R_VALUES, ("equal", "generic"), (1, 2, 4), (0, 1, 2)))
    print(f"{len(jobs)} cells", flush=True)
    rows = Parallel(n_jobs=12, verbose=2, batch_size=4)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "observer_amplitude.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")
    for kind in ("equal", "generic"):
        s = df[df.observer == kind]
        print(f"\n=== observer = {kind} ===")
        print("MG by tau:")
        print(s.pivot_table(index="tau", columns="r", values="MG").round(2).to_string())
        print("spectral PR (native resolution), and the measured active dimension:")
        print(s.pivot_table(index="r", values=["specPR0", "specPR256", "state_PR"])
              .round(2).to_string())


if __name__ == "__main__":
    main()

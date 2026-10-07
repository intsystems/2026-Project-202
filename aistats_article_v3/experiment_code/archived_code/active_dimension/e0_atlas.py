"""E0 -- the identifiability atlas.  Which generator class, if any, lets MG see r?

Nothing about a network is asked here.  The question is prior to that: for a 1-D scalar
observation of an r-dimensional system, is r recoverable *in principle*, and what does the
window have to look like for it to be?  Three families with the same nominal r (see
``generators.py``), swept over coverage, embedding dimension and observer.

The two things this must decide before any network experiment is worth running:

1.  does MG track r for stochastically excited dynamics?  (Mini-batch gradient noise is
    exactly that, so the answer determines whether experiment E2's stochastic arm can
    possibly succeed.)
2.  for the one family where r *is* identifiable, how much window is needed at each r?

Every table also carries the nulls -- ``roughness``, autocorrelation time, and the linear
delay-matrix participation ratio -- because the exp10-12 audit found the linear PR recovers
k better than MG on identical data, and no run in that suite reported it.

    python e0_atlas.py            # ~20 min on 12 cores
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import generators as G
from mg import MGConfig, all_estimators, ESTIMATOR_NAMES

OUT = Path(__file__).resolve().parent / "results" / "e0_atlas"
OUT.mkdir(parents=True, exist_ok=True)

R_VALUES = (1, 2, 3, 4, 5, 6, 8)
SEEDS = (0, 1, 2)
OBSERVERS = ("generic", "norm")


def one(family, r, N, seed, obs, max_E, extra):
    rng = np.random.default_rng(10_000 * seed + 97 * r + max_E + sum(map(ord, family)))
    X, meta = G.FAMILIES[family](r, N, rng, **extra)
    y = G.observe(X, rng, obs)
    hard, pr = G.state_rank(X)
    cfg = MGConfig(max_E=max_E, window=N, stride=N, theiler="embedding")
    est = all_estimators(y, cfg, seed=seed)
    row = dict(family=family, r=r, N=N, seed=seed, observer=obs, max_E=max_E,
               state_rank=hard, state_PR=pr,
               margin=float(meta.get("margin", np.nan)),
               innov_ratio=float(meta.get("innov_ratio", np.nan)),
               tau_c=float(meta.get("tau_c", np.nan)))
    row.update({k: est.get(k, np.nan) for k in ESTIMATOR_NAMES})
    row["degenerate"] = bool(est["degenerate"])
    return row


def main():
    t0 = time.time()
    jobs = []

    # --- A. coverage sweep on the deterministic torus (the only identifiable family).
    #     Two base periods so that "cycles of the slowest mode" and "number of samples" can
    #     be told apart: (f0=1/100, 200 cycles) and (f0=1/400, 50 cycles) are both N=20000.
    for r, f0, cyc, seed, obs, mE in itertools.product(
            R_VALUES, (1 / 400.0, 1 / 100.0), (50, 200, 800), SEEDS, OBSERVERS, (10, 20)):
        N = int(cyc / f0)
        if N > 100_000:
            continue
        jobs.append(("qp", r, N, seed, obs, mE, dict(f0=f0)))

    # --- B. stochastic families, swept over correlation time and length ---
    for r, tau_c, N, seed, obs, mE in itertools.product(
            R_VALUES, (50.0, 200.0, 1000.0), (20_000, 60_000), SEEDS[:2], ("generic",), (10, 20)):
        jobs.append(("ou", r, N, seed, obs, mE, dict(tau_c=tau_c)))
        jobs.append(("colored", r, N, seed, obs, mE, dict(tau_c=tau_c, order=3)))

    print(f"{len(jobs)} windows", flush=True)
    rows = Parallel(n_jobs=12, verbose=5, batch_size=4)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "atlas_raw.csv", index=False)

    # --- identifiability ratio: MG(E=20)/MG(E=10).  ~1 = a property of the data,
    #     ~2 = a property of the embedding, i.e. no dimension is identifiable.
    key = ["family", "r", "N", "seed", "observer", "tau_c"]
    p = df.pivot_table(index=key, columns="max_E", values="MG", dropna=False).reset_index()
    p["ident_ratio"] = p[20] / p[10]
    p.to_csv(OUT / "identifiability_ratio.csv", index=False)

    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")

    d = df[df.max_E == 20]
    for fam in ("qp", "ou", "colored"):
        s = d[(d.family == fam) & (d.observer == "generic")]
        if not len(s):
            continue
        print(f"\n=== {fam}: MG (max_E=20, generic observer) ===")
        idx = "N" if fam == "qp" else ["tau_c", "N"]
        print(s.pivot_table(index=idx, columns="r", values="MG").round(2).to_string())
        print(f"--- {fam}: roughness null ---")
        print(s.pivot_table(index=idx, columns="r", values="roughness").round(3).to_string())
        print(f"--- {fam}: linear PR of the delay matrix ---")
        print(s.pivot_table(index=idx, columns="r", values="PRdelay").round(2).to_string())
        print(f"--- {fam}: spectral PR (native resolution), the sharpest linear null ---")
        print(s.pivot_table(index=idx, columns="r", values="specPR0").round(2).to_string())

    print("\n=== identifiability ratio MG(E=20)/MG(E=10) ===")
    print(p.pivot_table(index="family", columns="r", values="ident_ratio").round(2).to_string())
    print("\n=== resonance margins of the torus frequency sets ===")
    q = df[df.family == "qp"].groupby("r").margin.first()
    print(q.round(5).to_string())


if __name__ == "__main__":
    main()

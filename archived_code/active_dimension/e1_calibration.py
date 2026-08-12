"""E1 -- choose the estimator's free parameters, once, on data no later experiment uses.

The exp10-12 audit found that the earlier calibration grid mixed estimator parameters with
*system* parameters (``cycles_per_window``, and the learning rate in exp11/12), so selecting
a configuration changed the data-generating process; and the objective was error against the
known k on the very grid that was then reported, so the absolute level of the estimate was
tuned on the answer (MG at k=20 moves 5.3 -> 16.3 across that grid, and the winner sat at
the grid boundary).  Two rules follow, and are enforced here:

1.  the grid contains **only** estimator parameters -- ``max_E``, ``tau``, ``k_neighbors``,
    ``theiler``, ``window``.  One system configuration is simulated per (seed, r) and every
    estimator configuration is scored on those same logs, so the data cannot move.
2.  the calibration split is disjoint in **both** seed and r.  Calibration uses seeds 90-92
    and r in {2, 4, 6}; every later experiment uses seeds 0-5, and reports r in {1, 3, 5, 7,
    8} separately from {2, 4, 6}.  In exp10-12 "held out" meant held-out seeds while
    ``systems.py`` ignored the rng under ``band_mode="matched"``, so the frequency geometry
    -- the thing the estimator responds to -- was bit-identical across the split.

The objective is scored against the **measured** active dimension ``traj_PR``, not against
nominal r.  What is frozen goes to ``results/e1_calibration/frozen_config.json``.

    python e1_calibration.py        # ~15 min on 12 cores
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from dynamics import F_FAST, OBSERVER_FAMILY, Spec, equalise_gains, simulate
from runner import get_system, mae, spearman

OUT = Path(__file__).resolve().parent / "results" / "e1_calibration"
OUT.mkdir(parents=True, exist_ok=True)

CAL_SEEDS = (90, 91, 92)
CAL_R = (2, 4, 6)
T, BURN = 30_000, 4_000
#: one observer per family, so the choice of configuration is not a choice of observer
CAL_OBS = ("w_fro", "c_norm", "g_fro", "c_proj1")

#: ``tau`` must be able to reach the oscillation period, or a null result is a statement
#: about tau rather than about MG: on a period-400 torus MG reads 1.42/2.72/2.70 at r=1/4/6
#: with tau=1 and 1.44/14.7/22.6 with tau = acf/4.  The grid here spans only tau <= 4 because
#: the whole calibration runs at f0 = 1/16, where four is already more than a full period;
#: the slow regime is not calibrated here but measured directly in ``e6_tau_sensitivity.py``,
#: which is the honest place for it -- a grid whose winner is chosen in one regime cannot
#: license a claim about the other.
GRID = [MG.MGConfig(max_E=mE, tau=tau, k_neighbors=kn, theiler=th, window=W, stride=2000)
        for mE, tau, kn, th, W in itertools.product(
            (10, 20), (1, 2, 4), (5, 20), ("embedding", "autocorr"), (4000, 8000))]


def job(seed, r):
    """Simulate once, then score every estimator configuration on the same logs."""
    A, c_star = get_system(seed, 10)
    spec = Spec(seed=seed, k=10, r=r, T=T, burn=BURN, mode="qp", drive_amp=0.8,
                precondition=True, f0=F_FAST)
    mix, cond = equalise_gains(A, spec, c_star)
    logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)
    out = []
    for o in CAL_OBS:
        x = logs[o]
        if x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()
        for cid, cfg in enumerate(GRID):
            rec = MG.summarise(z, cfg)
            out.append(dict(cfg_id=cid, seed=seed, r=r, observer=o,
                            family=OBSERVER_FAMILY[o], traj_PR=info["traj_PR"],
                            upd_PR=info["upd_PR"], drive_cond=cond, **cfg.as_dict(), **rec))
    return out


def main():
    t0 = time.time()
    jobs = [(s, r) for s in CAL_SEEDS for r in CAL_R]
    print(f"{len(jobs)} simulations x {len(GRID)} estimator configurations "
          f"x {len(CAL_OBS)} observers", flush=True)
    res = Parallel(n_jobs=9, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "calibration_raw.csv", index=False)

    recs = []
    for (cid, obs), g in df.groupby(["cfg_id", "observer"]):
        # Spearman on all 9 (seed, r) points, not on the 3 r-medians: over 3 points rho can
        # only take {-1, -0.5, 0.5, 1} and a 2*(1-rho) penalty then swamps the MAE term.
        m = g.groupby("r").agg(MG=("MG", "median"), truth=("traj_PR", "median"),
                               deg=("frac_degenerate", "mean")).reset_index()
        rec = dict(cfg_id=cid, observer=obs,
                   rho=spearman(g.MG, g.traj_PR), mae_raw=mae(m.MG, m.truth),
                   rho_rough=spearman(g.roughness, g.traj_PR),
                   rho_prd=spearman(g.PRdelay, g.traj_PR),
                   mae_prd=mae(m.MG * 0 + g.groupby("r").PRdelay.median().values, m.truth),
                   degenerate=float(m.deg.mean()),
                   sd_across_seeds=float(g.groupby("r").MG.std().mean()),
                   **GRID[cid].as_dict())
        for nb in MG.SPEC_NBINS:
            col = f"specPR{nb}"
            rec[f"rho_spec{nb}"] = spearman(g[col], g.traj_PR)
            rec[f"mae_spec{nb}"] = mae(g.groupby("r")[col].median().values, m.truth)
        recs.append(rec)
    sc = pd.DataFrame(recs)
    sc.to_csv(OUT / "calibration_scores.csv", index=False)

    per_cfg = (sc[sc.degenerate < 0.05].groupby("cfg_id")
               .agg(mae=("mae_raw", "median"), rho=("rho", "median"),
                    sd=("sd_across_seeds", "median"), n=("observer", "count")).reset_index())
    # a configuration that is degenerate for most observers must not win on the survivor
    per_cfg = per_cfg[per_cfg.n == len(CAL_OBS)]
    per_cfg["score"] = per_cfg["mae"] + 2.0 * (1 - per_cfg["rho"]) + 0.5 * per_cfg["sd"]
    per_cfg = per_cfg.sort_values("score").reset_index(drop=True)
    per_cfg = per_cfg.merge(pd.DataFrame([{**GRID[i].as_dict(), "cfg_id": i}
                                          for i in range(len(GRID))]), on="cfg_id")
    per_cfg.to_csv(OUT / "config_ranking.csv", index=False)
    best = int(per_cfg.iloc[0].cfg_id)
    cfg = GRID[best]

    b = df[df.cfg_id == best]
    cal_maps = {}
    for obs, g in b.groupby("observer"):
        v = g[np.isfinite(g.MG)]
        if len(v) < 3:
            continue
        iso = MG.Calibration("isotonic").fit(v.MG.values, v.traj_PR.values)
        grid = np.linspace(float(v.MG.min()), float(v.MG.max()), 41)
        cal_maps[obs] = dict(x=grid.tolist(), y=np.asarray(iso.predict(grid)).tolist())

    (OUT / "frozen_config.json").write_text(json.dumps(
        dict(config=cfg.as_dict(), cfg_id=best, cal_seeds=list(CAL_SEEDS),
             cal_r=list(CAL_R), score=float(per_cfg.iloc[0].score), isotonic=cal_maps),
        indent=1))

    print(f"\ndone in {time.time()-t0:.0f}s")
    print("\n=== top 10 estimator configurations (calibration seeds only) ===")
    print(per_cfg.head(10)[["cfg_id", "max_E", "tau", "k_neighbors", "theiler", "window",
                            "mae", "rho", "sd", "score"]].round(3).to_string(index=False))
    print(f"\nFROZEN -> {cfg}")
    print("\n=== per-observer, at the frozen configuration ===")
    print(sc[sc.cfg_id == best].sort_values("mae_raw")
          [["observer", "rho", "mae_raw", "rho_rough", "rho_prd", "mae_prd",
            "degenerate", "sd_across_seeds"]].round(3).to_string(index=False))


def load_frozen():
    p = OUT / "frozen_config.json"
    if not p.exists():
        raise FileNotFoundError("run e1_calibration.py first")
    d = json.loads(p.read_text())
    c = d["config"]
    return MG.MGConfig(**c), d


if __name__ == "__main__":
    main()

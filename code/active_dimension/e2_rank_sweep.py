"""E2 -- the main test.  Available k is fixed at 10; the *active* rank r is varied 1..8.

Seven arms, all on the same real data (digits), the same frozen nonlinear backbone and the
same 10 available adapter directions.  Only the way the optimiser is excited changes.  The
measured trajectory participation ratio for each, at r = 1/2/4/6/8, is quoted so that what
"active dimension" means in that arm is a measurement and not a label:

``qp``          r data groups' loss weights modulated by r incommensurate sinusoids at
                f0 = 1/16.  Deterministic, recurrent, an r-torus.  PR = 1.00/2.00/4.00/6.00/
                8.00.  The only arm in which a delay embedding of dimension r exists to find.
``qp_slow``     the same at f0 = 1/400, i.e. the timescale a training log actually has.
                PR = 1.00/2.00/3.98/5.94/7.83.
``noise``       rank-r Gaussian noise added to the update -- the brief's experiment 2.
                Stationary stochastic fluctuation.  PR = 1.00/2.00/4.00/5.96/7.92.
``batch_proj``  real mini-batch gradient noise projected onto r directions: the brief's
                experiment 2 with the covariance the *data* produces.  The data's own
                variance profile across those directions is not flat, so
                PR = 1.00/1.75/3.34/4.42/5.47 -- below r, and measured rather than assumed.
``batch``       ordinary mini-batch SGD near the solution -- the brief's experiment 1.
                **r has no effect here**: PR = 5.94/5.93/6.07/5.95/6.09.  The noise rank is
                whatever the data says it is, which is itself the answer to "can r be set by
                choosing a batch size".  Excluded from any pooled rho or MAE over r.
``mixed``       ``qp`` and ``noise`` together, with the noise confined to the torus's own r
                directions so the arm is an r-torus in noise and not an r-torus plus an
                independent rank-r diffusion.  PR = 1.00/2.00/4.00/5.99/7.96.
``gd``          full-batch descent from a start displaced inside the r-dimensional subspace,
                slow enough that the transient fills the window.  PR = 1.04/1.04/1.02/1.02/
                1.04 -- a transient is a 1-D curve for every r, which is why exp15 v1 read
                1.33 at every k.

Plus two controls that the audits of the earlier suite showed are indispensable:

``eta_zero``       the same drive with the learning rate multiplied by zero.  Re-running
                   exp14 that way reproduced its headline (MAE 2.06 -> 1.87, rho unchanged),
                   because its observers read the drive through the residual instead of the
                   optimiser state.  Any observer whose MG survives this is disqualified.
``precondition=0`` the same rank-r noise without the ``H^{-1}`` preconditioner.  The
                   stationary covariance is then supported on the Krylov space of ``I-eta H``
                   over the noise subspace, which is generically all of R^k -- so the
                   *active* dimension is k even though the *injected* rank is r.  This is the
                   available-vs-active distinction made by measurement.

Every row carries the measured ``traj_PR`` (active), ``func_PR`` (functional) and
``available`` (=k), the nulls, and the identifiability ratio E(2 max_E)/E(max_E).

    python e2_rank_sweep.py         # ~90 min on 12 cores
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dynamics import F_FAST, F_SLOW, Spec
from e1_calibration import load_frozen
from runner import flatten, run_one

OUT = Path(__file__).resolve().parent / "results" / "e2_rank_sweep"
OUT.mkdir(parents=True, exist_ok=True)

R_VALUES = (1, 2, 3, 4, 5, 6, 8)
#: every observer family is represented; the four dropped from OBSERVERS (w_fro_sq,
#: c_proj2, c_proj3, fn_proj2) are near-duplicates of one that is kept, and each costs a
#: full sliding-window pass of the estimator.
SWEEP_OBS = ("loss_step", "loss_full", "loss_probe", "w_fro", "c_norm", "fn_fro",
             "g_fro", "g_proj", "c_proj1", "fn_proj1", "margin", "acc_probe")
SEEDS = (0, 1, 2, 3)
SUB_SEEDS = (0, 1)
T, BURN = 30_000, 4_000


def make_spec(mode, r, seed, **kw):
    """One place where each arm's dynamical parameters live, so they cannot drift apart.

    Amplitudes are chosen so that every arm's trajectory excursion is ~0.03-0.06 in the
    k-dimensional coordinate, i.e. all arms sit at a comparable distance from the operating
    point and none is compared at an unfair signal-to-noise ratio.
    """
    base = dict(seed=seed, k=10, r=r, T=T, burn=BURN, mode=mode, precondition=True,
                eta=0.15, f0=F_FAST)
    if mode == "qp":
        base.update(drive_amp=0.8, noise_amp=0.0)
    elif mode == "qp_slow":
        base.update(mode="qp", drive_amp=0.8, noise_amp=0.0, f0=F_SLOW)
    elif mode == "noise":
        base.update(drive_amp=0.0, noise_amp=0.08)
    elif mode == "batch":
        base.update(drive_amp=0.0, noise_amp=0.0, batch=64)
    elif mode == "batch_proj":
        base.update(drive_amp=0.0, noise_amp=3.0, batch=64)
    elif mode == "mixed":
        base.update(drive_amp=0.8, noise_amp=0.02)
    elif mode == "gd":
        # a transient must be slow enough to fill the window, and must not be preconditioned
        # away: with H's eigenvalues 0.017-0.072, eta=0.006 gives time constants 2.3k-10k
        base.update(drive_amp=0.0, noise_amp=0.0, eta=0.006, precondition=False,
                    burn=0, gd_disp=1.0)
    else:
        raise ValueError(mode)
    base.update(kw)
    return Spec(**base)


def job(mode, r, seed, cfg, tag, **kw):
    import dataclasses
    # keep the number of windows ~10 whatever window length the calibration froze, so the
    # cost of the sweep does not depend on that choice
    n = T - (0 if mode == "gd" else BURN)
    cfg = dataclasses.replace(cfg, stride=max(500, (n - cfg.window) // 6))
    spec = make_spec(mode, r, seed, **kw)
    # The identifiability ratio needs a second embedding at max_E = 40.  A KD-tree query for
    # 321 neighbours in 40 dimensions costs far more than one in 20 -- measured, it would be
    # three quarters of the whole sweep -- so it is computed on one seed and two r values per
    # arm, which is all that is needed to say whether a dimension is identifiable in that arm.
    rows, info, _ = run_one(spec, cfg, second_E=(seed == 0 and r in (2, 6)),
                            observers=SWEEP_OBS)
    out = flatten(spec, rows, info)
    for o in out:
        o["arm"] = tag
    return out


def main():
    cfg, meta = load_frozen()
    print(f"frozen estimator config: {cfg}", flush=True)
    t0 = time.time()

    jobs = []
    for m in ("qp", "qp_slow", "noise", "batch", "batch_proj", "mixed", "gd"):
        for r in R_VALUES:
            for s in SEEDS:
                jobs.append((m, r, s, cfg, m, {}))
    for m in ("qp", "noise"):                       # the eta = 0 control
        for r in R_VALUES:
            for s in SUB_SEEDS:
                jobs.append((m, r, s, cfg, m + "_eta0", dict(eta_zero=True)))
    for m in ("qp", "noise"):                       # the Krylov / available-vs-active demo
        for r in R_VALUES:
            for s in SUB_SEEDS:
                jobs.append((m, r, s, cfg, m + "_nopre", dict(precondition=False)))

    print(f"{len(jobs)} runs", flush=True)
    res = Parallel(n_jobs=12, verbose=5, batch_size=1)(
        delayed(job)(m, r, s, c, tg, **kw) for m, r, s, c, tg, kw in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "sweep_raw.csv", index=False)
    print(f"\nsimulated + measured in {time.time()-t0:.0f}s -> {OUT/'sweep_raw.csv'}")

    print("\n=== ground truth actually achieved (median over seeds) ===")
    gt = df.drop_duplicates(["arm", "r", "seed"]).pivot_table(
        index="arm", columns="r", values="traj_PR", aggfunc="median")
    print(gt.round(2).to_string())
    gt.to_csv(OUT / "ground_truth_PR.csv")


if __name__ == "__main__":
    main()

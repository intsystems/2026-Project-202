"""S3: WHEN TO DECAY THE LEARNING RATE from a stationarity diagnostic. Main runner.

PROTOCOL (written 7 Oct 2026 ~02:15 UTC+3, after the pilot on seed 99 (pilot.py) and BEFORE any
calibration or test run; the pilot looked only at losses/accuracies, never at MG or competitors).

Task. 3-conv CIFAR CNN (research_trajectory_reference/cifar_events.model, 14.7k params) on the
cached 10k CIFAR subset: 1 000 images held out as validation (only plateau_val_loss reads it),
the rest is the training pool (optionally subsampled to n and with symmetric label noise),
2 000 clean test images. SGD momentum 0.9, weight decay 5e-4, constant LR lr0, batch with
replacement, fixed budget of T steps. Action: ONE decay of the LR by x10 at a time chosen by a
rule (or never). Score: final clean test accuracy at T (primary) and test loss (secondary).

Measurement (lrrun.py). Per unit (condition, seed): the constant-LR trunk is trained for T
steps; checkpoints at t_j = j*T/16, j = 2..15; from each a branch with lr0/10 is trained to T
on the same batch sequence. A rule whose alarm is at step a decays at the first t_j >= a
(causal; alarm after t_15 or none = constant LR). So every rule is scored on the same measured
outcome curve; the comparison is paired. Cosine decay over T (same batches) is a further
competitor; "oracle" = best decay time in hindsight per unit (upper bound, not a rule).

Conditions (name: lr0, batch, n train, label noise, T):
  A  0.01  32 9000 0.0 6000      B  0.04 32 9000 0.0 6000      C  0.02 32 4500 0.0 6000
  D  0.02  32 9000 0.2 6000      E  0.02 64 9000 0.0 4000      F  0.06 16 9000 0.0 8000
  new test-only conditions:
  G  0.005 32 9000 0.0 6000      H  0.03 32 6000 0.1 8000      I  0.015 64 9000 0.0 4000
  J  0.08  32 9000 0.0 5000
Calibration units: A-F x seeds {0, 1} (12 units).
Test units: A-F x seeds {10, 11} (fresh seeds, seen conditions) and G-J x seeds {10, 11, 12}
(fresh seeds, unseen conditions) = 24 units. Optional extension if time allows (decided
before seeing any test result): A-J x seed 12/13 more test units.

Rules (rules.py), each calibrated on the 12 calibration units by maximising mean final test
accuracy over its grid; the chosen parameters are frozen and applied to test units.
Every rule has the same warm-up grid tmin in {2,4,6,8}/16 * T (earliest allowed decay).
  fixed_step            decay at f*T, f in {2..15}/16 or never
  cosine                (separate run, no parameters)
  plateau_train_loss    torch ReduceLROnPlateau on mean mini-batch loss per 100 steps,
                        patience {3,5,10,20} evals, rel. threshold {1e-3,1e-2,3e-2,1e-1}
  plateau_val_loss      same on held-out validation loss every 100 steps (uses extra data)
  pflug_chee_toulis     running sum of <g_t, g_{t-1}> (stochastic gradients) < 0, summed
                        from 0 (original) or from tmin
  sasa                  Lang et al. 2019 fluctuation-dissipation statistic
                        <x_t,g_t> - lr/2 (1+beta) |d_{t+1}|^2 (PyTorch momentum form), tested every
                        100 steps on the last half of the samples, batch-means s.e.; original
                        "0 in CI" (z in {1.28, 1.96}) and SASA+ equivalence test (delta in
                        {.02,.05,.1,.2,.35,.5,.75} of the mean dissipation term)
  pesme_distance        Pesme et al. 2020: slope of log|x_t - x_0|^2 vs log t between t/q and t,
                        q in {1.5, 2}, decay when slope < {0.25,...,1.25}
  scalar window stats   on the trunk parameter-norm log and mini-batch-loss log, windows
                        W=1000, stride 250 (stats.py): MG (E=20, tau=1, k=20, Theiler=embedding;
                        THE MG of this study), MG_tau4 (secondary variant), self_repeat,
                        spectral_entropy, roughness, perm_entropy, recurrence_rate, corr_dim,
                        twonn, linear_pr, crossings, lag1, det_std. Same rule grid for all:
                        level threshold (sign +-1, threshold = calibration quantile 0.1..0.9),
                        relative change D_k = (med last M - med previous 4)/|med previous 4|
                        with sign +-1, M in {2,4}, delta in {.02,.05,.1,.2,.4}, and
                        stabilisation |D_k| < eps in {.01,.02,.05,.1}; the log is also chosen on
                        calibration.

Metrics on test units: mean test accuracy, mean regret vs oracle, mean test loss, paired
bootstrap CI of MG minus each rule, wins/ties/losses per unit; by condition. Secondary
analysis: the same with tmin fixed at 2/16 T ("pure diagnostic", no time prior).

Predictions (hypothesis of the campaign): MG rule >= fixed_step and >= every scalar competitor
and >= pflug/sasa/pesme/plateau_train; plateau_val_loss and cosine may be better (they use
held-out data / a different schedule family). My own prior: fixed_step is hard to beat in a
known-budget setting; Pflug will rarely fire under momentum 0.9 (pilot: <g_t,g_{t-1}> stays
positive on average at every LR).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

CONDS = {
    "A": dict(lr=0.01, bs=32, n=9000, noise=0.0, T=6000),
    "B": dict(lr=0.04, bs=32, n=9000, noise=0.0, T=6000),
    "C": dict(lr=0.02, bs=32, n=4500, noise=0.0, T=6000),
    "D": dict(lr=0.02, bs=32, n=9000, noise=0.2, T=6000),
    "E": dict(lr=0.02, bs=64, n=9000, noise=0.0, T=4000),
    "F": dict(lr=0.06, bs=16, n=9000, noise=0.0, T=8000),
    "G": dict(lr=0.005, bs=32, n=9000, noise=0.0, T=6000),
    "H": dict(lr=0.03, bs=32, n=6000, noise=0.1, T=8000),
    "I": dict(lr=0.015, bs=64, n=9000, noise=0.0, T=4000),
    "J": dict(lr=0.08, bs=32, n=9000, noise=0.0, T=5000),
}
CAL = [(k, s) for s in (0, 1) for k in "ABCDEF"]
TEST = [(k, s) for s in (10, 11) for k in "ABCDEF"] + [(k, s) for s in (10, 11, 12) for k in "GHIJ"]
EXTRA = [(k, 12) for k in "ABCDEF"] + [(k, 13) for k in "ABCDEFGHIJ"]


def job(args):
    name, seed, split = args
    sys.path.insert(0, str(HERE))
    tag = f"{name}_s{seed}"
    if (OUT / f"win_{tag}.csv").exists():
        return tag, "skip"
    import numpy as np
    import lrrun
    import stats
    c = dict(CONDS[name], name=name, cid=ord(name), split=split)
    t0 = time.perf_counter()
    r = lrrun.run_unit(c, seed, OUT)
    lg = dict(np.load(OUT / f"logs_{tag}.npz"))
    t1 = time.perf_counter()
    win, cost = stats.windows(lg, c["T"])
    win.to_csv(OUT / f"win_{tag}.csv", index=False)
    json.dump({"stat_cost_s": cost, "t_stats": time.perf_counter() - t1, "t_total": time.perf_counter() - t0,
               "t_trunk": r["t_trunk"], "t_branches": r["t_branches"], "t_cosine": r["t_cosine"]},
              open(OUT / f"cost_{tag}.json", "w"), indent=1)
    msg = (f"{tag} {split} none {r['none']['test_acc']:.3f} cos {r['cosine']['test_acc']:.3f} "
           f"best-branch {max(v['test_acc'] for v in r['branches'].values()):.3f} "
           f"total {time.perf_counter() - t0:.0f}s")
    print(msg, flush=True)
    return tag, msg


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = []
    if which in ("all", "cal"):
        jobs += [(k, s, "cal") for k, s in CAL]
    if which in ("all", "test"):
        jobs += [(k, s, "test") for k, s in TEST]
    if which == "extra":
        jobs += [(k, s, "test") for k, s in EXTRA]
    # longest jobs first within each split keeps the 3 workers busy
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 3, maxtasksperchild=4) as p:
        for tag, msg in p.imap_unordered(job, jobs):
            pass
    print("ALL DONE", flush=True)

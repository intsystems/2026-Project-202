"""S1 -- loss of plasticity in continual learning: can MG of a scalar training log trigger
resets better than every alternative?

PROTOCOL (written 7 Oct 2026 ~01:58 UTC+3, after the pilot on seed 99 that looked only at
online accuracy / internal probes, before any MG or competitor value was computed).

Setting. Online permuted MNIST (Dohare et al. 2024; Lyle et al. 2023) on a fixed 10 000-image
subset, 2x2 pooled (196 inputs), MLP 196-w-w-10 ReLU, batch 32, 400 SGD/Adam steps per task,
T = 40 tasks. Online accuracy of a task = mean accuracy on its mini-batches BEFORE each update
(the network's own learning speed = plasticity). Pilot: Adam 3e-3 w256 falls 0.89 -> 0.83,
Adam 3e-3 w64 0.87 -> 0.78, SGD 0.3 w256 0.88 -> 0.66 within 60 tasks; SGD 0.1 w256 barely
declines; a fresh network (reset) recovers ~0.89 but pays on its first task (SGD 0.1: 0.84 vs
0.90 warm). So the best reset schedule differs between conditions (never ... every task).

Two stream families (both pre-registered, reported separately):
  F1 homogeneous: every task = new random pixel permutation.
  F2 heterogeneous difficulty: as F1, plus a fixed Gaussian input corruption of the task's
     data with sigma_k drawn iid from {0, 0.5, 1.0} (fresh-net accuracy 0.89 / 0.87 / 0.82),
     i.e. logs jump in level between tasks for reasons unrelated to plasticity.
Conditions (optimizer, lr, width):
  calibration and test-seen: A3W256 adam 3e-3 w256; A3W64 adam 3e-3 w64;
                             S1W256 sgd 0.1 w256;   S3W256 sgd 0.3 w256
  test-unseen (never used in calibration): A1W128 adam 1e-3 w128; S2W128 sgd 0.2 w128
Seeds: calibration 0-3 (4 per condition), test 10-14 (5 per condition), each family.
This script trains the base streams (no intervention) and computes every monitor value at
every task boundary; eval_s1.py implements the decision protocol below on these files.

Intervention: full reset (re-initialise all weights and the optimiser state, i.e. train from
scratch; Ash & Adams 2020 baseline, Nikishin et al. 2022 resets), applied at a task boundary.
After a full reset the future is a fresh network on fresh iid tasks, so a policy with any
number of resets is evaluated EXACTLY in distribution by renewal: stream i of a condition uses
base run i of the pool until the first alarm, then base run i+1 from its task 0, etc.
(cyclic). Monitors restart after a reset (they only see post-reset logs).

Monitors, value v_j after task j (local index since the last reset):
  scalar logs L in {pnorm (L2 norm of all parameters per step), loss (mini-batch loss),
  gnorm (gradient L2 norm)}, two window types:
    span   = last 1000 steps (crosses 2 task boundaries; available for j >= 2);
    within = the 400 steps of task j only (no task boundary inside).
  statistics on the window: MG (E=20, tau=1, k=20, Theiler=embedding; PRIMARY);
    MG_t4 (tau=4, k=50; secondary, reported but not used for the verdict);
    competitors (baselines.py, tau=1 versions): spectral_entropy, self_repeat, roughness,
    perm_entropy(lag 1), recurrence_rate, corr_dim, twonn, linear_pr; cifar_events.simple:
    crossings, lag1, det_std.
  level monitors from the same scalar logs (practitioner rules): task mean online accuracy
    (acc_mean), task mean loss (loss_mean), task mean grad norm (gnorm_mean), parameter norm
    at task end (pnorm_end; = weight-norm growth monitor of Dohare / Lyle).
  domain monitors needing internals (probe batch of 512 training images of the task):
    dormant-unit fraction with ReDo scores (tau 0, 0.025, 0.1; Sokar et al. 2023),
    srank (Kumar et al. 2021, delta 0.01) and entropy effective rank of last hidden layer.
Rules (same grid for every monitor, sign chosen by calibration):
  REL: ref = median(v_{j0..j0+B-1}) (j0 = first available j after reset), cur = median of the
       last M values, (window [j-M+1, j] must start at >= j0+B); D = s*(cur-ref)/|ref|; reset
       when D < -delta.  s in {+1,-1}, B in {2,4}, M in {1,2,4},
       delta in {.005,.01,.02,.03,.05,.075,.1,.15,.2,.3,.5,1}.
  ABS: cur = median of last M values (from j0 on); reset when s*(cur-theta) < 0;
       theta = 5%,10%,...,95% quantiles of that monitor over all calibration runs of the
       family; s in {+1,-1}; M in {1,2,4}.
  Fixed interval: reset after every I tasks, I in 1..39, or never (I = 40).
Selection: for each monitor (stat x log x window) the rule maximising mean calibration
utility (conditions weighted equally; ties -> fewer resets); for each statistic the best
(log, window) on calibration. Frozen, then applied to the test pools.
Utility of a stream = mean online accuracy over its 40 tasks (Dohare's average online
accuracy), number of resets reported alongside. Reference points: never, reset every task,
calibrated fixed interval, per-condition best fixed interval in hindsight on test (oracle).
Test metric: mean test utility (test-seen and test-unseen separately and pooled), paired
differences MG - X over test streams with bootstrap 95% CI, per-condition wins.
Secondary (a) detection: on test base runs, AUC of v_j for "reset now is beneficial":
  mean online acc of tasks j+1..j+3 of the run < mean fresh-network acc of tasks 0..2 of the
  same condition (test pool), sign chosen on calibration runs.
Secondary (b) early prediction (added 02:06, before any monitor value was looked at): per
  run, early score = mean of v_j over tasks j = 2..4 (first three decision points with span
  windows); target = plasticity loss of the run = mean online acc of tasks 1..5 minus mean of
  tasks 35..39 (no intervention). Spearman rho over test runs (30 per family), per monitor;
  for each statistic the (log, window) with the largest |rho| on calibration runs, sign kept.
Predictions: plasticity loss is gradual simplification (dormant units, srank collapse), so
MG of pnorm should fall with it; the accuracy rule is expected to be strong in F1 (it sees the
target directly) and weaker in F2 (difficulty confound); level-sensitive scalar statistics
(self_repeat, roughness, det_std, recurrence_rate) should suffer from boundary level jumps in
span windows. Claim to test: MG-triggered resets give higher test utility than never, the
calibrated fixed interval, every scalar competitor and every domain monitor.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "research_trajectory_reference"))
import pl_common as P  # noqa: E402
import baselines as BL  # noqa: E402
from cifar_events import simple  # noqa: E402
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
CFG4 = EstimatorConfig(max_E=20, tau=4, k_neighbors=50, theiler="embedding")
T, STEPS, BATCH, WSPAN = 40, 400, 32, 1000
CONDS = {"A3W256": dict(opt="adam", lr=3e-3, width=256), "A3W64": dict(opt="adam", lr=3e-3, width=64),
         "S1W256": dict(opt="sgd", lr=0.1, width=256), "S3W256": dict(opt="sgd", lr=0.3, width=256),
         "A1W128": dict(opt="adam", lr=1e-3, width=128), "S2W128": dict(opt="sgd", lr=0.2, width=128)}
CAL_CONDS = ("A3W256", "A3W64", "S1W256", "S3W256")
UNSEEN = ("A1W128", "S2W128")
CAL_SEEDS, TEST_SEEDS = (0, 1, 2, 3), (10, 11, 12, 13, 14)
FAMILIES = {"F1": None, "F2": [0.0, 0.5, 1.0]}
STATS = {"MG": lambda s: estimate(s, CFG).MG, "MG_t4": lambda s: estimate(s, CFG4).MG,
         "spectral_entropy": BL.spectral_entropy, "self_repeat": BL.self_repeat,
         "roughness": BL.roughness, "perm_entropy": partial(BL.perm_entropy, lag=1),
         "recurrence_rate": partial(BL.recurrence_rate, tau=1), "corr_dim": partial(BL.corr_dim, tau=1),
         "twonn": partial(BL.twonn_fit, tau=1), "linear_pr": partial(BL.linear_pr, tau=1)}
LOGS = ("pnorm", "loss", "gnorm")
OUT = HERE / "runs"


def jobs():
    js = []
    for split, conds, seeds in (("cal", CAL_CONDS, CAL_SEEDS), ("test", CAL_CONDS + UNSEEN, TEST_SEEDS)):
        for fam in FAMILIES:
            for c in conds:
                for s in seeds:
                    js.append((split, fam, c, s))
    return js


def monitor_rows(logs, tasks):
    rows, cost = [], {}
    for j in range(T):
        row = {"j": j}
        rec = tasks[j]
        a, b = j * STEPS, (j + 1) * STEPS
        row.update({"acc_mean": rec["online_acc"], "loss_mean": float(logs["loss"][a:b].mean()),
                    "gnorm_mean": float(logs["gnorm"][a:b].mean()), "pnorm_end": float(logs["pnorm"][b - 1])})
        for k in ("dormant_0.0", "dormant_0.025", "dormant_0.1", "srank", "erank"):
            row[k] = rec[k]
        for L in LOGS:
            x = logs[L].astype(float)
            wins = {"within": x[a:b]}
            if b >= WSPAN:
                wins["span"] = x[b - WSPAN:b]
            for wn, seg in wins.items():
                for name, f in STATS.items():
                    t0 = time.perf_counter()
                    try:
                        v = float(f(seg))
                    except Exception:
                        v = np.nan
                    cost[(name, wn)] = cost.get((name, wn), 0.0) + time.perf_counter() - t0
                    row[f"{name}|{L}|{wn}"] = v
                t0 = time.perf_counter()
                try:
                    sm = simple(seg)
                except Exception:
                    sm = {"crossings": np.nan, "lag1": np.nan, "det_std": np.nan}
                cost[("simple", wn)] = cost.get(("simple", wn), 0.0) + time.perf_counter() - t0
                for k, v in sm.items():
                    row[f"{k}|{L}|{wn}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows), cost


def do_job(job):
    split, fam, c, s = job
    tag = f"{fam}_{c}_s{s}"
    if (OUT / f"mon_{tag}.csv").exists():
        return
    cfg = dict(depth=2, wd=0.0, steps=STEPS, batch=BATCH, noise_levels=FAMILIES[fam], **CONDS[c])
    t0 = time.perf_counter()
    r = P.run_stream(cfg, s, T)
    t_train = time.perf_counter() - t0
    np.savez_compressed(OUT / f"logs_{tag}.npz", **r["logs"])
    json.dump(r["tasks"], open(OUT / f"tasks_{tag}.json", "w"))
    mon, cost = monitor_rows(r["logs"], r["tasks"])
    mon["split"], mon["family"], mon["cond"], mon["seed"] = split, fam, c, s
    mon.to_csv(OUT / f"mon_{tag}.csv", index=False)
    json.dump({"train_s": t_train, **{f"{k[0]}|{k[1]}": v for k, v in cost.items()}},
              open(OUT / f"cost_{tag}.json", "w"))
    print(f"{tag} train {t_train:.0f}s monitors {sum(cost.values()):.0f}s "
          f"acc first/last5 {np.mean([t['online_acc'] for t in r['tasks'][:5]]):.3f}/"
          f"{np.mean([t['online_acc'] for t in r['tasks'][-5:]]):.3f}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    wid, nw = int(sys.argv[1]), int(sys.argv[2])
    js = jobs()
    for i, job in enumerate(js):
        if i % nw == wid:
            do_job(job)
    print("worker done", flush=True)

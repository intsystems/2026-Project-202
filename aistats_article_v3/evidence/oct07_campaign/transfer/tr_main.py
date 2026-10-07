"""Setting T: does an alarm rule calibrated on ONE training setup transfer, without
recalibration, to heterogeneous setups?  (campaign 7 Oct 2026, agent `transfer`)

PROTOCOL (written after the pilot on seed 999, which looked only at training behaviour and the
ground-truth measures, and BEFORE any main run; nothing below is changed after results).

Task. A practitioner calibrates an alarm "training has simplified" once, on a source setup,
and deploys it unchanged on other runs. Simplification events (one per event run, at a log
index t_e drawn from the seed, 3500..6000, run length 10 000 optimizer steps):
  freeze  all layers except the linear head stop training (optimizer rebuilt on the head);
  prune   global magnitude pruning of 90 % of the weights, masks kept;
  wd      weight decay x100 (5e-4 -> 5e-2 in mean units), feature-rank collapse.
No-event runs: identical training without the event.
Source setup S: MLP 784-64-64-10 (ReLU) on MNIST, SGD lr 0.02, momentum 0.9, wd 5e-4,
batch 64 (with replacement), cross-entropy, mean reduction, fp32, monitored log = parameter
norm ||theta|| of the fp32 weights (accumulated in float64, logged every optimizer step).
Target setups (pre-specified; all differences are real training/logging differences):
  T0_fresh        S on fresh seeds (in-distribution control)
  T1_width4       width 256 (269k params; standard parametrisation, same lr)
  T2_width16      width 1024 (1.86M params; same lr)                [fewer seeds: cost]
  T3_labelsmooth  label smoothing 0.1
  T4_focal        focal loss, gamma 2
  T5_sum_accum    sum reduction, micro-batch 32 x 4 accumulation (effective batch 128),
                  lr 0.04/128 and wd 5e-4*128 (linear scaling, mean-equivalent), all logs in
                  sum units (loss x128, grad norm x128)
  T6_norm2_log    S, but the team logs squared norms (||theta||^2 and ||g||^2)
  T7_bf16         bf16 autocast of forward/loss (fp32 master weights, fp32/float64 norm log)
  T8_restart      job restarts at t_r from the checkpoint taken 500 steps earlier, optimizer
                  state reset, new data order; the log is appended in wall-clock order, so
                  it contains a level jump back. Event runs: t_r = t_e - 1500 or t_e + 1500
                  (seed-drawn); null runs: t_r in 3000..7500.
  T9_fashion      Fashion-MNIST instead of MNIST
Seeds. Calibration (S): event seeds 0-3 x 3 events (12 runs), null seeds 0-5 (6 runs).
Target i (T0..T9 -> i=0..9): seeds 100*(i+1)+j; events j=0,1,2 x 3 events (9 runs), nulls
j=0..5 (6 runs). T2: events j=0,1 (6 runs), nulls j=0..3 (4 runs). 163 runs in all.
Pilot seed 999 is never scored.

Ground truth (independent internal measurements every 50 steps, all parameters / a fixed
probe of 512 training images): fraction of parameters that moved, update participation
ratio, effective rank (entropy) and srank_0.01 of last-hidden-layer features, dormant
fraction (tau = 0.1, Sokar 2023). An event is CONFIRMED if, comparing (t_e, t_e+1000] with
[t_e-1000, t_e): freeze/prune: frac_moving ratio < 0.5; wd: erank ratio < 0.8 (computed on
(t_e+500, t_e+1500]). Hits are reported on all events and on confirmed events.

Monitors (every one calibrated on S only, with the same procedure; then frozen):
  Scalar statistics of the parameter-norm log, windows W=1000 stride 500, E6 ratio rule
  D_k = median(last M windows)/median(previous B windows) - 1, alarm when sign*D_k < -delta;
  grid M in {2,3,4}, B in {3,4,6}, sign in {+1,-1}; delta = max(0, largest drop on S null
  runs) + 0.02; (M,B,sign) maximising S hits, then shortest median delay. Windows ending
  before step 1500 are warm-up. Statistics:
    MG (primary: E20 tau4 k50), MG_t1k20 (training-log default; secondary),
    self_repeat, spectral_entropy, roughness, perm_entropy (lag 1), recurrence_rate,
    corr_dim, twonn, linear_pr (delay-space ones with tau=1, as in cnn_competitors.py),
    crossings, lag1, det_std (cifar_events.simple).
  Domain-standard / practitioner rules (same calibration logic):
    gradnorm_z      z-score of block-mean grad norm vs trailing blocks (block in {20,50,100},
                    reference {20,40} blocks, sign), threshold = 1.05 x largest S-null excursion
    gradnorm_abs    absolute threshold on window-median grad norm (sign chosen on S),
                    2 % beyond the most extreme S-null value
    gradnorm_ratio  E6 ratio rule on window-median grad norm
    loss_plateau    ReduceLROnPlateau (rel mode) on EMA(0.01) of the logged mini-batch loss,
                    theta in {1e-4,1e-3,1e-2,.05,.1}, patience in {250,...,3000}: fewest S-null
                    alarms, then most hits, then delay
    srank_abs, dormant_abs   absolute thresholds on feature srank / dormant fraction
                    (internals; window medians), 2 % beyond the most extreme S-null value
    srank_ratio, erank_ratio  E6 ratio rule on the internal measures
Scoring on each target (thresholds of S, no recalibration). Event run: hit if the first alarm
is in (t_e, t_e+5000] (T8 with t_r > t_e: and before t_r, otherwise a miss); an alarm at or
before t_e is an early false alarm. Null run: any alarm is a false alarm. Per target: hits,
false alarms (null runs), early alarms, median delay. Youden J = hit rate - null FA rate.
Overall: mean J over T1..T9 (primary) and over T0..T9.

Predictions (stated before the runs):
  H1 (primary) MG has the highest mean J over T1..T9 among all monitors. (subjective p~0.25)
  H2 On T8 MG has <= 1/6 null false alarms while self_repeat and recurrence_rate have >= 2/6.
  H3 Absolute-threshold monitors (gradnorm_abs, srank_abs, dormant_abs) have J < 0.5 on T1
     and T2 and on T5 (gradnorm_abs) - their thresholds do not transfer.
  H4 On T5 and T6 every scale-invariant scalar statistic has the same hit and FA counts as on
     T0 within +-1 run.
  H5 (descriptive) width changes the dynamics: the pre-event median of MG on the norm log at
     width 1024 differs from width 64 by > 20 %.
Costs: wall-clock per run (training) and per window (each statistic) are reported.

ADDENDUM 05:30 (after the source calibration, before looking at ANY target result).
On the source runs the parameter-norm log of the MLP is trend-dominated, and MG reaches only
4/12 calibration hits (it chose the 'rise' sign); several competitors reach 8-12/12. The
primary analysis above stays as registered. SECONDARY analysis (also fixed now): every scalar
statistic is additionally computed on the per-window linearly detrended log (least-squares
line removed inside each 1000-step window); each scalar monitor chooses raw vs detrended on
the source calibration by the same key (hits, then delay); scored on targets with the frozen
choice. Reported separately as "secondary (log choice on S)".
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = HERE / "runs"
EVENTS = ("freeze", "prune", "wd")
TARGETS = ["T0_fresh", "T1_width4", "T2_width16", "T3_labelsmooth", "T4_focal", "T5_sum_accum",
           "T6_norm2_log", "T7_bf16", "T8_restart", "T9_fashion"]


def jobs():
    J = [("source", s, e) for s in range(4) for e in EVENTS] + [("source", s, None) for s in range(6)]
    late = []
    for i, t in enumerate(TARGETS):
        nev, nnull = (2, 4) if t == "T2_width16" else (3, 6)
        base = 100 * (i + 1)
        jj = [(t, base + j, e) for j in range(nev) for e in EVENTS] + [(t, base + j, None) for j in range(nnull)]
        (late if t == "T2_width16" else J).extend(jj)
    # interleave the slow width-16 runs from the middle on so both workers stay busy to the end
    out = J[:len(J) // 3]
    rest = J[len(J) // 3:]
    k = max(1, len(rest) // max(1, len(late)))
    for n, j in enumerate(rest):
        out.append(j)
        if n % k == 0 and late:
            out.append(late.pop(0))
    return out + late


def stem(setup, seed, event):
    return f"{setup}_{event or 'none'}_s{seed}"


def job(a):
    import tr_common as C
    import tr_monitors as Mo
    setup, seed, event = a
    st = stem(setup, seed, event)
    npz, csv = RUNS / f"{st}.npz", RUNS / f"{st}_win.csv"
    t0 = time.perf_counter()
    if not npz.exists():
        C.run(setup, seed, event, wd_factor=100.0, out=npz)
    if not csv.exists():
        res = dict(np.load(npz))
        rows = Mo.window_rows(res)
        pd.DataFrame(rows).to_csv(csv, index=False)
    return st, time.perf_counter() - t0


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    J = jobs()
    print(len(J), "jobs", flush=True)
    with Pool(2, maxtasksperchild=8) as p:
        for st, dt in p.imap_unordered(job, J, chunksize=1):
            print(time.strftime("%H:%M:%S"), st, f"{dt:.0f}s", flush=True)

"""S4: early warning of loss spikes (slingshot instabilities) in a small transformer.

PROTOCOL (written 05:40 on 7 Oct 2026, before any calibration/test run; pilots used seed 100
and looked only at the loss / internals, never at MG or at competitors).

Task. Modular addition a+b mod 97, inputs [a, b, '='], answer at the last position;
train set = 30 % or 50 % of all 9 409 pairs (fixed split). Decoder-only pre-LN transformer,
2 layers, d_model 64, 4 heads, no qk-layernorm, no z-loss. Adam(W) with weight decay 0,
eps 1e-8, batch 256, linear warm-up 100 steps then constant LR, 8 000 steps.
Without weight decay Adam runs into the slingshot regime (Thilak et al. 2022): the training
loss falls by orders of magnitude, then explodes back (spike) and the cycle repeats.
Pilot 4 (seed 100): lr 1e-3 / beta2 .98 / 30 % -> 3 spikes in 7k steps; lr 2e-3 -> 1;
beta2 .95 -> none; pilot 3 lr 1e-3 / .98 / 50 % -> 1 spike in 10k.

Configurations (all used for calibration and test):
  c1 lr 1e-3 beta2 .98 train 30 %     c2 lr 2e-3 beta2 .98 train 30 %
  c3 lr 1e-3 beta2 .95 train 30 %     c4 lr 1e-3 beta2 .98 train 50 %
Calibration seeds 0-3, test seeds 10-13 (16 runs each).

Truth (truth.py, fixed in the pilot from the loss only): onset at t if the 10-step mean of
log10 loss exceeds the trailing-200 median by > 1 (10x) and > 6 MAD-sigmas; 300-step cool-down;
non-finite loss = onset.

Logs (every step): loss, grad_norm, update_norm, param_norm (cheap scalars);
internals attn_max (max attention logit), attn_ent (min head attention entropy), logz
(mean output log-partition).

Statistics (features.py), at eval times t = 500, 550, ... on the trailing window [t-500, t):
MG (E=20, tau=1, k=20, Theiler=embedding), MG_t4k50 (variant), baselines.py competitors
(self_repeat, spectral_entropy, roughness, perm_entropy lag 1, recurrence_rate/corr_dim/
twonn/linear_pr tau 1), cifar_events.simple (crossings, detrended lag-1 = Scheffer AR1,
det_std) plus detrended std (Scheffer variance) - each on each of the 4 scalar logs
(loss, grad, update norms in log10). Levels (mean / max / min of the last 50 steps) of all 7
logs: these give the practitioner rules (grad-norm / loss threshold) and the internal
domain-standard monitors (attention-logit growth, entropy collapse, output-logit drift).

Rules (identical grid for every column): score_t = sign * X_t ("level") or
sign * (X_t - median of X at t-500-50j, j=0..B-1) ("change", B in {4,10}); sign +-1.
Alarm when score_t >= theta. Scored eval times: t >= 1500 and t not in [s, s+500] for any
onset s (the window would contain the spike). A spike s (s > 1500) is warned if an alarm
occurs at some scored t in [s-500, s); lead = s - earliest such t. An alarm at t with no
onset in (t, t+500] is a false alarm; false alarms closer than 500 steps to the previous
counted one are merged. FA rate = false alarms per 10 000 scored steps.
Calibration (cal seeds only): for each column and rule, theta = the lowest value with
FA <= 2 / 10k on the calibration runs; the rule with the highest calibration recall (ties:
lower FA, then longer median lead) is frozen per column; a METHOD (e.g. "MG", "self_repeat",
"attn_max") picks its best column among its allowed logs on calibration only. Test: recall,
FA/10k, precision of alarms, median lead on seeds 10-13; threshold-free AUROC of the frozen
score for "onset within the next 500 steps" on test eval times. Secondary budget FA <= 5/10k.

Prediction (before running): the slingshot is preceded by a smooth multi-decade fall of the
loss, so loss/grad level rules should be strong; MG might drop (smooth trend = low dimension)
but is not expected to beat level rules. A tie or loss is reported as such.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs"
CONFIGS = {"c1": dict(lr=1e-3, beta2=0.98, train_frac=0.3), "c2": dict(lr=2e-3, beta2=0.98, train_frac=0.3),
           "c3": dict(lr=1e-3, beta2=0.95, train_frac=0.3), "c4": dict(lr=1e-3, beta2=0.98, train_frac=0.5)}
CAL_SEEDS, TEST_SEEDS = (0, 1, 2, 3), (10, 11, 12, 13)
STEPS = 8000


def job(a):
    cfg, seed = a
    import spk_common as C
    import features as FE
    import pandas as pd
    f = RUNS / f"{cfg}_s{seed}.npz"
    if not f.exists():
        r = C.train(seed, steps=STEPS, warm=100, wd=0.0, eps=1e-8, d=64, layers=2, heads=4,
                    batch=256, task="mod", diverge_loss=20.0, **CONFIGS[cfg])
        np.savez(f, **r)
    fx = RUNS / f"{cfg}_s{seed}_feat.csv"
    if not fx.exists():
        r = dict(np.load(f))
        t0 = time.perf_counter()
        rows, cost = FE.run_features(r)
        pd.DataFrame(rows).to_csv(fx, index=False)
        json.dump({"feature_seconds": time.perf_counter() - t0, "train_seconds": float(r["seconds"]),
                   "n_eval": len(rows), **cost}, open(RUNS / f"{cfg}_s{seed}_cost.json", "w"))
    print("done", cfg, seed, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = {"cal": CAL_SEEDS, "test": TEST_SEEDS, "all": CAL_SEEDS + TEST_SEEDS}[which]
    jobs = [(c, s) for s in seeds for c in CONFIGS]
    with Pool(3) as p:
        p.map(job, jobs, chunksize=1)

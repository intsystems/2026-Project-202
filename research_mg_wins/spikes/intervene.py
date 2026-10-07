"""Downstream test: act on alarms online (test seeds only, rules frozen from calibration).

PROTOCOL (written before any intervention run). For the frozen rule of a method (column,
family, B, sign, theta from results_cal_percolumn.csv, budget 2/10k, chosen on calibration
runs only) the statistic is computed online every 50 steps on the trailing 500-step window
from step 1500 on. Each alarm (at most one per 500 steps) halves the learning rate for the
rest of the run (preemptive LR reduction, PaLM-style practice of lowering LR around spikes).
Arms on the same test seeds/configs: none (the original test runs), MG rule, the best
non-MG rule on calibration, and a fixed schedule (LR halved at steps 3000 and 5000, the
practitioner rule with no monitor). Outcomes: number of spikes (truth.py) after step 1500,
final training log10-loss (mean of last 200 steps), final validation loss, number of LR cuts.
Usage: python intervene.py <arm> <col> <fam> <B> <sign> <theta>   (arm 'fixed' needs no rule)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spikes_main import CONFIGS, STEPS  # noqa: E402

OUT = HERE / "intervene"
CFGS = ("c1", "c2", "c4")
SEEDS = (10, 11, 12, 13)


def make_rule(col, fam, B, sign, theta):
    import features as FE
    stat, log = col.split("|")
    hist = {}
    state = {"last": -10 ** 9, "cuts": []}

    def value(logs, t):
        x = FE.transform(log, logs[log][:t])
        if stat in ("level", "max", "min"):
            seg = x[t - FE.S:t]
            return float({"level": np.mean, "max": np.max, "min": np.min}[stat](seg))
        seg = x[t - FE.W:t]
        if not np.isfinite(seg).all() or seg.std() == 0:
            return np.nan
        if stat in FE.WSTATS:
            return float(FE.WSTATS[stat](seg))
        return float(FE.simple(seg)[stat])

    def cb(step, logs):
        t = step + 1
        if t % FE.S or t < FE.W:
            return None
        need = fam == "level" and t >= 1500
        if fam == "change" or need:
            hist[t] = value(logs, t)
        if t < 1500:
            return None
        if fam == "level":
            sc = sign * hist[t]
        else:
            ref = [hist.get(t - FE.W - FE.S * k, np.nan) for k in range(B)]
            sc = sign * (hist[t] - np.nanmedian(ref)) if np.isfinite(ref).any() else np.nan
        if np.isfinite(sc) and sc >= theta and t - state["last"] >= 500:
            state["last"] = t
            state["cuts"].append(t)
            return ("lr_mult", 0.5)
        return None
    return cb, state


def fixed_rule():
    state = {"cuts": []}

    def cb(step, logs):
        if step + 1 in (3000, 5000):
            state["cuts"].append(step + 1)
            return ("lr_mult", 0.5)
        return None
    return cb, state


def job(a):
    arm, rule, cfg, seed = a
    import spk_common as C
    f = OUT / f"{arm}_{cfg}_s{seed}.npz"
    if f.exists():
        return
    cb, state = fixed_rule() if arm == "fixed" else make_rule(*rule)
    r = C.train(seed, steps=STEPS, warm=100, wd=0.0, eps=1e-8, d=64, layers=2, heads=4, batch=256,
                task="mod", diverge_loss=20.0, intervene=cb, **CONFIGS[cfg])
    r["cuts"] = np.array(state["cuts"], np.int64)
    np.savez(f, **r)
    print("done", arm, cfg, seed, "cuts", state["cuts"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    arm = sys.argv[1]
    rule = None if arm == "fixed" else (sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    json.dump({"arm": arm, "rule": rule}, open(OUT / f"rule_{arm}.json", "w"))
    jobs = [(arm, rule, c, s) for s in SEEDS for c in CFGS]
    with Pool(int(os.environ.get("NWORK", "3"))) as p:
        p.map(job, jobs, chunksize=1)

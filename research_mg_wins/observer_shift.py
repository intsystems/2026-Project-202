"""E11: does a log monitor survive changes in how the log is written?

PROTOCOL (fixed before evaluation). Logs: E5 test runs (seeds 10-13, all 11 arms), already
recorded. Rules: the E6-style rules of results_cnn/chosen_rules.json, calibrated on raw E4
logs, applied unchanged. Each logging change is applied to every test log; event arms keep
their training event at step 4 000.
  units      x10 from a random step in [2 000, 9 000] (log switched to other units)
  square     whole log replaced by its square (norm^2 logged instead of norm)
  lognorm    whole log replaced by its natural log
  ema        from a random step in [2 000, 9 000] the logger reports EMA(0.9) values
  restart    level shift of +0.5 % at a random step in [2 000, 9 000] (resumed run)
  noise      additive N(0, (1e-4 x mean)^2) measurement noise on the whole log
Random steps are drawn from a fixed generator per (seed, arm, change).
Scores per statistic and change: hits on strong events (28), false alarms on the no-event
arms base and batch_up (8), alarms before the event in event runs.
Prediction: MG keeps >= 24/28 hits and <= 1/8 false alarms under every change, and has
fewer total false alarms than each competitor summed over changes.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import zlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(REF)); sys.path.insert(0, str(HERE))
import detector as D  # noqa: E402
from cnn_competitors import EXTRA  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from cifar_events import simple  # noqa: E402

OUT = HERE / "results_observer"
CHANGES = ("units", "square", "lognorm", "ema", "restart", "noise")
STRONG = ["lr10", "lr100", "freeze_head", "freeze_bias", "prune50", "prune80", "prune95"]


def transform(x, change, rng):
    x = x.astype(float).copy()
    k = int(rng.integers(2000, 9000))
    if change == "units":
        x[k:] *= 10
    elif change == "square":
        x = x ** 2
    elif change == "lognorm":
        x = np.log(x)
    elif change == "ema":
        y = x.copy()
        for t in range(k, len(x)):
            y[t] = 0.9 * y[t - 1] + 0.1 * x[t]
        x = y
    elif change == "restart":
        x[k:] *= 1.005
    elif change == "noise":
        x = x + 1e-4 * x.mean() * rng.normal(size=len(x))
    return x


def windows(args):
    path, change = args
    arm, seed = path.stem[5:].rsplit("_s", 1)
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{arm}|{change}".encode()))
    x = transform(np.load(path)["param_norm"], change, rng)
    rows = []
    for a in range(0, len(x) - D.W + 1, D.S):
        seg = x[a:a + D.W]
        rows.append({"arm": arm, "seed": int(seed), "change": change, "start": a, "end": a + D.W,
                     "MG": estimate(seg, D.CFG).MG, **simple(seg), **{k: f(seg) for k, f in EXTRA.items()}})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted((REF / "results_graded").glob("logs_*_s*.npz"))
    jobs = [(f, c) for f in files for c in CHANGES]
    path = OUT / "windows.csv"
    if path.exists():
        W = pd.read_csv(path)
    else:
        with Pool(5) as p:
            W = pd.DataFrame([r for rr in p.map(windows, jobs, chunksize=2) for r in rr])
        W.to_csv(path, index=False)
    rules = json.load(open(HERE / "results_cnn" / "chosen_rules.json"))
    rows = []
    for (change, stat), r in [((c, s), rules[s]) for c in CHANGES for s in rules]:
        w = W[W.change == change]
        pr = D.evaluate(w, stat, r["M"], r["B"], r["delta"], r["sign"], ("base", "batch_up"), D.EVENT)
        rows.append({"change": change, "stat": stat,
                     "hits": int(pr[pr.arm.isin(STRONG)].hit.sum()),
                     "null_alarms": int(pr[pr.arm.isin(["base", "batch_up"])].false_alarm.sum()),
                     "early": int(pr[pr.event].false_alarm.sum())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.pivot(index="stat", columns="change", values="hits").to_string())
    print(R.pivot(index="stat", columns="change", values="null_alarms").to_string())
    print(R.pivot(index="stat", columns="change", values="early").to_string())
    tot = R.groupby("stat")[["hits", "null_alarms", "early"]].sum().sort_values("null_alarms")
    print(tot.to_string())


if __name__ == "__main__":
    main()

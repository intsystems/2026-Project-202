"""E12 analysis, following E12_PROTOCOL.md."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REF))
from cnn_competitors import EXTRA  # noqa: E402
from cifar_events import simple  # noqa: E402
import detector as D  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

R = HERE / "results_noise"
LOGS = ("param_norm", "batch_loss", "grad_norm")
W, S, WARM = 1000, 500, 1500
DELTAS = (0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5)


def runs(tag):
    out = []
    for f in sorted(R.glob(f"eval_{tag}_s*.json")):
        e = json.load(open(f))
        z = np.load(str(f).replace("eval_", "logs_").replace(".json", ".npz"))
        out.append({"name": f.stem[5:], "seed": e["args"]["seed"], "noise": e["args"]["noise"],
                    "eval": pd.DataFrame(e["eval"]), "logs": {k: z[k].astype(float) for k in z.files}})
    return out


def window_stats(run):
    rows = []
    for lg in LOGS:
        x = run["logs"][lg]
        for a in range(0, len(x) - W + 1, S):
            seg = x[a:a + W]
            r = {"name": run["name"], "log": lg, "start": a, "end": a + W, "MG": estimate(seg, D.CFG).MG, **simple(seg)}
            for k, f in EXTRA.items():
                try:
                    r[k] = float(f(seg))
                except Exception:
                    r[k] = np.nan
            rows.append(r)
    return rows


def score(run, step):
    ev = run["eval"]
    i = int(np.argmin(np.abs(ev.step.to_numpy() - step)))
    return ev.test_acc.max() - ev.test_acc.iloc[i]


def alarm_step(g, stat, M, B, sign, delta, last):
    for end, d in D.drop_series(g, stat, M, B, sign):
        if end > WARM and np.isfinite(d) and d < -delta:
            return end
    return last


def main():
    cal, test = runs("cal"), runs("test")
    wpath = R / "windows.csv"
    if wpath.exists():
        Wd = pd.read_csv(wpath)
    else:
        with Pool(12) as p:
            Wd = pd.DataFrame([r for rr in p.map(window_stats, cal + test) for r in rr])
        Wd.to_csv(wpath, index=False)
    last = len(cal[0]["logs"]["param_norm"])
    results = []

    # fixed step
    fixed = int(np.median([r["eval"].step[r["eval"].val_acc.idxmax()] for r in cal]))
    results.append(("fixed", {"step": fixed}, [score(r, fixed) for r in test]))

    # loss plateau and train accuracy
    def plateau_step(r, f):
        m = pd.Series(r["logs"]["batch_loss"]).rolling(1000).mean().to_numpy()
        for t in range(2000, last, 250):
            if m[t] > (1 - f) * m[t - 1000]:
                return t
        return last

    def acc_step(r, a):
        m = pd.Series(r["logs"]["batch_acc"]).rolling(1000).mean().to_numpy()
        hit = np.where(m >= a)[0]
        return int(hit[0]) if len(hit) else last

    for name, fn, grid in (("loss_plateau", plateau_step, (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)),
                           ("train_acc", acc_step, tuple(np.round(np.arange(0.3, 0.95, 0.05), 2)))):
        best = min(grid, key=lambda v: np.mean([score(r, fn(r, v)) for r in cal]))
        results.append((name, {"param": best}, [score(r, fn(r, best)) for r in test]))

    # log statistics with the change detector
    stats = ["MG", "crossings", "lag1", "det_std"] + list(EXTRA)
    for stat in stats:
        best = None
        for lg, (M, B), sign, delta in itertools.product(LOGS, itertools.product((2, 3, 4), (3, 4, 6)), (1, -1), DELTAS):
            regs = []
            for r in cal:
                g = Wd[(Wd.name == r["name"]) & (Wd.log == lg)]
                regs.append(score(r, alarm_step(g, stat, M, B, sign, delta, last)))
            key = np.mean(regs)
            if best is None or key < best[0]:
                best = (key, dict(log=lg, M=M, B=B, sign=sign, delta=delta))
        p = best[1]
        regs = []
        for r in test:
            g = Wd[(Wd.name == r["name"]) & (Wd.log == p["log"])]
            regs.append(score(r, alarm_step(g, stat, p["M"], p["B"], p["sign"], p["delta"], last)))
        results.append((stat, p, regs))

    rows = []
    for name, p, regs in results:
        reg = np.array(regs)
        by_noise = {f"regret_n{n}": float(np.mean([g for g, r in zip(reg, test) if r["noise"] == n]))
                    for n in (0.2, 0.4, 0.6)}
        rows.append({"rule": name, "params": json.dumps(p), "mean_regret": reg.mean(), **by_noise})
    out = pd.DataFrame(rows).sort_values("mean_regret")
    out.to_csv(R / "stopping.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print("never stopping:", np.mean([score(r, last) for r in test]).round(3))
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Transfer of every CNN-calibrated rule to ResNet-18 (E7 logs), without recalibration.

Rules: results_cnn/chosen_rules.json (calibrated on the 14.7k-parameter CNN of E4).
Logs: parameter norm of ResNet-18 (11.2M parameters), scratch and finetune, seeds 20-22,
event at step 4 000; observer controls x10 scale and 16-step smoothing built from base as in
E7. Scores: freeze_head hits (6), other-event hits (lr10, lr100, prune80, prune95: 24),
false alarms on base, batch_up, scale, smooth (24).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
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
from cifar_events import simple  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

OUT = HERE / "results_resnet"
NULL = ("base", "batch_up", "scale", "smooth")


def windows(args):
    scen, key, x = args
    arm, seed = key.rsplit("_s", 1)
    variants = {arm: x}
    if arm == "base":
        sc = x.copy(); sc[4000:] *= 10
        sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[4000:] = c[4000:]
        variants.update({"scale": sc, "smooth": sm})
    rows = []
    for name, v in variants.items():
        for a in range(0, len(v) - D.W + 1, D.S):
            seg = v[a:a + D.W]
            rows.append({"scen": scen, "arm": name, "seed": f"{scen}{seed}", "start": a, "end": a + D.W,
                         "MG": estimate(seg, D.CFG).MG, **simple(seg), **{k: f(seg) for k, f in EXTRA.items()}})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    jobs = []
    for scen in ("scratch", "finetune"):
        z = np.load(REF / "results_resnet" / f"observers_{scen}.npz")
        for k in sorted({k.split("|")[0] for k in z.keys()}):
            jobs.append((scen, k, z[f"{k}|param_norm"].astype(float)))
    with Pool(6) as p:
        W = pd.DataFrame([r for rr in p.map(windows, jobs) for r in rr])
    W.to_csv(OUT / "windows.csv", index=False)
    rules = json.load(open(HERE / "results_cnn" / "chosen_rules.json"))
    rows = []
    for s, r in rules.items():
        pr = D.evaluate(W, s, r["M"], r["B"], r["delta"], r["sign"], NULL, 4000)
        rows.append({"stat": s, "freeze_head": f"{int(pr[pr.arm == 'freeze_head'].hit.sum())}/6",
                     "other_events": f"{int(pr[pr.arm.isin(['lr10', 'lr100', 'prune80', 'prune95'])].hit.sum())}/24",
                     "false_alarms": f"{int(pr[pr.arm.isin(NULL)].false_alarm.sum())}/24",
                     "early": int(pr[pr.event].false_alarm.sum())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "summary.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()

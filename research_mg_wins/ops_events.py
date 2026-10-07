"""E14: operational events that a training monitor must NOT report as simplification.

PROTOCOL (fixed before any run). Training exactly as E5 (3-conv CNN, 10 000 CIFAR-10
images, SGD lr 0.02, momentum 0.9, wd 5e-4, batch 64, 10 000 steps), new seeds 20-23.
At step 4 000 one operational event happens; training otherwise continues unchanged:
  resume_exact  state dicts of model and optimizer saved and reloaded (sanity: no change)
  opt_reset     resumed without optimizer state (momentum buffers lost)
  reseed        the batch sampler is re-seeded (resumed with a new data order)
  bf16          forward/backward under bfloat16 autocast from then on
  clip          gradient-norm clipping at twice the median gradient norm of steps 3 000-4 000
Rules: results_cnn/chosen_rules.json (calibrated on E4, unchanged). Score per statistic:
number of runs with any alarm (a false alarm: nothing simplified). Prediction: MG raises
at most 2 alarms over the 20 runs and fewer than each competitor with >= 24/28 strong-event
hits in E6.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(REF)); sys.path.insert(0, str(HERE))
from cifar_events import model, simple  # noqa: E402
from cifar_cache import load  # noqa: E402
import detector as D  # noqa: E402
from cnn_competitors import EXTRA  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

OUT = HERE / "results_ops"
ARMS = ("resume_exact", "opt_reset", "reseed", "bf16", "clip")
EVENT, STEPS = 4000, 10000
_DATA = None


def train(args):
    global _DATA
    arm, seed = args
    torch.set_num_threads(1)
    if _DATA is None:
        _DATA = load()
    X, y = _DATA[0], _DATA[1]
    net = model(seed)
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    pn = np.empty(STEPS); gn = np.empty(STEPS)
    bf16 = False; clip = None
    for t in range(STEPS):
        if t == EVENT:
            if arm == "resume_exact":
                sd, so = copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict())
                net.load_state_dict(sd); opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)
                opt.load_state_dict(so)
            elif arm == "opt_reset":
                opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)
            elif arm == "reseed":
                rng = np.random.default_rng(777 + seed)
            elif arm == "bf16":
                bf16 = True
            elif arm == "clip":
                clip = 2 * float(np.median(gn[3000:4000]))
        idx = torch.as_tensor(rng.integers(0, len(X), 64))
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            loss = lossf(net(X[idx]), y[idx])
        loss.backward()
        g = torch.sqrt(sum((p.grad ** 2).sum() for p in params)).item()
        gn[t] = g
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        with torch.no_grad():
            pn[t] = torch.sqrt(sum((p ** 2).sum() for p in params)).item()
    rows = []
    for a in range(0, STEPS - D.W + 1, D.S):
        seg = pn[a:a + D.W]
        rows.append({"arm": arm, "seed": seed, "start": a, "end": a + D.W, "MG": estimate(seg, D.CFG).MG,
                     **simple(seg), **{k: f(seg) for k, f in EXTRA.items()}})
    np.save(OUT / f"pn_{arm}_s{seed}.npy", pn)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(a, s) for s in (20, 21, 22, 23) for a in ARMS]
    with Pool(4) as p:
        W = pd.DataFrame([r for rr in p.map(train, jobs) for r in rr])
    W.to_csv(OUT / "windows.csv", index=False)
    rules = json.load(open(HERE / "results_cnn" / "chosen_rules.json"))
    rows = []
    for s, r in rules.items():
        pr = D.evaluate(W, s, r["M"], r["B"], r["delta"], r["sign"], ARMS, EVENT)
        rows.append({"stat": s, **{a: int(pr[pr.arm == a].false_alarm.sum()) for a in ARMS},
                     "total": int(pr.false_alarm.sum())})
    R = pd.DataFrame(rows).sort_values("total")
    R.to_csv(OUT / "summary.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()

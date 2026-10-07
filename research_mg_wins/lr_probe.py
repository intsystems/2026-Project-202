"""E13: choosing the learning rate from a short probe run.

A practitioner tries several (lr, batch) settings for a short probe and keeps the one that
looks best. Ranking by the probe loss is the standard and is known to be short-sighted
(short-horizon bias). Each setting is trained to the end here, so the final clean test
accuracy is known; the question is which statistic of the probe logs predicts it best.

This file trains and logs; lr_probe_analyse.py evaluates the probe statistics.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import json
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research_trajectory_reference"))
from cifar_events import load, model  # noqa: E402

OUT = HERE / "results_lr"
LRS = (0.003, 0.01, 0.03, 0.1, 0.3)
BATCHES = (32, 128)
_DATA = None


def job(args):
    global _DATA
    lr, bs, seed, steps = args
    torch.set_num_threads(1)
    if _DATA is None:
        _DATA = load()
    X, y, _, _, Xt, yt = _DATA
    net = model(seed)
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(3000 + seed)
    logs = {k: np.full(steps, np.nan, dtype=np.float32) for k in ("param_norm", "batch_loss", "grad_norm")}
    t0 = time.perf_counter()
    ev = []
    for t in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), bs))
        loss = lossf(net(X[idx]), y[idx])
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            logs["grad_norm"][t] = torch.sqrt(sum((p.grad ** 2).sum() for p in params)).item()
        opt.step()
        with torch.no_grad():
            logs["param_norm"][t] = torch.sqrt(sum((p ** 2).sum() for p in params)).item()
        logs["batch_loss"][t] = loss.item()
        if (t + 1) % 2500 == 0:
            with torch.no_grad():
                ev.append({"step": t + 1, "test_acc": (net(Xt).argmax(1) == yt).float().mean().item()})
    name = f"lr{lr}_b{bs}_s{seed}"
    np.savez_compressed(OUT / f"logs_{name}.npz", **logs)
    final = ev[-1]["test_acc"] if ev and len(ev) == steps // 2500 else 0.1
    rec = {"lr": lr, "batch": bs, "seed": seed, "final_acc": final, "eval": ev,
           "wall": time.perf_counter() - t0}
    json.dump(rec, open(OUT / f"eval_{name}.json", "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = [(lr, b, s, a.steps) for s in a.seeds for lr, b in product(LRS, BATCHES)
            if not (OUT / f"eval_lr{lr}_b{b}_s{s}.json").exists()]
    print(len(jobs), "runs", flush=True)
    with Pool(a.procs) as p:
        for r in p.imap_unordered(job, jobs):
            print(r["lr"], r["batch"], r["seed"], round(r["final_acc"], 3), f"{r['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

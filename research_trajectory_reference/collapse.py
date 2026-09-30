"""E8: a cheap alarm for dying ReLUs, from the parameter-norm log.

PROTOCOL (written before any E8 run; the spike settings come from collapse_pilot.py,
seed 99, where only dead fractions, loss and accuracy were looked at, never MG).

Question. A short learning-rate spike can kill ReLU channels permanently: their input
goes negative on every example, the gradient through them vanishes, and the network
continues with fewer active units. That is a real, permanent simplification, and it is
measured independently by the fraction of dead channels, which needs the activations of
every channel over a probe set. Does MG on the parameter-norm log -- free, no forward
pass -- raise an alarm when channels die, and stay quiet after a spike of the same kind
that kills nothing?

Model/data/optimiser: exactly E4/E5 (cifar_events.py): the 3-conv CNN (80 ReLU channels,
no normalisation), 10 000 CIFAR-10 images, SGD momentum 0.9, wd 5e-4, lr 0.02, batch
64, 10 000 steps. Spike at step 4 000. Seeds 30-35.

Arms (factor x lr for `dur` steps):
  base                no spike
  shock20, shock30    x20 / x30 for 1 step    -- pilot: loss jumps, nothing dies
  kill15x3, kill20x2  x15 / 3 steps, x20 / 2  -- pilot: ~9-12 % of channels die
  kill50, kill100     x50 / x100 for 1 step   -- pilot: ~15 % / ~44 % die
Observer controls on base logs: x10 scale and 16-step smoothing after step 4 000.
Deaths are stochastic, so the ground truth is MEASURED per run, not the arm label:
  dead fraction  = share of the 80 ReLU channels whose output is 0 on all 500 probe
                   images, every 250 steps;
  collapse run   = median dead fraction over [5 000, 9 000] minus the median over
                   [2 000, 4 000) >= 0.05 (at least 4 channels);
  network death  = final training accuracy below 0.2 (reported apart: any statistic
                   sees a dead network).
MG: frozen from E5 (E=20, tau=1, k=20, Theiler = embedding span, windows 1 000 / 500).
Detector: frozen rule from results_detector/chosen_rules.json (MG: M=2, B=3, delta=0.088).

Predictions.
  P1  the MG detector alarms within 5 000 steps after the spike in >= 75 % of collapse
      runs that are not network deaths;
  P2  it alarms in <= 15 % of non-collapse runs (base, shocks that kill nothing, scale,
      smoothing);
  P3  across all runs the MG change (after/before) correlates negatively with the
      change in dead fraction (Spearman < 0);
  P4  a detector on the RISE of the mini-batch loss (the naive "training broke" alarm,
      calibrated like E6 on the E5 no-event runs) alarms on the harmless shocks more
      often than the MG detector does.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from cifar_events import load, model, simple  # noqa: E402
from collapse_pilot import dead_fraction  # noqa: E402

RES = HERE / "results_collapse"
CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
W, S, EVENT = 1000, 500, 4000
ARMS = {"base": (1, 0), "shock20": (20, 1), "shock30": (30, 1), "kill15x3": (15, 3),
        "kill20x2": (20, 2), "kill50": (50, 1), "kill100": (100, 1)}


def train(arm, seed, steps, data):
    X, y, *_ = data
    Xp = X[:500]
    factor, dur = ARMS[arm]
    net = model(seed)
    opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    logs = {"param_norm": np.full(steps, np.nan), "batch_loss": np.full(steps, np.nan)}
    dead = []
    t_dead = 0.0
    for t in range(steps):
        lr = 0.02 * (factor if EVENT <= t < EVENT + dur else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.as_tensor(rng.integers(0, len(X), 64))
        opt.zero_grad()
        loss = lossf(net(X[idx]), y[idx])
        if not torch.isfinite(loss):
            return logs, pd.DataFrame(dead), {"diverged_at": t}
        loss.backward()
        opt.step()
        with torch.no_grad():
            logs["param_norm"][t] = torch.cat([p.reshape(-1) for p in net.parameters()]).norm().item()
        logs["batch_loss"][t] = loss.item()
        if t % 250 == 0 or t == steps - 1:
            t0 = time.perf_counter()
            f, per = dead_fraction(net, Xp)
            t_dead += time.perf_counter() - t0
            dead.append({"step": t, "dead": f, "dead_l2": per[1], "dead_l3": per[2]})
    with torch.no_grad():
        acc = (net(X[:2000]).argmax(1) == y[:2000]).float().mean().item()
    return logs, pd.DataFrame(dead), {"train_acc": acc, "t_dead_total": t_dead}


def windows(logs, arm, seed):
    x = logs["param_norm"]
    variants = {arm: x}
    if arm == "base":
        sc = x.copy(); sc[EVENT:] *= 10
        sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[EVENT:] = c[EVENT:]
        variants.update({"scale": sc, "smooth": sm})
    rows = []
    for name, v in variants.items():
        for a in range(0, len(v) - W + 1, S):
            seg = v[a:a + W]
            bl = logs["batch_loss"][a:a + W]
            t0 = time.perf_counter()
            mg = estimate(seg, CFG).MG
            rows.append({"arm": name, "seed": seed, "start": a, "end": a + W, "MG": mg,
                         "t_MG": time.perf_counter() - t0, "loss_median": float(np.median(bl)),
                         **simple(seg)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[30, 31, 32, 33, 34, 35])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()
    torch.set_num_threads(6)
    RES.mkdir(parents=True, exist_ok=True)
    data = load()
    rows, deads, meta = [], [], []
    for seed in args.seeds:
        for arm in args.arms:
            t0 = time.perf_counter()
            logs, dead, m = train(arm, seed, args.steps, data)
            np.savez_compressed(RES / f"logs_{arm}_s{seed}.npz", **logs)
            dead["arm"], dead["seed"] = arm, seed
            deads.append(dead)
            if "diverged_at" not in m:
                rows += windows(logs, arm, seed)
            meta.append({"arm": arm, "seed": seed, **m, "wall_s": time.perf_counter() - t0})
            print(f"{arm:9s} s{seed} {m} dead_end {dead.dead.iloc[-1] if len(dead) else 'nan':.3f} "
                  f"{meta[-1]['wall_s']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RES / "windows.csv", index=False)
            pd.concat(deads).to_csv(RES / "dead.csv", index=False)
            json.dump(meta, open(RES / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()

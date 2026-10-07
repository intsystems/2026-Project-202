"""E5: graded, explicit simplification events in a CIFAR-10 CNN (confirmatory run).

PROTOCOL (written before running; E5 uses seeds never used before).

Why. E4 found MG on the parameter-norm log separating three simplification events
from controls, but by 8-13 % only. Two changes were chosen on E4's logs alone
(`explore_observer.py`, seeds 0-3) and are frozen here: the estimator configuration
of the article (E=20, tau=1, k=20, Theiler = embedding span) on windows of 1 000
steps, and the raw parameter norm as the observer. E5 also makes the events explicit
and graded, so each family has a dose-response, and gives the effect longer to build.

Model, data, optimiser: exactly E4 (cifar_events.py). 10 000 steps, event at 4 000.
Seeds 10, 11, 12, 13.

Arms.
  controls   base; batch_up (64 -> 256, less noise, same parameters moving);
             observer controls on base logs: x10 scale and 16-step smoothing after 4 000.
  lr         lr / 3, lr / 10, lr / 100
  freeze     conv1+conv2 frozen (65 % of parameters still move), everything but the
             head frozen (2.2 %), everything but the head bias frozen (10 parameters)
  prune      global magnitude pruning of 50 %, 80 %, 95 % of weights, mask kept.

Primary observer: parameter norm (free: O(P), no forward pass).
Primary statistic: r = median MG over windows starting in [5 000, 9 000] divided by the
median over windows inside [2 000, 4 000). Windows 1 000, stride 500.
Independent confirmation, measured: fraction of parameters that move after the event
(update std > 1e-7 in the window) and the participation ratio of the covariance of the
parameter UPDATES in the window (unlike the trajectory PR, not dominated by the
random-walk shape), plus mean |update| for the lr family.

Pre-registered predictions.
  P1  every event arm has median r below every control arm's median r;
  P2  the strongest dose of each family is below every single control run (all seeds);
  P3  within each family r decreases with dose (Spearman over runs < 0);
  P4  scale leaves MG unchanged; smoothing and batch_up do not lower it by more than
      the base arm does.
Secondary, reported, not tested: MG of the mini-batch loss, per-layer norms, simple
competitors on the parameter norm (trend crossings, lag-1, detrended std), IAAFT ratio.
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
from actdim.estimator.surrogates import iaaft  # noqa: E402
from cifar_events import load, model, simple  # noqa: E402

RES = HERE / "results_graded"
CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
W, S, EVENT = 1000, 500, 4000
ARMS = ("base", "batch_up", "lr3", "lr10", "lr100", "freeze12", "freeze_head", "freeze_bias",
        "prune50", "prune80", "prune95")


def pr(G):
    s = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else float("nan")


def train(arm, seed, steps, data):
    X, y, *_ = data
    Xt, yt = data[4], data[5]
    net = model(seed)
    params = list(net.parameters())
    names = [n for n, _ in net.named_parameters()]
    P = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    bs, masks = 64, None
    traj = np.empty((steps, P), dtype=np.float32)
    logs = {"param_norm": np.empty(steps), "batch_loss": np.empty(steps)}
    for n in names:
        logs[f"norm_{n}"] = np.empty(steps)
    for t in range(steps):
        if t == EVENT:
            if arm.startswith("lr"):
                for g in opt.param_groups:
                    g["lr"] /= float(arm[2:])
            elif arm.startswith("freeze"):
                keep = {"freeze12": lambda n: not (n.startswith("0.") or n.startswith("3.")),
                        "freeze_head": lambda n: n.startswith("10."),
                        "freeze_bias": lambda n: n == "10.bias"}[arm]
                train_p = [p for n, p in zip(names, params) if keep(n)]
                for n, p in zip(names, params):
                    p.requires_grad_(keep(n))
                opt = torch.optim.SGD(train_p, lr=0.02, momentum=0.9, weight_decay=5e-4)
            elif arm == "batch_up":
                bs = 256
            elif arm.startswith("prune"):
                frac = float(arm[5:]) / 100
                w = [p for p in params if p.dim() > 1]
                allw = torch.cat([p.detach().abs().reshape(-1) for p in w])
                thr = torch.quantile(allw, frac)
                masks = [(p.detach().abs() > thr).float() for p in w]
                with torch.no_grad():
                    for p, m in zip(w, masks):
                        p.mul_(m)
        idx = torch.as_tensor(rng.integers(0, len(X), bs))
        opt.zero_grad()
        loss = lossf(net(X[idx]), y[idx])
        loss.backward()
        opt.step()
        if masks is not None:
            with torch.no_grad():
                for p, m in zip([p for p in params if p.dim() > 1], masks):
                    p.mul_(m)
        with torch.no_grad():
            flat = torch.cat([p.detach().reshape(-1) for p in params])
            traj[t] = flat.numpy()
            logs["param_norm"][t] = flat.norm().item()
            logs["batch_loss"][t] = loss.item()
            for n, p in zip(names, params):
                logs[f"norm_{n}"][t] = p.detach().norm().item()
    with torch.no_grad():
        acc = (net(Xt).argmax(1) == yt).float().mean().item()
    return traj, logs, {"P": P, "test_acc": acc, "nonzero_final": int((flat != 0).sum())}


def score(traj, logs, arm, seed):
    steps = len(traj)
    variants = {arm: logs["param_norm"]}
    if arm == "base":
        x = logs["param_norm"]
        sc = x.copy(); sc[EVENT:] *= 10
        sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:steps]; sm[EVENT:] = c[EVENT:]
        variants.update({"scale": sc, "smooth": sm})
    ref = {}
    for a in range(0, steps - W + 1, S):
        U = np.diff(traj[a:a + W].astype(np.float64), axis=0)
        sd = U.std(0)
        Uc = U - U.mean(0)
        ref[a] = {"update_PR": pr(Uc @ Uc.T), "moving_frac": float((sd > 1e-7).mean()),
                  "update_size": float(np.abs(U).mean())}
    rows = []
    for name, x in variants.items():
        for a in range(0, steps - W + 1, S):
            seg = x[a:a + W]
            t0 = time.perf_counter()
            mg = estimate(seg, CFG).MG
            row = {"arm": name, "seed": seed, "start": a, "MG": mg,
                   "t_MG": time.perf_counter() - t0, **ref[a], **simple(seg)}
            rng = np.random.default_rng(a)
            row["MG_surr"] = float(np.median([estimate(iaaft(seg, rng=rng), CFG).MG for _ in range(2)]))
            if name == arm:
                row["MG_batch_loss"] = estimate(logs["batch_loss"][a:a + W], CFG).MG
                for k in logs:
                    if k.startswith("norm_") and k.endswith("weight"):
                        row[f"MG_{k}"] = estimate(logs[k][a:a + W], CFG).MG
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()
    torch.set_num_threads(6)
    RES.mkdir(parents=True, exist_ok=True)
    data = load()
    rows, meta = [], []
    for seed in args.seeds:
        for arm in args.arms:
            t0 = time.perf_counter()
            traj, logs, m = train(arm, seed, args.steps, data)
            np.savez_compressed(RES / f"logs_{arm}_s{seed}.npz", **logs)
            rows += score(traj, logs, arm, seed)
            del traj
            meta.append({"arm": arm, "seed": seed, **m, "wall_s": time.perf_counter() - t0})
            print(f"{arm:12s} s{seed} test {m['test_acc']:.3f} nonzero {m['nonzero_final']} "
                  f"{meta[-1]['wall_s']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RES / "windows.csv", index=False)
            json.dump(meta, open(RES / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()

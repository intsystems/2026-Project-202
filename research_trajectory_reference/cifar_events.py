"""E4: practical simplification events in a CIFAR-10 CNN, seen from the loss log.

PROTOCOL (written before any run; changes after the first result go to the report).

Question. Standard training practice contains events that simplify the optimisation
by construction: a step decay of the learning rate, freezing the backbone to train only
the head (transfer-learning style), magnitude pruning of 80 % of the weights. Does MG on
a loss log that training already writes register them, when the simplification is
confirmed independently?

Model/data. CIFAR-10 (official batches), 1 000 training images per class (10 000),
fixed split seed 20260929. CNN: conv3x3 3->16, ReLU, maxpool; conv 16->32, ReLU,
maxpool; conv 32->32, ReLU, global average pool; linear 32->10 (14 730 parameters).
SGD, momentum 0.9, weight decay 5e-4, lr 0.02, batch 64 with replacement, 8 000 steps.
Intervention at step 4 000. Seeds 0-3.

Arms (intervention at T/2):
  base      nothing
  lr_step   lr / 10                           (standard step schedule)
  freeze    only the linear head keeps training (backbone frozen)
  prune     global magnitude pruning of 80 % of conv/linear weights, mask kept
  batch_up  batch 64 -> 256: less gradient noise, same parameters moving
            (amplitude control: the dimension should NOT fall much)
Observer controls on base logs (no retraining): x10 scale after T/2; 16-step moving
average after T/2 (same trajectory, different log).

Independent confirmation ("expensive"): detrended participation ratio of the full
parameter trajectory, all 14 730 weights stored at every step, window 500 steps.
Cheap: MG on (i) the mini-batch training loss -- free, training computes it --
and (ii) the loss on a fixed probe of 100 training images (one extra forward).
MG config frozen from the SGD pilot: E=10, tau=1, k=10, Theiler = embedding span,
window 500, stride 250. Also E=20 (identifiability), 3 IAAFT surrogates per window,
and simple competitors: trend crossings, lag-1 autocorrelation, relative detrended std.

Primary comparison: per run, median over post windows (start >= 4 500) divided by
median over pre windows (inside [2 000, 4 000)); arms compared by these ratios.
Prediction: MG ratio lower in lr_step, freeze, prune than in base; batch_up near base;
scale identical to base. Known risk (from the pilot): smoothing the log lowers MG at
tau=1 although the trajectory is unchanged. We report it whatever happens.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402
from pilot import detrended_pr  # noqa: E402

DATA = Path(r"C:\Users\karlo\notebooks\Detecting Optimization Regimes via Convergent Cross Mapping"
            r"\Poisoned_batch\data\cifar-10-batches-py")
RES = HERE / "results_cifar"
CFG = EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding")
CFG2 = CFG.replace(max_E=20)
W, S = 500, 250
ARMS = ("base", "lr_step", "freeze", "prune", "batch_up")
MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)


def load():
    def read(name):
        with open(DATA / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32) / 255.0, np.array(d[b"labels"])
    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    X, y = np.concatenate(xs), np.concatenate(ys)
    xt, yt = read("test_batch")
    rng = np.random.default_rng(20260929)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), 1000, replace=False) for c in range(10)])
    probe = np.concatenate([rng.choice(idx[y[idx] == c], 10, replace=False) for c in range(10)])
    f = lambda a: torch.tensor((a - MEAN) / STD, dtype=torch.float32)  # noqa: E731
    return (f(X[idx]), torch.tensor(y[idx]), f(X[probe]), torch.tensor(y[probe]),
            f(xt[:2000]), torch.tensor(yt[:2000]))


def model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))


def train(arm, seed, steps, data):
    X, y, Xp, yp, Xt, yt = data
    net = model(seed)
    params = list(net.parameters())
    P = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    half, bs, masks = steps // 2, 64, None
    traj = np.empty((steps, P), dtype=np.float32)
    logs = {k: np.empty(steps) for k in ("batch_loss", "probe_loss", "grad_norm", "param_norm")}
    t_train = t_probe = 0.0
    for t in range(steps):
        if t == half:
            if arm == "lr_step":
                for g in opt.param_groups:
                    g["lr"] /= 10
            elif arm == "freeze":
                for p in params[:-2]:
                    p.requires_grad_(False)
                opt = torch.optim.SGD(params[-2:], lr=0.02, momentum=0.9, weight_decay=5e-4)
            elif arm == "batch_up":
                bs = 256
            elif arm == "prune":
                w = [p for p in params if p.dim() > 1]
                allw = torch.cat([p.detach().abs().reshape(-1) for p in w])
                thr = torch.quantile(allw, 0.8)
                masks = [(p.detach().abs() > thr).float() for p in w]
                with torch.no_grad():
                    for p, m in zip(w, masks):
                        p.mul_(m)
        t0 = time.perf_counter()
        idx = torch.as_tensor(rng.integers(0, len(X), bs))
        opt.zero_grad()
        loss = lossf(net(X[idx]), y[idx])
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        opt.step()
        if masks is not None:
            with torch.no_grad():
                for p, m in zip([p for p in params if p.dim() > 1], masks):
                    p.mul_(m)
        t_train += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            logs["probe_loss"][t] = lossf(net(Xp), yp).item()
        t_probe += time.perf_counter() - t0
        with torch.no_grad():
            flat = torch.cat([p.detach().reshape(-1) for p in params])
        traj[t] = flat.numpy()
        logs["batch_loss"][t] = loss.item()
        logs["grad_norm"][t] = gn.item()
        logs["param_norm"][t] = flat.norm().item()
    with torch.no_grad():
        acc = (net(Xt).argmax(1) == yt).float().mean().item()
        tr = (net(X[:2000]).argmax(1) == y[:2000]).float().mean().item()
    return traj, logs, {"P": P, "test_acc": acc, "train_acc": tr, "t_train": t_train,
                        "t_probe": t_probe, "traj_MB": traj.nbytes / 2 ** 20,
                        "nonzero_final": int((flat != 0).sum())}


def simple(seg):
    t = np.arange(len(seg))
    r = seg - np.polyval(np.polyfit(t, seg, 1), t)
    return {"crossings": trend_crossings(seg), "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            "det_std": float(r.std() / (abs(seg.mean()) + 1e-12))}


def windows(traj, logs, arm, seed):
    steps = len(traj)
    variants = {arm: logs}
    if arm == "base":                      # observer controls, same trajectory
        sc = {k: v.copy() for k, v in logs.items()}
        sm = {k: v.copy() for k, v in logs.items()}
        for k in logs:
            sc[k][steps // 2:] *= 10
            c = np.convolve(logs[k], np.ones(16) / 16, mode="full")[:steps]
            sm[k][steps // 2:] = c[steps // 2:]
        variants.update({"scale": sc, "smooth": sm})
    rows = []
    prs = {}
    for a in range(0, steps - W + 1, S):
        t0 = time.perf_counter()
        prs[a] = (detrended_pr(traj[a:a + W]), time.perf_counter() - t0)
    for name, lg in variants.items():
        for a in range(0, steps - W + 1, S):
            row = {"arm": name, "seed": seed, "start": a, "traj_PR": prs[a][0], "t_PR": prs[a][1]}
            for o in ("batch_loss", "probe_loss"):
                seg = lg[o][a:a + W]
                t0 = time.perf_counter()
                row[f"MG_{o}"] = estimate(seg, CFG).MG
                row[f"t_MG_{o}"] = time.perf_counter() - t0
                row[f"MG20_{o}"] = estimate(seg, CFG2).MG
                rng = np.random.default_rng(a)
                row[f"MGs_{o}"] = float(np.median([estimate(iaaft(seg, rng=rng), CFG).MG
                                                   for _ in range(3)]))
                for k, v in simple(seg).items():
                    row[f"{k}_{o}"] = v
            for o in ("grad_norm", "param_norm"):
                row[f"MG_{o}"] = estimate(lg[o][a:a + W], CFG).MG
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
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
            rows += windows(traj, logs, arm, seed)
            del traj
            meta.append({"arm": arm, "seed": seed, **m, "wall_s": time.perf_counter() - t0})
            print(f"{arm:9s} s{seed}  test {m['test_acc']:.3f} train {m['train_acc']:.3f} "
                  f"{meta[-1]['wall_s']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RES / "windows.csv", index=False)
            json.dump(meta, open(RES / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()

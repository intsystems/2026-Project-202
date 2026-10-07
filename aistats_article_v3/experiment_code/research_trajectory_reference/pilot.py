"""Pilot: does MG on a scalar log follow the *trajectory's* effective dimension?

Hypothesis behind this directory. The earlier applied attempts (neural collapse,
LLM LR-switch) scored MG against the wrong expensive reference: representation
geometry (NC1/NC2), Hessian top eigenvalue, activation PR. MG on a delay
reconstruction measures how many directions the *optimisation trajectory*
fluctuates in. The expensive quantity it should be compared with is therefore
the windowed detrended participation ratio of the parameter trajectory itself,
which needs every weight at every step (memory O(P T), an SVD per window).
That is exactly the pair section 7.2/7.3 of the article used on grokking.

Setting: MLP 64-64-64-10 on sklearn digits, SGD. At step T/2 one intervention
per arm; all of them change the trajectory's fluctuations in a known direction.

  base      nothing                      (control: expect ratio ~1 for both)
  lr_drop   learning rate / 10           (noise ball shrinks; dimension?)
  freeze    only the last layer trains   (available directions 9k -> 650)
  batch_up  batch 16 -> 256              (gradient noise / 16)
  scale     base, but the logged scalars are multiplied by 10 after T/2
            (scale must not move MG; checks it is not reading amplitude)

Per window we record: detrended PR of the full parameter trajectory (reference),
MG on four scalar logs, and cheap non-MG scalar competitors.
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
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
ARMS = ("base", "lr_drop", "freeze", "batch_up", "scale", "smooth", "momentum")


def detrended_pr(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    t = np.arange(len(X), dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    X = X - X.mean(0, keepdims=True)
    X = X - np.outer(t, (t[:, None] * X).sum(0) / (t @ t))
    G = X @ X.T                                  # window x window Gram: cheap for P >> W
    s = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(s.sum() ** 2 / (s ** 2).sum())


def train(arm: str, seed: int, steps: int, lr: float) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32)
    y = torch.tensor(y)
    perm = torch.tensor(np.random.default_rng(12345).permutation(len(X)))
    probe, train_idx = perm[:200], perm[200:]
    net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 10))
    params = list(net.parameters())
    P = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    half = steps // 2
    traj = np.empty((steps, P), dtype=np.float32)
    logs = {k: np.empty(steps) for k in ("batch_loss", "probe_loss", "param_norm", "grad_norm")}
    bs = 16
    for t in range(steps):
        if t == half:
            if arm == "lr_drop":
                for g in opt.param_groups:
                    g["lr"] = lr / 10
            elif arm == "freeze":
                for p in params[:-2]:
                    p.requires_grad_(False)
            elif arm == "batch_up":
                bs = 256
            elif arm == "momentum":
                for g in opt.param_groups:
                    g["momentum"] = 0.9
                    g["lr"] = lr / 10          # same effective step size lr/(1-beta)
        idx = train_idx[torch.tensor(rng.integers(0, len(train_idx), bs))]
        opt.zero_grad()
        loss = lossf(net(X[idx]), y[idx])
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        opt.step()
        with torch.no_grad():
            flat = torch.cat([p.detach().reshape(-1) for p in params])
            traj[t] = flat.numpy()
            logs["batch_loss"][t] = loss.item()
            logs["probe_loss"][t] = lossf(net(X[probe]), y[probe]).item()
            logs["param_norm"][t] = flat.norm().item()
            logs["grad_norm"][t] = gn.item()
    if arm == "scale":
        for k in logs:
            logs[k][half:] *= 10.0
    if arm == "smooth":
        # observer control: same trajectory, the logged scalar is a 16-step causal
        # moving average after T/2 -- roughness/autocorrelation change, dynamics do not
        for k in logs:
            x = logs[k].copy()
            c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]
            logs[k][half:] = c[half:]
    with torch.no_grad():
        acc = (net(X[train_idx]).argmax(1) == y[train_idx]).float().mean().item()
    return {"traj": traj, "logs": logs, "P": P, "train_acc": acc}


def cheap_stats(x: np.ndarray) -> dict:
    t = np.arange(len(x))
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    lag1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    return {"det_std_rel": r.std() / (abs(x.mean()) + 1e-12), "lag1": lag1,
            "crossings": trend_crossings(x)}


def analyse(run: dict, cfg: EstimatorConfig, W: int, S: int) -> pd.DataFrame:
    rows = []
    T = len(run["traj"])
    for a in range(0, T - W + 1, S):
        row = {"start": a, "centre": a + W // 2}
        t0 = time.perf_counter()
        row["traj_PR"] = detrended_pr(run["traj"][a:a + W])
        row["t_ref"] = time.perf_counter() - t0
        for k, x in run["logs"].items():
            seg = x[a:a + W]
            t0 = time.perf_counter()
            e = estimate(seg, cfg)
            row[f"t_mg_{k}"] = time.perf_counter() - t0
            row[f"MG_{k}"] = e.MG
            row[f"deg_{k}"] = e.degenerate
            for s, v in cheap_stats(seg).items():
                row[f"{s}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding",
                          window=args.window, stride=args.stride)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, meta = [], []
    for seed in args.seeds:
        for arm in args.arms:
            t0 = time.perf_counter()
            run = train(arm, seed, args.steps, args.lr)
            ttrain = time.perf_counter() - t0
            np.savez_compressed(OUT / f"logs_{arm}_s{seed}.npz", **run["logs"])
            df = analyse(run, cfg, args.window, args.stride)
            df["arm"], df["seed"] = arm, seed
            frames.append(df)
            meta.append({"arm": arm, "seed": seed, "P": run["P"], "train_acc": run["train_acc"],
                         "train_s": ttrain, "traj_MB": run["traj"].nbytes / 2**20})
            print(f"{arm:9s} seed {seed}: acc {run['train_acc']:.3f}  train {ttrain:5.1f}s", flush=True)
    pd.concat(frames).to_csv(OUT / f"windows_{'_'.join(args.arms)}.csv" if args.arms != list(ARMS) else OUT / "windows.csv", index=False)
    json.dump({"args": vars(args), "cfg": cfg.as_dict(), "runs": meta},
              open(OUT / "meta.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

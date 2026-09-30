"""Full-batch arm: is there a training regime where the log is NOT spectrum-only?

pilot.py showed that on SGD logs MG equals MG of an IAAFT surrogate (ratio 0.98,
IQR 0.93-1.03): the estimate is a function of the power spectrum alone, as
Osborne & Provenzale predict for coloured noise. MG can only add something a
spectral statistic cannot where the log carries deterministic nonlinear
structure. Full-batch descent at a large step (edge of stability) is the cheapest
real-training candidate: no sampling noise, oscillation from the dynamics itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.surrogates import iaaft  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402
from pilot import detrended_pr  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
W, S = 500, 250


def run(lr: float, seed: int, arm: str, steps: int):
    torch.manual_seed(seed)
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32)
    y = torch.tensor(y)
    net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 10))
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    traj, loss_log = [], []
    for t in range(steps):
        if arm == "freeze" and t == steps // 2:
            for p in params[:-2]:
                p.requires_grad_(False)
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
        traj.append(torch.cat([p.detach().reshape(-1) for p in params]).numpy().copy())
        loss_log.append(loss.item())
    x = np.array(loss_log)
    if arm == "smooth":
        c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]
        x[steps // 2:] = c[steps // 2:]
    return np.array(traj, dtype=np.float32), x


def main() -> None:
    cfg = EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding")
    rows = []
    steps = 6000
    for lr in (0.5, 2.0, 5.0, 10.0):
        for seed in (0, 1, 2):
            for arm in ("base", "freeze", "smooth"):
                traj, x = run(lr, seed, arm, steps)
                for a in range(1000, steps - W + 1, S):
                    seg = x[a:a + W]
                    rng = np.random.default_rng(1)
                    mg = estimate(seg, cfg).MG
                    ms = np.median([estimate(iaaft(seg, rng=rng), cfg).MG for _ in range(3)])
                    rows.append(dict(lr=lr, seed=seed, arm=arm, start=a,
                                     traj_PR=detrended_pr(traj[a:a + W]), MG=mg, MG_surr=ms,
                                     crossings=trend_crossings(seg),
                                     rises=float(np.mean(np.diff(seg) > 0))))
                print(lr, seed, arm, f"final loss {x[-1]:.2e}", flush=True)
    d = pd.DataFrame(rows)
    d["rel"] = d.MG / d.MG_surr
    d.to_csv(RES / "fullbatch.csv", index=False)
    pre, post = d[d.start + W <= steps // 2], d[d.start >= steps // 2 + W]
    r = (post.groupby(["lr", "arm", "seed"]).median(numeric_only=True)
         / pre.groupby(["lr", "arm", "seed"]).median(numeric_only=True))
    pd.set_option("display.width", 200)
    print("\npost/pre:\n", r.groupby(["lr", "arm"])[["traj_PR", "MG", "MG_surr", "rel", "crossings"]]
          .median().round(2).to_string())
    print("\nlevel pre, base:\n", pre[pre.arm == "base"].groupby("lr")[["traj_PR", "MG", "MG_surr", "rel", "rises"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()

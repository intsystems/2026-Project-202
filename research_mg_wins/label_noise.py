"""E12: early stopping without a validation set under label noise, from scalar training logs.

With noisy labels a network first fits the clean structure, then memorises the wrong labels
and its clean accuracy falls. The best stopping step is found expensively with a clean
held-out set. The question is whether a rule on a free scalar log (parameter norm,
mini-batch loss, gradient norm) stops near that step, better than validation-free rules
a practitioner already has.

This file trains and logs; stopping rules are evaluated in label_noise_analyse.py.
"""
from __future__ import annotations

import os
import sys

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research_trajectory_reference"))
from cifar_cache import load  # noqa: E402

OUT = HERE / "results_noise"


def net(seed, width):
    torch.manual_seed(seed)
    w = width
    return nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(2 * w * 16, 10))


def run(seed, noise, width, lr, steps, eval_every, threads, data):
    torch.set_num_threads(threads)
    X, y, _, _, Xt, yt = data
    Xv, yv, Xs, ys = Xt[:1000], yt[:1000], Xt[1000:], yt[1000:]
    rng = np.random.default_rng(5000 + seed)
    yn = y.clone()
    flip = rng.random(len(y)) < noise
    yn[flip] = torch.tensor((y[flip].numpy() + rng.integers(1, 10, flip.sum())) % 10)
    m = net(seed, width)
    params = list(m.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    logs = {k: np.empty(steps, dtype=np.float32) for k in ("param_norm", "batch_loss", "grad_norm", "batch_acc")}
    ev = []
    t0 = time.perf_counter()
    for t in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), 128))
        out = m(X[idx])
        loss = lossf(out, yn[idx])
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            logs["grad_norm"][t] = torch.sqrt(sum((p.grad ** 2).sum() for p in params)).item()
        opt.step()
        with torch.no_grad():
            logs["param_norm"][t] = torch.sqrt(sum((p ** 2).sum() for p in params)).item()
            logs["batch_loss"][t] = loss.item()
            logs["batch_acc"][t] = (out.argmax(1) == yn[idx]).float().mean().item()
        if t % eval_every == 0 or t == steps - 1:
            with torch.no_grad():
                m.eval()
                acc = lambda A, B: (m(A).argmax(1) == B).float().mean().item()  # noqa: E731
                fit_noisy = (m(X[flip][:1000]).argmax(1) == yn[flip][:1000]).float().mean().item()
                ev.append({"step": t, "val_acc": acc(Xv, yv), "test_acc": acc(Xs, ys),
                           "fit_noisy": fit_noisy})
                m.train()
    return logs, ev, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data = load()
    logs, ev, wall = run(a.seed, a.noise, a.width, a.lr, a.steps, a.eval_every, a.threads, data)
    name = f"{a.tag}_s{a.seed}_n{a.noise}_w{a.width}_lr{a.lr}"
    np.savez_compressed(OUT / f"logs_{name}.npz", **logs)
    json.dump({"args": vars(a), "eval": ev, "wall": wall}, open(OUT / f"eval_{name}.json", "w"))
    best = max(ev, key=lambda e: e["val_acc"])
    print(name, f"wall {wall:.0f}s best val {best['val_acc']:.3f} at {best['step']} final {ev[-1]['val_acc']:.3f}",
          " ".join(f"{e['step']}:{e['val_acc']:.2f}" for e in ev[::8]), flush=True)


if __name__ == "__main__":
    main()

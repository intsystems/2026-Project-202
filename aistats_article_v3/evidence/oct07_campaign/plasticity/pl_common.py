"""Shared engine for the plasticity experiments (S1): permuted-MNIST task stream, MLP,
per-step scalar logs, per-task internal probes, interventions.

Nothing in here looks at MG or at any competitor; it only produces logs.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "research_gan_collapse" / "data" / "MNIST" / "raw"
_CACHE = {}


def _read_idx(path):
    with open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(n_train=10000, n_test=2000, pool=2):
    """Fixed subset (split seed 20261007), 2x2 average pooled to 14x14 = 196 inputs."""
    key = (n_train, n_test, pool)
    if key in _CACHE:
        return _CACHE[key]
    X = _read_idx(RAW / "train-images-idx3-ubyte").astype(np.float32) / 255.0
    y = _read_idx(RAW / "train-labels-idx1-ubyte").astype(np.int64)
    Xt = _read_idx(RAW / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
    yt = _read_idx(RAW / "t10k-labels-idx1-ubyte").astype(np.int64)
    rng = np.random.default_rng(20261007)
    i = rng.choice(len(X), n_train, replace=False)
    j = rng.choice(len(Xt), n_test, replace=False)
    X, y, Xt, yt = X[i], y[i], Xt[j], yt[j]
    if pool > 1:
        X = X.reshape(-1, 28 // pool, pool, 28 // pool, pool).mean((2, 4))
        Xt = Xt.reshape(-1, 28 // pool, pool, 28 // pool, pool).mean((2, 4))
    X = X.reshape(len(X), -1)
    Xt = Xt.reshape(len(Xt), -1)
    mu, sd = X.mean(), X.std()
    X = (X - mu) / sd
    Xt = (Xt - mu) / sd
    out = (torch.tensor(X), torch.tensor(y), torch.tensor(Xt), torch.tensor(yt))
    _CACHE[key] = out
    return out


class MLP(nn.Module):
    def __init__(self, d_in, width, depth=2, d_out=10):
        super().__init__()
        dims = [d_in] + [width] * depth
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(width, d_out)

    def forward(self, x, return_feats=False):
        feats = []
        for lin in self.hidden:
            x = F.relu(lin(x))
            feats.append(x)
        out = self.head(x)
        return (out, feats) if return_feats else out


def make_opt(model, cfg):
    if cfg["opt"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("mom", 0.0),
                           weight_decay=cfg.get("wd", 0.0))


@torch.no_grad()
def probe(model, Xp, tau_dormant=(0.0, 0.025, 0.1)):
    """Internal monitors on a probe batch (domain methods; need activations / weights)."""
    _, feats = model(Xp, return_feats=True)
    out = {}
    for t in tau_dormant:
        fr = []
        for h in feats:                       # ReDo score: mean|h_i| / mean over units
            s = h.abs().mean(0)
            s = s / (s.mean() + 1e-12)
            fr.append((s <= t).float().mean().item())
        out[f"dormant_{t}"] = float(np.mean(fr))
    h = feats[-1]
    sv = torch.linalg.svdvals(h).numpy()
    c = np.cumsum(sv) / max(sv.sum(), 1e-12)
    out["srank"] = int(np.searchsorted(c, 0.99) + 1)                       # Kumar et al. 2021
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    out["erank"] = float(np.exp(-(p * np.log(p)).sum()))                   # Roy & Vetterli
    out["wnorm"] = float(torch.sqrt(sum((q ** 2).sum() for q in model.parameters())).item())
    return out


@torch.no_grad()
def shrink_perturb(model, lam=0.8, sigma=0.01, gen=None):
    for q in model.parameters():
        q.mul_(lam).add_(sigma * torch.randn(q.shape, generator=gen))


def full_reset(model, gen_seed):
    torch.manual_seed(gen_seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()


@torch.no_grad()
def redo(model, Xp, tau=0.1):
    """ReDo (Sokar et al. 2023): reinit incoming weights of dormant units, zero outgoing."""
    _, feats = model(Xp, return_feats=True)
    layers = list(model.hidden) + [model.head]
    n = 0
    for li, h in enumerate(feats):
        s = h.abs().mean(0)
        s = s / (s.mean() + 1e-12)
        dead = s <= tau
        n += int(dead.sum())
        if dead.any():
            lin, nxt = layers[li], layers[li + 1]
            fresh = nn.Linear(lin.in_features, lin.out_features)
            lin.weight[dead] = fresh.weight[dead]
            lin.bias[dead] = 0.0
            nxt.weight[:, dead] = 0.0
    return n


def run_stream(cfg, seed, n_tasks, policy=None, log_steps=True, data=None, intervention="reset",
               verbose=False):
    """Train on a stream of permuted tasks.

    cfg: dict(opt, lr, width, depth, wd, steps, batch, task='perm'|'label').
    policy(history) -> bool, called at each task boundary (before the next task) with the
    logs so far; True means apply the intervention now.
    Returns dict of per-step logs and per-task records.
    """
    X, y, Xt, yt = data if data is not None else load_mnist()
    d = X.shape[1]
    g = torch.Generator().manual_seed(1000 + seed)
    torch.manual_seed(seed)
    model = MLP(d, cfg["width"], cfg.get("depth", 2))
    opt = make_opt(model, cfg)
    rng = np.random.default_rng(seed)
    steps, bs = cfg["steps"], cfg["batch"]
    N = len(X)
    T = n_tasks * steps
    logs = {k: np.zeros(T, np.float32) for k in ("loss", "acc", "gnorm", "pnorm")}
    tasks = []
    params = list(model.parameters())
    probe_idx = torch.tensor(rng.choice(N, 512, replace=False))
    levels = cfg.get("noise_levels")            # family F2: per-task input-noise difficulty
    rng_noise = np.random.default_rng(10_000 + seed)
    g_noise = torch.Generator().manual_seed(20_000 + seed)
    n_int = 0
    t0 = time.perf_counter()
    for k in range(n_tasks):
        if policy is not None and k > 0 and policy(logs, tasks, k):
            if intervention == "reset":
                full_reset(model, seed * 7919 + k)
            elif intervention == "sp":
                shrink_perturb(model, gen=g)
            opt = make_opt(model, cfg)
            tasks[-1]["intervened_after"] = True
            n_int += 1
        if cfg.get("task", "perm") == "perm":
            perm = torch.tensor(rng.permutation(d))
            Xk, Xtk = X[:, perm], Xt[:, perm]
            yk, ytk = y, yt
        else:                                   # label permutation
            lp = torch.tensor(rng.permutation(10))
            Xk, Xtk, yk, ytk = X, Xt, lp[y], lp[yt]
        sigma = 0.0
        if levels:
            sigma = float(rng_noise.choice(levels))
            if sigma > 0:
                Xk = Xk + sigma * torch.randn(Xk.shape, generator=g_noise)
                Xtk = Xtk + sigma * torch.randn(Xtk.shape, generator=g_noise)
        order = torch.tensor(rng.integers(0, N, size=steps * bs))
        accs = 0.0
        for s in range(steps):
            i = order[s * bs:(s + 1) * bs]
            xb, yb = Xk[i], yk[i]
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            t = k * steps + s
            with torch.no_grad():
                a = (out.argmax(1) == yb).float().mean().item()
                gn = torch.sqrt(sum((q.grad ** 2).sum() for q in params)).item()
            opt.step()
            with torch.no_grad():
                pn = torch.sqrt(sum((q ** 2).sum() for q in params)).item()
            logs["loss"][t], logs["acc"][t], logs["gnorm"][t], logs["pnorm"][t] = loss.item(), a, gn, pn
            accs += a
        rec = {"task": k, "online_acc": accs / steps, "sigma": sigma}
        with torch.no_grad():
            rec["test_acc"] = float((model(Xtk).argmax(1) == ytk).float().mean())
        rec.update(probe(model, Xk[probe_idx]))
        rec["intervened_after"] = False
        tasks.append(rec)
        if verbose and (k % 5 == 0 or k == n_tasks - 1):
            print(f"  task {k:3d} online {rec['online_acc']:.3f} test {rec['test_acc']:.3f} "
                  f"dorm {rec['dormant_0.0']:.2f} srank {rec['srank']} w {rec['wnorm']:.1f} "
                  f"t={time.perf_counter() - t0:.0f}s", flush=True)
    return {"logs": logs, "tasks": tasks, "n_interventions": n_int, "seconds": time.perf_counter() - t0}

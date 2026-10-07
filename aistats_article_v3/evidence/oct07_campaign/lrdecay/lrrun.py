"""Training engine for S3 (when to decay the LR). One call = one (condition, seed) unit.

A unit trains the constant-LR "trunk" for the whole budget T and logs every scalar that any
decay rule may read (mini-batch loss, parameter norm, gradient norm, Pflug inner product of
successive stochastic gradients, SASA fluctuation-dissipation terms, distance from init for
the Pesme et al. diagnostic, validation loss every EVAL steps). Model + momentum buffers are
checkpointed on the decay grid t_j = j*T/16, j = 2..15. From every checkpoint a "branch" is
trained to T with lr/10 (one x10 decay at t_j). Because the trajectory of any single-decay
policy is identical to the trunk until it decays, the final test accuracy/loss of ANY rule
whose alarm falls in (t_{j-1}, t_j] is the outcome of branch j (decay applied at the next
checkpoint, causal). No decay = the trunk itself. A cosine-schedule run on the same batch
sequence is trained as an extra competitor.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "research_trajectory_reference"))
from cifar_cache import load  # noqa: E402
from cifar_events import model  # noqa: E402

MOM, WD, EVAL, NGRID = 0.9, 5e-4, 100, 16
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        X, y, _, _, Xt, yt = load()
        perm = np.random.default_rng(777).permutation(len(X))
        va, tr = perm[:1000], perm[1000:]
        _DATA = (X[tr], y[tr], X[va], y[va], Xt.clone(), yt.clone())
        del X, y, Xt, yt
        import gc; gc.collect()
    return _DATA


def grid(T):
    return [j * T // NGRID for j in range(2, NGRID)]


def evaluate(net, X, y, lossf):
    with torch.no_grad():
        out = torch.cat([net(X[i:i + 500]) for i in range(0, len(X), 500)])
        return float(lossf(out, y).item()), float((out.argmax(1) == y).float().mean().item())


def setup(c, seed):
    Xtr, ytr, Xv, yv, Xt, yt = data()
    rng = np.random.default_rng(10_000 + 97 * seed + int(c.get("cid", 0)))
    n = c.get("n", len(Xtr))
    if n < len(Xtr):
        sub = np.sort(rng.choice(len(Xtr), n, replace=False))
        X, y = Xtr[sub], ytr[sub].clone()
    else:
        X, y = Xtr, ytr.clone()
    noise = c.get("noise", 0.0)
    if noise > 0:
        flip = rng.random(n) < noise
        y[flip] = torch.tensor((y[flip].numpy() + rng.integers(1, 10, flip.sum())) % 10)
    batches = rng.integers(0, n, size=(c["T"], c["bs"]))
    return X, y, Xv, yv, Xt, yt, batches


def step(net, opt, X, y, idx, lossf):
    opt.zero_grad()
    loss = lossf(net(X[idx]), y[idx])
    loss.backward()
    return loss


def run_unit(c, seed, out_dir, do_branches=True, do_cosine=True, grid_override=None):
    torch.set_num_threads(1)
    T, lr0 = c["T"], c["lr"]
    X, y, Xv, yv, Xt, yt, batches = setup(c, seed)
    lossf = nn.CrossEntropyLoss()
    net = model(seed)
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=lr0, momentum=MOM, weight_decay=WD)
    G = grid_override or grid(T)
    keys = ("batch_loss", "param_norm", "grad_norm", "gg", "xg", "dd", "dist0")
    logs = {k: np.full(T, np.nan) for k in keys}
    ev = []
    x0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    gprev = None
    ckpt = {}
    t0 = time.perf_counter()
    for t in range(T):
        if t in G:
            ckpt[t] = (copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict()))
        if t % EVAL == 0:
            vl, va = evaluate(net, Xv, yv, lossf)
            rec = {"step": t, "val_loss": vl, "val_acc": va}
            if t % (8 * EVAL) == 0:
                rec["test_loss"], rec["test_acc"] = evaluate(net, Xt, yt, lossf)
            ev.append(rec)
        idx = torch.as_tensor(batches[t])
        loss = step(net, opt, X, y, idx, lossf)
        with torch.no_grad():
            x = torch.cat([p.detach().reshape(-1) for p in params])
            g = torch.cat([p.grad.reshape(-1) for p in params]) + WD * x   # stochastic grad of objective
            logs["batch_loss"][t] = loss.item()
            logs["grad_norm"][t] = g.norm().item()
            logs["xg"][t] = float(x @ g)
            logs["gg"][t] = float(g @ gprev) if gprev is not None else np.nan
            gprev = g
        opt.step()
        with torch.no_grad():
            x1 = torch.cat([p.detach().reshape(-1) for p in params])
            d = torch.cat([opt.state[p]["momentum_buffer"].reshape(-1) for p in params])
            logs["dd"][t] = float(d @ d)
            logs["param_norm"][t] = x1.norm().item()
            logs["dist0"][t] = (x1 - x0).norm().item()
        if not np.isfinite(logs["batch_loss"][t]):
            break
    t_trunk = time.perf_counter() - t0
    fl, fa = evaluate(net, Xt, yt, lossf)
    vl, va = evaluate(net, Xv, yv, lossf)
    trunk_state = net.state_dict()
    res = {"cond": c, "seed": seed, "grid": G, "none": {"test_loss": fl, "test_acc": fa, "val_loss": vl,
                                                         "val_acc": va}, "branches": {}, "t_trunk": t_trunk}
    np.savez_compressed(out_dir / f"logs_{c['name']}_s{seed}.npz", **logs)
    json.dump({"ev": ev}, open(out_dir / f"ev_{c['name']}_s{seed}.json", "w"))
    t0 = time.perf_counter()
    if do_branches:
        for tj in G:
            sd, osd = ckpt.pop(tj)
            net.load_state_dict(sd)
            opt = torch.optim.SGD(params, lr=lr0, momentum=MOM, weight_decay=WD)
            opt.load_state_dict(osd)
            for gr in opt.param_groups:
                gr["lr"] = lr0 / 10
            for t in range(tj, T):
                step(net, opt, X, y, torch.as_tensor(batches[t]), lossf)
                opt.step()
            fl, fa = evaluate(net, Xt, yt, lossf)
            vl, va = evaluate(net, Xv, yv, lossf)
            res["branches"][str(tj)] = {"test_loss": fl, "test_acc": fa, "val_loss": vl, "val_acc": va}
            json.dump(res, open(out_dir / f"res_{c['name']}_s{seed}.json", "w"), indent=1)
    res["t_branches"] = time.perf_counter() - t0
    if do_cosine:
        t0 = time.perf_counter()
        net = model(seed)
        params = list(net.parameters())
        opt = torch.optim.SGD(params, lr=lr0, momentum=MOM, weight_decay=WD)
        for t in range(T):
            for gr in opt.param_groups:
                gr["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * t / T))
            step(net, opt, X, y, torch.as_tensor(batches[t]), lossf)
            opt.step()
        fl, fa = evaluate(net, Xt, yt, lossf)
        vl, va = evaluate(net, Xv, yv, lossf)
        res["cosine"] = {"test_loss": fl, "test_acc": fa, "val_loss": vl, "val_acc": va}
        res["t_cosine"] = time.perf_counter() - t0
    res["done"] = True
    json.dump(res, open(out_dir / f"res_{c['name']}_s{seed}.json", "w"), indent=1)
    del trunk_state
    return res

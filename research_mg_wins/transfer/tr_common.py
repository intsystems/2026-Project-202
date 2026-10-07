"""Shared training code for setting T (monitor transfer across heterogeneous setups).

One run = MLP (784-w-w-10, ReLU) trained with SGD+momentum on MNIST or Fashion-MNIST for
STEPS optimizer steps; optionally a simplification event at log index t_e (freeze all but
the head, global magnitude pruning of 90 % of weights, or a weight-decay jump), optionally a
restart from a checkpoint with optimizer reset. Logged every optimizer step: parameter norm
(fp32 master weights, accumulated in float64), raw gradient norm, mini-batch loss.
Ground truth every TRUTH_EVERY steps: fraction of parameters that moved, update participation
ratio, feature effective rank / srank of the last hidden layer and dormant-unit fraction on a
fixed probe of 512 training images.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = {"mnist": ROOT / "research_gan_collapse" / "data" / "MNIST" / "raw",
        "fashion": Path(r"C:\Users\karlo\notebooks\OPTIMIZATION-METHODS-COURSE\Домашние задания"
                        r"\Домашнее задание 6\FashionMNIST\raw")}
NORM = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}
STEPS, TRUTH_EVERY, CKPT_BACK = 10000, 50, 500
WD0, LR0, MOM = 5e-4, 0.02, 0.9

_CACHE = {}


def _idx(path):
    with open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(ds):
    if ds not in _CACHE:
        d = DATA[ds]
        X = torch.from_numpy(_idx(d / "train-images-idx3-ubyte").reshape(-1, 784).copy())
        y = torch.from_numpy(_idx(d / "train-labels-idx1-ubyte").astype(np.int64))
        Xt = torch.from_numpy(_idx(d / "t10k-images-idx3-ubyte").reshape(-1, 784).copy())
        yt = torch.from_numpy(_idx(d / "t10k-labels-idx1-ubyte").astype(np.int64))
        _CACHE[ds] = (X, y, Xt, yt)
    return _CACHE[ds]


def to_float(xb, ds):
    m, s = NORM[ds]
    return (xb.float() / 255.0 - m) / s


def mlp(width, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 10))


def focal_loss(logits, y, gamma=2.0, reduction="mean"):
    logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None])[:, 0]
    loss = -((1 - logp.exp()) ** gamma) * logp
    return loss.mean() if reduction == "mean" else loss.sum()


def make_loss(kind, reduction):
    if kind == "ce":
        f = nn.CrossEntropyLoss(reduction=reduction)
        return lambda z, y: f(z.float(), y)
    if kind == "ls":
        f = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
        return lambda z, y: f(z.float(), y)
    if kind == "focal":
        return lambda z, y: focal_loss(z, y, 2.0, reduction)
    raise ValueError(kind)


# ---- setups ---------------------------------------------------------------------------------
# lr / wd for sum reduction: linear scaling to the mean-equivalent lr of the effective batch,
# then lr_sum = lr_mean / B_eff and wd_sum = wd * B_eff so that lr*wd (shrink per step) and the
# update in parameter units are those of a mean-reduced run with batch B_eff.
BASE = dict(width=64, dataset="mnist", loss="ce", reduction="mean", micro=64, accum=1,
            lr=LR0, wd=WD0, bf16=False, restart=False, log="norm")
SETUPS = {
    "source": {},
    "T0_fresh": {},
    "T1_width4": dict(width=256),
    "T2_width16": dict(width=1024),
    "T3_labelsmooth": dict(loss="ls"),
    "T4_focal": dict(loss="focal"),
    "T5_sum_accum": dict(reduction="sum", micro=32, accum=4, lr=(LR0 * 2) / 128, wd=WD0 * 128),
    "T6_norm2_log": dict(log="norm2"),
    "T7_bf16": dict(bf16=True),
    "T8_restart": dict(restart=True),
    "T9_fashion": dict(dataset="fashion"),
}


def setup_cfg(name):
    c = dict(BASE)
    c.update(SETUPS[name])
    return c


def times_for(seed, event):
    """Event and restart log indices, a deterministic function of the seed."""
    r = np.random.default_rng(12345 + seed)
    t_e = int(3500 + 100 * r.integers(0, 26))           # 3500..6000
    if event is None:
        t_r = int(100 * r.integers(30, 76))              # 3000..7500
    else:
        t_r = t_e - 1500 if r.random() < 0.5 else t_e + 1500
    return (t_e if event else None), t_r


def svd_stats(h):
    s = torch.linalg.svdvals(h.double()).numpy()
    if s.sum() <= 0:
        return 0.0, 0
    p = s / s.sum()
    pp = p[p > 0]
    erank = float(np.exp(-(pp * np.log(pp)).sum()))
    srank = int(np.searchsorted(np.cumsum(p), 0.99) + 1)
    return erank, srank


def dormant(hs, tau=0.1):
    n = d = 0
    for h in hs:
        a = h.abs().mean(0)
        sc = a / (a.mean() + 1e-12)
        d += int((sc <= tau).sum()); n += len(a)
    return d / n


def run(setup, seed, event, wd_factor=100.0, out=None, steps=STEPS, prune_frac=0.9):
    """Train one run, return dict of arrays (and save to `out` .npz if given)."""
    torch.set_num_threads(1)
    c = setup_cfg(setup)
    ds = c["dataset"]
    X, y, Xt, yt = load(ds)
    t_e, t_r = times_for(seed, event)
    if not c["restart"]:
        t_r = None
    net = mlp(c["width"], seed)
    params = list(net.parameters())
    names = [n for n, _ in net.named_parameters()]
    P = sum(p.numel() for p in params)
    wd = c["wd"]
    mk = lambda ps, wd_: torch.optim.SGD(ps, lr=c["lr"], momentum=MOM, weight_decay=wd_)  # noqa: E731
    opt = mk(params, wd)
    lossf = make_loss(c["loss"], c["reduction"])
    rng = np.random.default_rng(1000 + seed)
    prng = np.random.default_rng(424242)
    probe_idx = torch.as_tensor(prng.choice(len(X), 512, replace=False))
    Xp, yp = to_float(X[probe_idx], ds), y[probe_idx]
    trainable = [True] * len(params)
    masks = None
    pn, gn, bl = np.empty(steps), np.empty(steps), np.empty(steps)
    nT = steps // TRUTH_EVERY
    truth = {k: np.full(nT, np.nan) for k in ("t", "frac_moving", "upr", "erank", "srank",
                                               "dormant", "probe_acc")}
    prev = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    ckpt = None
    t0 = time.perf_counter()
    for t in range(steps):
        if t_r is not None and t == t_r - CKPT_BACK:
            ckpt = copy.deepcopy(net.state_dict())
        if t_r is not None and t == t_r:
            net.load_state_dict(ckpt)
            if masks is not None:
                with torch.no_grad():
                    for p, m in zip([p for p in params if p.dim() > 1], masks):
                        p.mul_(m)
            opt = mk([p for p, tr in zip(params, trainable) if tr], wd)   # optimizer reset
            rng = np.random.default_rng(5000 + seed)                       # new data order
        if t_e is not None and t == t_e:
            if event == "freeze":
                trainable = [n.startswith("4.") for n in names]
                for p, tr in zip(params, trainable):
                    p.requires_grad_(tr)
                opt = mk([p for p, tr in zip(params, trainable) if tr], wd)
            elif event == "prune":
                w = [p for p in params if p.dim() > 1]
                thr = torch.quantile(torch.cat([p.detach().abs().reshape(-1) for p in w]), prune_frac)
                masks = [(p.detach().abs() > thr).float() for p in w]
                with torch.no_grad():
                    for p, m in zip(w, masks):
                        p.mul_(m)
            elif event == "wd":
                wd = wd * wd_factor
                for g in opt.param_groups:
                    g["weight_decay"] = wd
            else:
                raise ValueError(event)
        opt.zero_grad(set_to_none=True)
        ltot = 0.0
        for _ in range(c["accum"]):
            idx = torch.as_tensor(rng.integers(0, len(X), c["micro"]))
            xb = to_float(X[idx], ds)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=c["bf16"]):
                z = net(xb)
            loss = lossf(z, y[idx])
            loss.backward()
            ltot += loss.item()
        with torch.no_grad():
            g2 = sum(torch.linalg.vector_norm(p.grad, dtype=torch.float64) ** 2
                     for p in params if p.grad is not None)
        opt.step()
        with torch.no_grad():
            if masks is not None:
                for p, m in zip([p for p in params if p.dim() > 1], masks):
                    p.mul_(m)
            p2 = sum(torch.linalg.vector_norm(p, dtype=torch.float64) ** 2 for p in params)
        pn[t], gn[t], bl[t] = float(p2) ** 0.5, float(g2) ** 0.5, ltot
        if (t + 1) % TRUTH_EVERY == 0:
            j = (t + 1) // TRUTH_EVERY - 1
            with torch.no_grad():
                cur = torch.cat([p.detach().reshape(-1) for p in params])
                d = (cur - prev).double()
                s2, s4 = float((d ** 2).sum()), float((d ** 4).sum())
                truth["t"][j] = t + 1
                truth["frac_moving"][j] = float((d != 0).double().mean())
                truth["upr"][j] = s2 ** 2 / (P * s4) if s4 > 0 else 0.0
                prev = cur.clone()
                h1 = net[1](net[0](Xp)); h2 = net[3](net[2](h1)); zz = net[4](h2)
                truth["erank"][j], truth["srank"][j] = svd_stats(h2)
                truth["dormant"][j] = dormant([h1, h2])
                truth["probe_acc"][j] = float((zz.argmax(1) == yp).float().mean())
    wall = time.perf_counter() - t0
    with torch.no_grad():
        test_acc = float((net(to_float(Xt, ds)).argmax(1) == yt).float().mean())
    if c["log"] == "norm2":
        log_pn, log_gn = pn ** 2, gn ** 2
    else:
        log_pn, log_gn = pn.copy(), gn.copy()
    res = {"param_norm_raw": pn, "grad_norm_raw": gn, "log_pn": log_pn, "log_gn": log_gn,
           "batch_loss": bl, **{f"truth_{k}": v for k, v in truth.items()}}
    meta = {"setup": setup, "seed": seed, "event": event, "t_e": t_e, "t_r": t_r, "P": P,
            "test_acc": test_acc, "wall_s": wall, "wd_factor": wd_factor, "cfg": c}
    if out is not None:
        out = Path(out)
        np.savez_compressed(out, **res)
        json.dump(meta, open(out.with_suffix(".json"), "w"))
    return res, meta

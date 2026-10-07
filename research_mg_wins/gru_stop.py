"""E10b: early stopping for a gradient-trained recurrent simulator, chosen from its own output.

A GRU is trained by teacher forcing (one-step MSE, Adam) on the scalar series of a known
system. Every few epochs the checkpoint is run in closed loop. The practitioner keeps one
checkpoint. Standard early stopping keeps the one with the smallest one-step validation
error; that error says little about the long-run dynamics. Here every checkpoint gets the
same ground truth as E10 (Lyapunov spectrum of the closed-loop map, frequencies, amplitude)
and the same cheap scalar selectors.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as BL  # noqa: E402
import esn_dsr as E  # noqa: E402  (targets, MG config, D_H)

OUT = HERE / "results_gru"
L_TRAIN, L_VAL, L_RUN, BURN, L_LYAP = 12000, 3000, 8192, 2000, 4000


class Sim(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.cell = nn.GRUCell(1, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, h=None):           # x: (B, T) teacher-forced inputs
        B, T = x.shape
        h = torch.zeros(B, self.cell.hidden_size) if h is None else h
        ys = []
        for t in range(T):
            h = self.cell(x[:, t:t + 1], h)
            ys.append(self.out(h))
        return torch.cat(ys, 1), h


@torch.no_grad()
def warm_state(model, u):
    _, h = model(torch.tensor(u[None, :], dtype=torch.float32))
    return h


@torch.no_grad()
def closed_loop(model, h, x0, n):
    out = np.empty(n)
    x = torch.tensor([[x0]], dtype=torch.float32)
    for t in range(n):
        h = model.cell(x, h)
        x = model.out(h)
        out[t] = x.item()
        if not np.isfinite(out[t]) or abs(out[t]) > 1e3:
            out[t:] = np.nan
            break
    return out, h


def lyapunov(model, h, x0, n, k=6):
    """Top-k exponents of the closed-loop map h -> cell(out(h), h)."""
    def step(hh):
        return model.cell(model.out(hh), hh)
    H = h.detach()
    Q = torch.linalg.qr(torch.randn(H.shape[1], k, generator=torch.Generator().manual_seed(0)))[0]
    logs = torch.zeros(k)
    m = 0
    for t in range(n):
        J = torch.func.jacrev(lambda v: step(v[None])[0])(H[0])
        with torch.no_grad():
            Q, R = torch.linalg.qr(J @ Q)
            if t >= 300:
                logs += torch.log(torch.abs(torch.diagonal(R)) + 1e-300); m += 1
            H = step(H)
    return np.sort((logs / m).numpy())[::-1]


@torch.no_grad()
def one_step_mse(model, u):
    y, _ = model(torch.tensor(u[None, :-1], dtype=torch.float32))
    return float(((y[0] - torch.tensor(u[1:], dtype=torch.float32)) ** 2)[100:].mean())


def vpt(model, u_val, thr=0.3, starts=8, horizon=300):
    s = np.std(u_val)
    res = []
    for i in np.linspace(200, len(u_val) - horizon - 2, starts).astype(int):
        h = warm_state(model, u_val[:i + 1])
        pred, _ = closed_loop(model, h, u_val[i + 1], horizon)
        err = np.abs(pred - u_val[i + 2:i + 2 + horizon]) / s
        bad = np.where(~(err < thr))[0]
        res.append(bad[0] if len(bad) else horizon)
    return float(np.median(res))


def evaluate(model, u, data):
    row = {"mse1": one_step_mse(model, u[L_TRAIN:L_TRAIN + L_VAL]),
           "vpt": vpt(model, u[L_TRAIN:L_TRAIN + L_VAL])}
    h = warm_state(model, u[L_TRAIN:L_TRAIN + L_VAL - 1])
    run, h2 = closed_loop(model, h, u[L_TRAIN + L_VAL - 1], BURN + L_RUN)
    gen = run[BURN:]
    row["diverged"] = bool(not np.all(np.isfinite(gen)))
    row["std_ratio"] = float(np.nanstd(gen))
    row["lyap"] = None
    if not row["diverged"] and np.std(gen) > 1e-3:
        row["lyap"] = lyapunov(model, h2, gen[-1], L_LYAP).round(5).tolist()
        row["D_H"] = E.hellinger(gen, data)
        for name, fn in BL.ALL.items():
            try:
                row[name] = float(fn(gen))
            except Exception:
                row[name] = np.nan
        f, P = np.fft.rfftfreq(len(gen)), np.abs(np.fft.rfft((gen - gen.mean()) * np.hanning(len(gen)))) ** 2
        row["peak_freqs"] = f[np.argsort(P)[::-1][:12]].tolist()
    return row


def job(args):
    tname, dseed, mseed, hidden, lr, epochs, every = args
    torch.set_num_threads(1)
    torch.manual_seed(mseed)
    u, _ = E.target(tname, L_TRAIN + L_VAL + L_RUN + 1, dseed)
    u = (u - u[:L_TRAIN].mean()) / u[:L_TRAIN].std()
    data = u[L_TRAIN + L_VAL:L_TRAIN + L_VAL + L_RUN]
    model = Sim(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seq, batch = 200, 32
    rng = np.random.default_rng(mseed)
    rows = []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        for _ in range(40):
            idx = rng.integers(0, L_TRAIN - seq - 1, batch)
            xb = torch.tensor(np.stack([u[i:i + seq] for i in idx]), dtype=torch.float32)
            yb = torch.tensor(np.stack([u[i + 1:i + seq + 1] for i in idx]), dtype=torch.float32)
            y, _ = model(xb)
            loss = ((y - yb) ** 2)[:, 20:].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % every == 0:
            row = {"target": tname, "dseed": dseed, "mseed": mseed, "hidden": hidden, "lr": lr,
                   "epoch": ep, "train_loss": float(loss), **evaluate(model, u, data),
                   "t": time.perf_counter() - t0}
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["T2"])
    ap.add_argument("--dseeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mseeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = [(t, d, m, h, lr, args.epochs, args.every) for t in args.targets for d in args.dseeds
            for m in args.mseeds for h in args.hidden for lr in args.lrs]
    print(len(jobs), "training runs", flush=True)
    rows = []
    with Pool(args.procs) as p:
        for i, rr in enumerate(p.imap_unordered(job, jobs)):
            rows += rr
            print(i, len(rows), flush=True)
            json.dump(rows, open(OUT / f"ckpt_{args.tag}.json", "w"), default=float)
    print("done", flush=True)


if __name__ == "__main__":
    main()

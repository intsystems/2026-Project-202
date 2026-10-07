"""E2: edge of stability -- does MG on the full-batch loss count oscillating modes?

Full-batch gradient descent at step size eta drives the top Hessian eigenvalues up
to the stability threshold 2/eta and then keeps them there (Cohen et al., 2021).
Every eigen-direction whose curvature sits at the threshold oscillates; the more of
them, the more directions the trajectory moves in. That count is the expensive
quantity here, measured two ways:

  n_unstable  number of Hessian eigenvalues >= 0.9 * 2/eta (Lanczos on
              Hessian-vector products, full batch, at every checkpoint);
  traj_PR     detrended participation ratio of the full weight trajectory in the
              window (needs every weight at every step).

The cheap quantity is MG on the full-batch loss, which training computes anyway.
Every MG window carries its IAAFT-surrogate value and the value on a smoothed copy
of the same log (an observer change that leaves the trajectory untouched).

Stage "sweep": constant eta over a grid. Stage "switch": eta changed at mid-run,
down (simplification) or up (complexification), against constant-eta controls.
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
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.surrogates import iaaft  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402
from pilot import detrended_pr  # noqa: E402

RES = Path(__file__).resolve().parent / "results_eos"
CFG = EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding")
CFG2 = CFG.replace(max_E=20)           # for the identifiability ratio
W, S = 1000, 500
K_EIG = 24
SMOOTH = 16


def data():
    """Digits with one-hot targets: MSE, as in Cohen et al., keeps descent at the edge
    of stability; with cross-entropy on separable data the curvature decays as the
    margins grow and the run simply converges (checked: eta=1 reaches 6e-4, no mode
    at the threshold) or, at eta=4, blows up."""
    X, y = load_digits(return_X_y=True)
    return torch.tensor(X / 16.0, dtype=torch.float32), torch.tensor(y)


def lossf(out, y):
    return 0.5 * ((out - torch.nn.functional.one_hot(y, 10).float()) ** 2).sum(1).mean()


def model(seed: int):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 10))


def top_eigs(net, X, y, k: int = K_EIG):
    """Top-k algebraic Hessian eigenvalues of the full-batch loss, by Lanczos."""
    params = [p for p in net.parameters()]
    loss = lossf(net(X), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    n = flat.numel()
    count = [0]

    def hvp(v):
        count[0] += 1
        v = torch.tensor(np.asarray(v).reshape(-1), dtype=torch.float32)
        hv = torch.autograd.grad(flat @ v, params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv]).double().numpy()

    op = LinearOperator((n, n), matvec=hvp, dtype=np.float64)
    if not torch.isfinite(loss):
        return np.full(k, np.nan), 0
    try:
        vals = eigsh(op, k=k, which="LA", tol=1e-3, return_eigenvectors=False,
                     v0=np.random.default_rng(0).standard_normal(n))
    except Exception:  # noqa: BLE001 -- a failed factorisation is recorded, not fatal
        return np.full(k, np.nan), count[0]
    return np.sort(vals)[::-1], count[0]


def smooth(x):
    c = np.convolve(x, np.ones(SMOOTH) / SMOOTH, mode="full")[:len(x)]
    c[:SMOOTH] = x[:SMOOTH]
    return c


def score_window(seg, rng):
    t0 = time.perf_counter()
    mg = estimate(seg, CFG).MG
    t_mg = time.perf_counter() - t0
    mg_s = np.median([estimate(iaaft(seg, rng=rng), CFG).MG for _ in range(3)])
    d = np.diff(seg)
    r = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg)))
    p = np.abs(np.fft.rfft(r - r.mean()))[1:] ** 2
    p = p / p.sum()
    return {"MG": mg, "MG_surr": mg_s, "t_MG": t_mg,
            "MG_2E": estimate(seg, CFG2).MG,
            "crossings": trend_crossings(seg), "rises": float(np.mean(d > 0)),
            "spec_entropy": float(-(p * np.log(p + 1e-300)).sum() / np.log(len(p))),
            "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1])}


def run(eta0: float, eta1: float, seed: int, steps: int, every: int, tag: str):
    X, y = data()
    net = model(seed)
    params = list(net.parameters())
    opt = torch.optim.SGD(params, lr=eta0)
    P = sum(p.numel() for p in params)
    traj = np.empty((steps, P), dtype=np.float32)
    loss_log = np.empty(steps)
    ckpt = []
    t_train = t_lanczos = 0.0
    for t in range(steps):
        eta = eta0 if t < steps // 2 else eta1
        if t == steps // 2:
            for g in opt.param_groups:
                g["lr"] = eta1
        if t % every == 0:
            t0 = time.perf_counter()
            ev, nhvp = top_eigs(net, X, y)
            dt = time.perf_counter() - t0
            t_lanczos += dt
            thr = 2.0 / eta
            ckpt.append({"step": t, "eta": eta, "lam1": ev[0], "lam1_eta_2": ev[0] * eta / 2,
                         "n_unstable": int((ev >= 0.9 * thr).sum()),
                         "n_unstable_80": int((ev >= 0.8 * thr).sum()),
                         "n_unstable_97": int((ev >= 0.97 * thr).sum()),
                         "t_lanczos": dt, "n_hvp": nhvp,
                         **{f"ev{i}": v for i, v in enumerate(ev)}})
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
        t_train += time.perf_counter() - t0
        loss_log[t] = loss.item()
        traj[t] = torch.cat([p.detach().reshape(-1) for p in params]).numpy()
        if not np.isfinite(loss_log[t]):
            raise FloatingPointError(f"diverged at step {t}")
    np.savez_compressed(RES / f"log_{tag}.npz", loss=loss_log)
    ck = pd.DataFrame(ckpt)
    rows = []
    sm = smooth(loss_log)
    for a in range(0, steps - W + 1, S):
        rng = np.random.default_rng(1)
        t0 = time.perf_counter()
        pr = detrended_pr(traj[a:a + W])
        t_pr = time.perf_counter() - t0
        inside = ck[(ck.step >= a) & (ck.step < a + W)]
        row = {"start": a, "centre": a + W // 2, "traj_PR": pr, "t_PR": t_pr,
               "n_unstable": inside.n_unstable.median(),
               "n_unstable_80": inside.n_unstable_80.median(),
               "n_unstable_97": inside.n_unstable_97.median(),
               "lam1_eta_2": inside.lam1_eta_2.median(),
               "loss_level": float(np.median(loss_log[a:a + W]))}
        row.update(score_window(loss_log[a:a + W], rng))
        sm_scores = score_window(sm[a:a + W], np.random.default_rng(1))
        row.update({f"{k}_smooth": v for k, v in sm_scores.items()})
        rows.append(row)
    win = pd.DataFrame(rows)
    meta = {"P": P, "t_train": t_train, "t_lanczos": t_lanczos,
            "traj_MB": traj.nbytes / 2 ** 20, "log_KB": loss_log.astype(np.float32).nbytes / 1024,
            "final_loss": float(loss_log[-1]),
            "final_acc": float((net(X).argmax(1) == y).float().mean())}
    return win, ck, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "switch", "all"], default="all")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument("--switches", nargs="+",
                    default=["0.3:0.1", "0.3:0.05", "0.2:0.05", "0.1:0.3"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    RES.mkdir(parents=True, exist_ok=True)
    plan = []
    if args.stage in ("sweep", "all"):
        plan += [("sweep", e, e) for e in args.etas]
    if args.stage in ("switch", "all"):
        plan += [("switch", float(s.split(":")[0]), float(s.split(":")[1])) for s in args.switches]
    wins, cks, metas = [], [], []
    for stage, e0, e1 in plan:
        for seed in args.seeds:
            tag = f"{stage}_{e0:g}-{e1:g}_s{seed}"
            if stage == "switch" and e0 == e1 and (RES / f"log_sweep_{e0:g}-{e1:g}_s{seed}.npz").exists():
                continue            # constant-eta controls are the sweep runs
            t0 = time.perf_counter()
            try:
                win, ck, meta = run(e0, e1, seed, args.steps, args.every, tag)
            except FloatingPointError as err:
                print(tag, "DIVERGED", err, flush=True)
                metas.append({"tag": tag, "diverged": True})
                continue
            for df in (win, ck):
                df["stage"], df["eta0"], df["eta1"], df["seed"] = stage, e0, e1, seed
            wins.append(win); cks.append(ck)
            metas.append({"tag": tag, **meta})
            print(f"{tag:22s} {time.perf_counter()-t0:6.1f}s  loss {meta['final_loss']:.3g} "
                  f"acc {meta['final_acc']:.3f}  n_unst {ck.n_unstable.median():.0f} "
                  f"MG {win.MG.median():.2f} rel {np.median(win.MG/win.MG_surr):.2f}", flush=True)
            pd.concat(wins).to_csv(RES / f"windows_{args.stage}.csv", index=False)
            pd.concat(cks).to_csv(RES / f"checkpoints_{args.stage}.csv", index=False)
            json.dump(metas, open(RES / f"meta_{args.stage}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

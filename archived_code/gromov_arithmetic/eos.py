"""Full-batch descent at the edge of stability: does an undriven run ever recur?

The article validates a delay-embedding dimension estimator on six systems in which
recurrence is *manufactured* -- an external quasiperiodic drive with ``r`` independent
phases imposes an ``r``-torus.  Real training has no such drive, and the two grokking
settings the article examines are refused by its own diagnostics, one for being
stochastic and one for being a monotone transient.  That leaves the obvious question
unanswered: is the admissible regime ever occupied by something anyone trains?

The one candidate the literature offers is the edge of stability
(Cohen et al., arXiv:2103.00065).  Full-batch gradient descent at a fixed rate
``eta`` sharpens progressively until the top Hessian eigenvalue reaches ``2/eta`` and
then hovers there, and while it hovers the loss is *non-monotone on short scales and
descending on long ones*.  That is a deterministic system whose orbit is not a plain
transient, which is exactly the pair of conditions Sec. 3.3 asks for.

So this module trains the quadratic perceptron of ``gromov.py`` at a range of rates,
measures the sharpness along the way, and writes a log the estimator can read.

THE SAMPLING POINT, which is the reason this file exists rather than a re-analysis
of the logs already committed.  Edge-of-stability oscillation has a period of a few
optimiser steps -- it is the two-cycle of the unstable mode.  Every full-batch log in
this repository is written at a stride of 10 steps or more (``Config.log_every``
defaults to 10; the article's inventory lists 10, 49 and 50).  A stride of 10 does not
merely blur that oscillation, it aliases it away completely, and a trend-crossing
count computed on such a log is a count of what survived the aliasing.  Hence
``log_every=1`` here, and hence ``eos_probe.py`` re-reads each log at several strides
to show what the published protocol would have seen.

    python eos.py --lrs 1e5 3e5 1e6 2e6 --seeds 1 2 --steps 30000 --outdir ./results/eos

Sharpness is the top eigenvalue of the Hessian of the *full-batch training* loss --
the same objective the optimiser descends, so that ``eta * lam_max / 2`` is exactly
the stability ratio of the linearised dynamics and not an approximation to it.  It is
obtained by power iteration on Hessian-vector products, which needs no explicit
Hessian: at 145 500 parameters the matrix would be 2.1e10 entries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tasks
from gromov import Config, GromovMLP, build_dataset, grok_summary

TRAIN_COLUMNS = ("step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm")
SHARP_COLUMNS = ("step", "lam_max", "eta_lam_over_2", "power_iters", "power_rel_change")


# ---------------------------------------------------------------------------
# sharpness
# ---------------------------------------------------------------------------

def _flat(vs):
    return torch.cat([v.reshape(-1) for v in vs])


def _power(grads, params, shift, iters, tol, generator):
    """Dominant eigenvalue of ``H + shift*I`` by power iteration, signed.

    Returns ``(rayleigh, iters_used, last_relative_change)``.  The Rayleigh quotient is
    reported rather than the norm ratio, so the sign survives.
    """
    v = [torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator)
         for p in params]
    n = torch.linalg.vector_norm(_flat(v))
    v = [vi / n for vi in v]

    lam, prev, used, rel = float("nan"), None, 0, float("nan")
    for it in range(1, iters + 1):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        if shift:
            hv = [h + shift * vi for h, vi in zip(hv, v)]
        flat_hv, flat_v = _flat(hv), _flat(v)
        lam = float(torch.dot(flat_hv, flat_v))
        nrm = float(torch.linalg.vector_norm(flat_hv))
        used = it
        if nrm <= 0 or not math.isfinite(nrm):
            break
        v = [h / nrm for h in hv]
        if prev is not None and abs(lam) > 0:
            rel = abs(lam - prev) / abs(lam)
            if rel < tol:
                break
        prev = lam
    return lam, used, rel


def sharpness(model, x, target, iters=30, tol=1e-4, generator=None):
    """Largest *algebraic* Hessian eigenvalue of the full-batch MSE, via HVPs.

    Returns ``(lam_max, iters_used, last_relative_change)``.  The last two are written
    to the log because an unconverged power iteration *underestimates* lam_max, and an
    underestimate is precisely the direction that would manufacture a false "below the
    stability threshold" reading.  Reporting them lets that be checked rather than
    trusted.

    Plain power iteration converges to the eigenvalue of largest MAGNITUDE, which is
    not what stability depends on.  The condition GD must satisfy is
    ``eta < 2 / lam_max`` with ``lam_max`` the largest algebraic eigenvalue, and this
    Hessian is indefinite: on the smoke run the dominant eigenvalue was negative at
    three of twenty-six measurements early in training, when the most negative
    direction is steeper than the most positive one.  Returning that as "lam_max"
    would report a negative sharpness and a nonsensical stability ratio.  So when the
    dominant eigenvalue comes back negative the iteration is repeated on
    ``H + |lam|*I``, whose spectrum is non-negative and whose dominant eigenvalue is
    therefore the top of the original spectrum, and the shift is removed afterwards.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    loss = ((model(x) - target) ** 2).mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    lam, used, rel = _power(grads, params, 0.0, iters, tol, generator)
    if math.isfinite(lam) and lam < 0:
        shift = abs(lam)
        lam2, used2, rel2 = _power(grads, params, shift, iters, tol, generator)
        lam, used, rel = lam2 - shift, used + used2, rel2
    return lam, used, rel


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _evaluate(model, x, y, chunk=4096):
    """Mean-reduced MSE over all elements including the class axis, and argmax accuracy.

    Byte-identical in convention to ``gromov._evaluate`` so that a log written here is
    on the same scale as every other log in the repository: the initial loss is 1/p.
    """
    loss_sum, correct, n = 0.0, 0, x.shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            xb, yb = x[i:i + chunk], y[i:i + chunk]
            out = model(xb)
            target = torch.zeros_like(out)
            target[torch.arange(yb.shape[0], device=yb.device), yb] = 1.0
            loss_sum += float(((out - target) ** 2).sum())
            correct += int((out.argmax(dim=1) == yb).sum())
    return loss_sum / (n * model.p), correct / n


def run_one(key, lr, seed, steps, p=97, width=500, fraction=0.5, task="add",
            dtype="float32", device="auto", sharp_every=100, sharp_iters=30,
            outdir=Path("./results/eos"), verbose=True):
    """One full-batch run at rate ``lr``, logged at every optimiser step."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tdtype = torch.float32 if dtype == "float32" else torch.float64

    cfg = Config(key=key, task=task, p=p, width=width, fraction=fraction,
                 optimizer="gd", lr=lr, weight_decay=0.0, init_seed=seed,
                 max_steps=steps, batch_size=None, log_every=1, obs_every=1,
                 n_snapshots=0, dtype=dtype, device=device)
    data = build_dataset(cfg, tasks.get(task))
    to = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    x_tr, y_tr = to(data["x_train"], tdtype), to(data["y_train"], torch.long)
    x_va, y_va = to(data["x_val"], tdtype), to(data["y_val"], torch.long)

    gen = torch.Generator(device=device).manual_seed(seed)
    model = GromovMLP(p, width, 2, "quadratic", generator=gen, dtype=tdtype, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

    target_tr = torch.zeros(x_tr.shape[0], p, dtype=tdtype, device=device)
    target_tr[torch.arange(x_tr.shape[0], device=device), y_tr] = 1.0
    # Its own generator: the power-iteration start vector must not consume draws from
    # the initialisation stream, or turning sharpness measurement on would change the
    # trajectory it is supposed to be measuring.
    pgen = torch.Generator(device=device).manual_seed(seed + 7_000_000)

    train_rows, sharp_rows, t0 = [], [], time.time()
    diverged_at = None

    for step in range(steps + 1):
        tr_loss, tr_acc = _evaluate(model, x_tr, y_tr)
        va_loss, va_acc = _evaluate(model, x_va, y_va)
        wn = float(torch.sqrt((model.W1.detach() ** 2).sum() + (model.W2.detach() ** 2).sum()))
        wn /= math.sqrt(model.W1.numel() + model.W2.numel())
        train_rows.append(dict(step=step, train_loss=tr_loss, val_loss=va_loss,
                               train_acc=tr_acc, val_acc=va_acc, weight_norm=wn))

        if step % sharp_every == 0:
            lam, used, rel = sharpness(model, x_tr, target_tr, iters=sharp_iters,
                                       generator=pgen)
            sharp_rows.append(dict(step=step, lam_max=lam,
                                   eta_lam_over_2=lr * lam / 2.0,
                                   power_iters=used, power_rel_change=rel))
            if verbose and step % (sharp_every * 20) == 0:
                print(f"    step {step:>6d}  loss {tr_loss:.4e}  val_acc {va_acc:6.2%}  "
                      f"lam {lam:.4e}  eta*lam/2 {lr * lam / 2.0:.3f}  "
                      f"{time.time() - t0:5.0f}s", flush=True)

        if step == steps:
            break

        opt.zero_grad(set_to_none=True)
        loss = ((model(x_tr) - target_tr) ** 2).mean()
        loss.backward()
        opt.step()
        if not torch.isfinite(loss.detach()):
            diverged_at = step
            print(f"    diverged at step {step}", flush=True)
            break

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"{key}_train.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(TRAIN_COLUMNS))
        w.writeheader()
        w.writerows(train_rows)
    with (outdir / f"{key}_sharp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SHARP_COLUMNS))
        w.writeheader()
        w.writerows(sharp_rows)

    lams = [r["eta_lam_over_2"] for r in sharp_rows if math.isfinite(r["eta_lam_over_2"])]
    tail = lams[len(lams) // 2:] if lams else []
    # ``max_steps`` and not ``steps``: grok_summary already returns ``steps``, the last
    # step actually logged, and the two differ exactly when a run diverges early --
    # which is the case this campaign most needs to be able to see.
    meta = dict(key=key, lr=lr, seed=seed, max_steps=steps, p=p, width=width,
                fraction=fraction, task=task, dtype=dtype, device=str(device),
                optimizer="gd", weight_decay=0.0, batch="full", log_every=1,
                sharp_every=sharp_every, sharp_iters=sharp_iters,
                diverged_at=diverged_at, n_rows=len(train_rows),
                seconds=round(time.time() - t0, 1),
                eta_lam_over_2_max=max(lams) if lams else None,
                eta_lam_over_2_median_tail=float(np.median(tail)) if tail else None,
                **grok_summary(train_rows))
    (outdir / f"{key}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e4, 3e4, 1e5, 3e5, 1e6, 2e6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--task", default="add")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sharp-every", type=int, default=100)
    ap.add_argument("--sharp-iters", type=int, default=30)
    ap.add_argument("--outdir", default="./results/eos")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for lr in args.lrs:
        for seed in args.seeds:
            key = f"eos_lr{lr:g}_s{seed}" + ("_f64" if args.dtype == "float64" else "")
            if (outdir / f"{key}_meta.json").exists() and not args.force:
                print(f"[skip] {key} (exists; --force to redo)", flush=True)
                summary.append(json.loads((outdir / f"{key}_meta.json").read_text()))
                continue
            print(f"==> {key}: lr={lr:g} seed={seed} steps={args.steps} "
                  f"{args.dtype}", flush=True)
            summary.append(run_one(
                key, lr, seed, args.steps, p=args.p, width=args.width,
                fraction=args.fraction, task=args.task, dtype=args.dtype,
                device=args.device, sharp_every=args.sharp_every,
                sharp_iters=args.sharp_iters, outdir=outdir))

    if summary:
        cols = sorted({k for m in summary for k in m})
        with (outdir / "eos_runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {outdir / 'eos_runs.csv'} ({len(summary)} runs)")
        for m in summary:
            r = m.get("eta_lam_over_2_median_tail")
            print(f"  {m['key']:<22} eta*lam/2 tail median "
                  f"{('%.3f' % r) if r is not None else 'n/a':>8}  "
                  f"grok={str(m.get('t_grok')):>8}  "
                  f"{'DIVERGED' if m.get('diverged_at') is not None else ''}")


if __name__ == "__main__":
    main()

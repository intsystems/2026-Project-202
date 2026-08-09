"""Locate the critical training fraction at a given modulus.

Sec. 4 of arXiv:2301.02679 says ``alpha_c`` is a decreasing function of ``p`` but gives
a number for neither. That matters here for a practical reason: the polynomial runs at
p = 23 have to be given a training fraction above ``alpha_c``, or they produce
non-grokking logs for a reason that has nothing to do with the polynomial, and the
learnable and perturbed arms become indistinguishable for the wrong cause.

This sweeps ``alpha`` at fixed ``p``, at several learning rates, so that a failure to
grok cannot be blamed on the rate either.

    python alpha_sweep.py --p 23 --width 200 --steps 60000 \
        --alphas 0.5 0.7 0.9 --lrs 1e4 3e4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import tasks
from gromov import Config, grok_summary, train


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="add")
    ap.add_argument("--p", type=int, default=23)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e4, 3e4])
    ap.add_argument("--outdir", default="./results/sweeps")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{args.task} mod {args.p}, N={args.width}, {args.steps} steps, "
          f"full-batch GD, wd=0", flush=True)

    for alpha, lr in itertools.product(args.alphas, args.lrs):
        cfg = Config(task=args.task, p=args.p, width=args.width, fraction=alpha,
                     optimizer="gd", lr=lr, weight_decay=0.0, max_steps=args.steps,
                     batch_size=None, log_every=max(1, args.steps // 400),
                     obs_every=max(1, args.steps // 400) * 10, n_snapshots=0,
                     progress_every=0, device=args.device)
        t0 = time.time()
        train_rows, _, _ = train(cfg, tasks.get(args.task), verbose=False)
        s = grok_summary(train_rows)
        rows.append(dict(alpha=alpha, lr=lr, seconds=round(time.time() - t0, 1), **s))
        print(f"  alpha={alpha:<5} lr={lr:<9g} mem={str(s['t_memorise']):>7} "
              f"grok={str(s['t_grok']):>7} val={s['final_val_acc']:6.2%} "
              f"best={s['best_val_acc']:6.2%} |W|={s['final_weight_norm']:.2f}",
              flush=True)

    name = f"alpha_sweep_{args.task}_p{args.p}_N{args.width}"
    with (outdir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {outdir / f'{name}.csv'}")

    grokked = sorted({r["alpha"] for r in rows if r["t_grok"] is not None})
    failed = sorted({r["alpha"] for r in rows
                     if all(x["t_grok"] is None for x in rows if x["alpha"] == r["alpha"])})
    if grokked and failed:
        print(f"alpha_c is bracketed between {max(failed)} and {min(grokked)}: "
              f"no rate groks at {max(failed)}, some rate groks at {min(grokked)}.")


if __name__ == "__main__":
    raise SystemExit(main())

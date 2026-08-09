"""Calibrate the one hyperparameter arXiv:2301.02679 never states.

The paper reports full-batch GD with no weight decay and gives no learning rate --
not in the text, not in a caption, not in an appendix.  Its own two figures are
mutually inconsistent about the timescale (Fig. 0 groks near 20 000 steps, Fig. 3a
reports "time to grok" near 1 000 for the same optimiser), which can only mean the
two used different rates.  So the rate has to be chosen here, and chosen visibly.

The mean-field parametrisation of Eqs. (1)-(2) puts ``1/(D N)`` in front of the
output, so at initialisation the gradient is of order ``1/(p^3 N)`` -- about 4e-9 at
p=97, N=500.  Useful rates are therefore four to five orders of magnitude larger
than anything an Adam-shaped intuition suggests, which is why this sweep is wide.

    python lr_sweep.py --steps 20000 --lrs 1e3 3e3 1e4 3e4 1e5 3e5

Selection rule, applied in ``report.md``: among the rates that reach 100% train
accuracy without diverging, take the one whose validation accuracy is highest at
the end; ties go to the smaller rate, since a smaller rate resolves the transition
into more logged steps and the dimension analysis is downstream of that resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import tasks
from gromov import Config, grok_summary, train


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="add")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--optimizer", default="gd")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"sweep: {args.task} mod {args.p}, N={args.width}, alpha={args.fraction}, "
          f"{args.optimizer} wd={args.weight_decay}, {args.steps} steps", flush=True)

    for lr in args.lrs:
        cfg = Config(key=f"sweep_lr{lr:g}", task=args.task, p=args.p, width=args.width,
                     fraction=args.fraction, optimizer=args.optimizer, lr=lr,
                     weight_decay=args.weight_decay, max_steps=args.steps,
                     batch_size=None, log_every=max(1, args.steps // 500),
                     obs_every=max(1, args.steps // 500) * 10, n_snapshots=0,
                     progress_every=0, device=args.device)
        t0 = time.time()
        train_rows, _, _ = train(cfg, tasks.get(args.task), verbose=False)
        s = grok_summary(train_rows)
        finite = np.isfinite([r["train_loss"] for r in train_rows]).all()
        row = dict(lr=lr, diverged=(not finite) or train_rows[-1]["step"] < args.steps,
                   seconds=round(time.time() - t0, 1), **s)
        rows.append(row)
        print(f"  lr={lr:<10g} memorise={str(row['t_memorise']):>8} "
              f"grok={str(row['t_grok']):>8} train={row['final_train_acc']:6.2%} "
              f"val={row['final_val_acc']:6.2%} best_val={row['best_val_acc']:6.2%} "
              f"{'DIVERGED' if row['diverged'] else ''} ({row['seconds']:.0f}s)",
              flush=True)

    name = (f"lr_sweep_{args.task}_p{args.p}_N{args.width}"
            f"_a{args.fraction:g}_{args.optimizer}")
    with (outdir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if not r["diverged"] and r["final_train_acc"] >= 0.99]
    if usable:
        best = max(usable, key=lambda r: (round(r["final_val_acc"], 3), -r["lr"]))
        print(f"\nselected lr = {best['lr']:g} "
              f"(val {best['final_val_acc']:.2%}, grok at {best['t_grok']})")
        (outdir / f"{name}.json").write_text(json.dumps(best, indent=2))
    else:
        print("\nno rate both memorised and stayed finite -- widen --lrs")


if __name__ == "__main__":
    main()

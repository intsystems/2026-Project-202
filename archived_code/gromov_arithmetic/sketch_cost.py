"""What does the trajectory sketch cost, relative to the training it observes?

The article's practical claim for measuring the trajectory directly is that it is cheap:
a CountSketch to 1024 dimensions, recorded at every logged step, instead of storing the
whole parameter vector. Cheapness was asserted and never measured, so this measures it --
the same configuration trained twice, once with the observer attached and once without,
everything else identical.

The comparison is only meaningful if the two runs take the same optimisation path, which
``verify_rank_noninvasive.py`` establishes separately (zero maximum absolute parameter
difference over 400 steps). Here we check the cheaper invariant, that the two runs agree
on their logged loss to the last bit, and report the wall-clock ratio.

    python sketch_cost.py --steps 2000 --log-every 10

Reported: seconds per run, the overhead as a percentage, and the storage the sketch
replaces. Wall-clock on one machine is not a portable number, so the ratio is the number
to quote, and the absolute times are given so the ratio can be judged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import tasks
from gromov import Config, train
from rank import GromovRankProbe


def timed(cfg, fn, observer):
    t0 = time.perf_counter()
    rows, _, _ = train(cfg, fn, verbose=False, observer=observer)
    return time.perf_counter() - t0, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="add")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e5)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="./results/sketch_cost.json")
    args = ap.parse_args()

    cfg = Config(key="sketch_cost", task=args.task, p=args.p, width=args.width,
                 optimizer="gd", lr=args.lr, max_steps=args.steps,
                 batch_size=None, log_every=args.log_every,
                 obs_every=args.log_every * 10, n_snapshots=0, progress_every=0,
                 device=args.device)
    fn = tasks.get(args.task)

    bare, probed, ref_rows = [], [], None
    for i in range(args.repeats):
        # Alternate the order so a warm cache or a thermal ramp cannot favour one arm.
        order = ["bare", "probed"] if i % 2 == 0 else ["probed", "bare"]
        for which in order:
            obs = (GromovRankProbe(dim=args.dim, n_sketch=2, n_probe=256, seed=0)
                   if which == "probed" else None)
            secs, rows = timed(cfg, fn, obs)
            (probed if which == "probed" else bare).append(secs)
            if ref_rows is None:
                ref_rows = rows
            else:
                a = np.array([r["train_loss"] for r in ref_rows])
                b = np.array([r["train_loss"] for r in rows])
                assert a.shape == b.shape and np.array_equal(a, b), (
                    "the two arms did not take the same optimisation path; the timing "
                    "comparison is meaningless if they diverge")
            print(f"  [{i}] {which:<7} {secs:7.2f}s", flush=True)

    b_med, p_med = float(np.median(bare)), float(np.median(probed))
    n_logged = args.steps // args.log_every + 1
    n_params = cfg.n_params
    sketched_floats = n_logged * args.dim * 2 * 2      # two spaces, two hash families
    full_floats = n_logged * n_params
    out = dict(steps=args.steps, log_every=args.log_every, n_logged=n_logged,
               n_params=n_params, sketch_dim=args.dim,
               seconds_bare=b_med, seconds_probed=p_med,
               overhead_frac=(p_med - b_med) / b_med,
               sketched_float32_MB=sketched_floats * 4 / 1e6,
               full_float32_MB=full_floats * 4 / 1e6,
               storage_ratio=full_floats / sketched_floats,
               device=str(cfg.device), repeats=args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\nbare   {b_med:7.2f}s")
    print(f"probed {p_med:7.2f}s")
    print(f"overhead {100 * out['overhead_frac']:.1f} % over {n_logged} logged steps")
    print(f"storage  {out['sketched_float32_MB']:.2f} MB sketched against "
          f"{out['full_float32_MB']:.1f} MB full, a factor of {out['storage_ratio']:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

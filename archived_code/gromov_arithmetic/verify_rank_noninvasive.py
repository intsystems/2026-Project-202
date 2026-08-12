"""Check that attaching the rank probe leaves training bit-identical.

``../active_rank/verify_noninvasive.py`` exists because ``grok/tasks.py`` seeds one
global torch stream that the split, the initialisation and the mini-batch order all
continue, so a single stray draw from a probe changes the run and can destroy grokking.

This trainer is built differently -- the split comes from a NumPy generator, the
initialisation and the mini-batch order from two dedicated ``torch.Generator``s, and the
model has no stochastic layers -- so the probe's forward pass should not be able to
perturb anything. "Should not" is the reason to check: the claim is cheap to test and
expensive to be wrong about, and it is the claim that lets the sketched runs be compared
against the unsketched campaign logs.

    python verify_rank_noninvasive.py            # p=23, 400 steps, both spaces
"""

from __future__ import annotations

import argparse

import numpy as np

import tasks
from gromov import Config, train
from rank import GromovRankProbe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=23)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--task", default="add")
    args = ap.parse_args()

    cfg = Config(task=args.task, p=args.p, width=args.width, fraction=0.8,
                 optimizer="gd", lr=533.0, weight_decay=0.0, max_steps=args.steps,
                 batch_size=None, log_every=10, obs_every=100, n_snapshots=0,
                 progress_every=0, device="cpu")
    fn = tasks.get(args.task)

    plain, _, _ = train(cfg, fn, verbose=False)
    probed, _, _ = train(cfg, fn, verbose=False,
                         observer=GromovRankProbe(dim=256, n_probe=64))

    if len(plain) != len(probed):
        raise SystemExit(f"FAIL: {len(plain)} rows without the probe, {len(probed)} with")

    worst, worst_col = 0.0, None
    for a, b in zip(plain, probed):
        for col in a:
            d = abs(float(a[col]) - float(b[col]))
            if d > worst:
                worst, worst_col = d, col

    print(f"{cfg.summary()}")
    print(f"{len(plain)} logged rows x {len(plain[0])} columns compared")
    print(f"largest absolute difference: {worst:.3e}"
          + (f" (in {worst_col})" if worst_col else ""))
    if worst == 0.0:
        print("PASS -- bit-identical with and without the probe")
        return 0
    print("FAIL -- the probe perturbs training")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

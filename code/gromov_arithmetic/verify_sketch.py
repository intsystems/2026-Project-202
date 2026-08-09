"""Check that a participation ratio of ~1 is a fact about the trajectory, not the sketch.

The full-batch runs report ``PR ~ 1`` for every statistic at every step, which is also
exactly what a broken observer would report -- a sketch that collapsed every step to the
same vector, or a ``pr()`` that returned 1 on degenerate input, would look identical.
Since the conclusion in ``report.md`` rests on that number, it is checked here rather
than assumed.

Three checks, in increasing strength:

1. synthetic trajectories of *known* rank pushed through the same CountSketch and the
   same ``pr()`` must come back with that rank;
2. the sketch of a real run must not be degenerate (its rows must differ);
3. the participation ratio of a real run computed **exactly, on the full parameter
   vector**, must agree with the sketched one -- which removes the sketch from the
   argument entirely.

    python verify_sketch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_AR = Path(__file__).resolve().parent.parent / "active_rank"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))

import tasks                                        # noqa: E402
from analyze_rank import detrend, pr                # noqa: E402
from gromov import Config, train                    # noqa: E402
from rank import GromovRankProbe                    # noqa: E402
from rank_probe import _count_sketch                # noqa: E402

N_PARAMS, DIM, WINDOW = 145_500, 1024, 60


def main():
    rng = np.random.default_rng(0)
    idx, sign = _count_sketch(N_PARAMS, DIM, 12345)

    print("1. synthetic trajectories of known rank, through the same sketch")
    print(f"   {'true rank':>10}{'PR(raw)':>10}{'PR(sketch)':>12}")
    for r in (1, 2, 5, 10):
        traj = rng.standard_normal((WINDOW, r)) @ rng.standard_normal((r, N_PARAMS))
        z = np.zeros((WINDOW, DIM))
        for t in range(WINDOW):
            np.add.at(z[t], idx, sign * traj[t])
        print(f"   {r:>10}{pr(traj):>10.2f}{pr(z):>12.2f}")

    print("\n2, 3. a real full-batch run: exact parameters against the sketch")
    cfg = Config(task="add", p=97, width=500, fraction=0.5, optimizer="gd", lr=1e5,
                 weight_decay=0.0, max_steps=600, batch_size=None, log_every=10,
                 obs_every=100, n_snapshots=0, progress_every=0, device="cpu")
    probe = GromovRankProbe(dim=DIM, n_probe=64)
    thetas = []

    class Tee:
        """Record the exact parameter vector alongside whatever the probe records."""

        def on_start(self, model, x):
            probe.on_start(model, x)

        def on_log(self, step, model):
            probe.on_log(step, model)
            thetas.append(torch.cat([p.detach().reshape(-1)
                                     for p in model.parameters()]).clone().numpy())

    train(cfg, tasks.get("add"), verbose=False, observer=Tee())
    raw = np.asarray(thetas)
    z = np.asarray(probe._z)[:, 0, :]

    print(f"   {len(raw)} logged rows, {raw.shape[1]} parameters, sketch dim {z.shape[1]}")
    print(f"   sketch rows all identical? {np.allclose(z, z[0])}")
    print(f"   {'':<22}{'exact':>10}{'sketched':>12}")
    for name, a, b in (("positions", raw, z),
                       ("positions, detrended", detrend(raw), detrend(z)),
                       ("increments", np.diff(raw, axis=0), np.diff(z, axis=0))):
        print(f"   {name:<22}{pr(a):>10.3f}{pr(b):>12.3f}")

    print("\nThe exact column needs no sketch. If it reads ~1, the trajectory really is")
    print("one-dimensional over the window, and the observer is not the reason.")


if __name__ == "__main__":
    raise SystemExit(main())

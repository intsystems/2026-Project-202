"""Record the trajectory sketch that ``../active_rank/`` measures the rank of.

``../active_rank/report.md`` finds that at generalisation the training trajectory
collapses to essentially one dimension -- participation ratio 1.09-1.60 against a
plateau of 2.7-7.0 -- in function space first, then in parameter space, and well before
the trajectory's displacement collapses.  That result rests on six runs of the
transformer in ``../grokking_train/``, and its own report names the weak point:

    "The controls are not perfectly matched. `wd0` differs from `wd1` in more than
     whether it generalises."

Those controls are the *no-weight-decay* runs, so in that run set "generalises" and
"has weight decay" are confounded, and a statistic that responded to weight decay
rather than to generalisation would look identical.  The Gromov runs break the
confound in both directions at once:

* they grok with ``weight_decay = 0``, so a weight-decay detector must stay silent;
* the controls are matched on the *label function alone*.  ``(4 n1 + n2^2)^3`` and
  ``(4 n1 + n2^2)^3 + n1 n2`` share every hyperparameter, both reach 100% training
  accuracy, and only the first generalises (``../gromov_polynomials/report.md``).

So this file is the observer, and nothing more.  The measurement is
``../active_rank/analyze_rank.py``, run unchanged on the output: the ``.npz`` written
here carries exactly the keys it reads, and the driver writes ``<key>_train.csv``
beside it under the names it globs for.

The sketch itself is imported from ``../active_rank/rank_probe.py`` rather than
reimplemented -- a second CountSketch would be a second thing to keep correct.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_AR = Path(__file__).resolve().parent.parent / "active_rank"
if str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))

try:
    from rank_probe import _count_sketch, _normalize_logits
except ImportError as exc:  # pragma: no cover - depends on a sibling folder
    raise SystemExit(f"cannot import ../active_rank/rank_probe.py ({exc}); "
                     f"this observer reuses its sketch and needs it present")


class GromovRankProbe:
    """``active_rank``'s observer, adapted to this trainer's hook signature.

    The differences from ``RankProbe`` are confined to plumbing: this trainer passes
    ``(model, x_train)`` rather than a task object, and its models are plain modules
    with two parameters instead of ``grok``'s.  The recorded quantities, the sketch
    dimension and the output keys are identical, which is the point -- the analysis
    downstream must not be able to tell the two apart.

    No torch RNG save/restore, unlike ``RankProbe``: this trainer draws its
    mini-batches from a dedicated generator and the model has no stochastic layers,
    so a forward pass here cannot advance any stream the run depends on.
    ``verify_rank_noninvasive.py`` checks that claim rather than asserting it.
    """

    def __init__(self, dim=1024, n_sketch=2, n_probe=256, seed=0, progress_every=0):
        self.dim, self.n_sketch, self.n_probe = dim, n_sketch, n_probe
        self.seed, self.progress_every = seed, progress_every
        self._param_sketch, self._fn_sketch = [], []
        self._probe_inputs = None
        self._steps, self._z, self._zf = [], [], []
        self._scalars = {"param_norm": [], "param_step": [], "fn_step": []}
        self._prev_theta = self._prev_fn = None
        self._started = None
        self.n_params = None

    # -- hooks ---------------------------------------------------------------
    def on_start(self, model, x_train):
        rng = np.random.default_rng(self.seed)
        self._started = time.perf_counter()
        theta = self._flat(model)
        self.n_params = int(theta.numel())
        for _ in range(self.n_sketch):
            idx, sign = _count_sketch(self.n_params, self.dim, int(rng.integers(2 ** 31)))
            self._param_sketch.append((
                torch.from_numpy(idx).to(device=theta.device, dtype=torch.long),
                torch.from_numpy(sign).to(device=theta.device, dtype=theta.dtype)))
        n = min(self.n_probe, len(x_train))
        index = np.sort(rng.choice(len(x_train), size=n, replace=False))
        self._probe_inputs = x_train[torch.from_numpy(index).to(x_train.device)]
        self._fn_seeds = [int(rng.integers(2 ** 31)) for _ in range(self.n_sketch)]

    @torch.no_grad()
    def on_log(self, step, model):
        theta = self._flat(model)
        self._steps.append(int(step))
        self._z.append(self._apply(theta, self._param_sketch).cpu().numpy())
        self._scalars["param_norm"].append(float(torch.linalg.norm(theta)))
        self._scalars["param_step"].append(
            float("nan") if self._prev_theta is None
            else float(torch.linalg.norm(theta - self._prev_theta)))
        self._prev_theta = theta.clone()

        fn = _normalize_logits(model(self._probe_inputs)).reshape(-1)
        if not self._fn_sketch:
            for s in range(self.n_sketch):
                idx, sign = _count_sketch(int(fn.numel()), self.dim, self._fn_seeds[s])
                self._fn_sketch.append((
                    torch.from_numpy(idx).to(device=fn.device, dtype=torch.long),
                    torch.from_numpy(sign).to(device=fn.device, dtype=fn.dtype)))
        self._zf.append(self._apply(fn, self._fn_sketch).cpu().numpy())
        self._scalars["fn_step"].append(
            float("nan") if self._prev_fn is None
            else float(torch.linalg.norm(fn - self._prev_fn)))
        self._prev_fn = fn.clone()

        if self.progress_every and len(self._steps) % self.progress_every == 0:
            print(f"    [rank] {len(self._steps)} rows, step {step}, "
                  f"{time.perf_counter() - self._started:.0f}s", flush=True)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _flat(model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def _apply(self, vec, sketches):
        out = torch.zeros(len(sketches), self.dim, dtype=vec.dtype, device=vec.device)
        for s, (idx, sign) in enumerate(sketches):
            out[s].index_add_(0, idx, sign * vec)
        return out

    def save(self, path):
        """Write the keys ``../active_rank/analyze_rank.py`` reads, and only those."""
        np.savez_compressed(
            path,
            step=np.asarray(self._steps, dtype=np.int64),
            z=np.asarray(self._z, dtype=np.float64),          # (T, n_sketch, dim)
            zf=np.asarray(self._zf, dtype=np.float64),
            n_params=self.n_params, dim=self.dim, n_sketch=self.n_sketch,
            n_probe=self.n_probe, seed=self.seed,
            **{k: np.asarray(v, dtype=np.float64) for k, v in self._scalars.items()})
        return path

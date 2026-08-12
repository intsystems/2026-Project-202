"""Record enough of the parameter trajectory to measure how many directions it explores.

``active_dimension/`` established that the quantity behind "the number of actively
expressed degrees of freedom" is the rank of the trajectory covariance -- a property of a
point cloud -- and that a delay embedding of a 1-D log estimates something else entirely,
the dimension of an invariant set, which for a converging or noise-driven trajectory is
either 1 or undefined.  The wanted quantity is directly measurable from the parameters, so
this observer measures it directly and skips the 1-D detour.

The full parameter vector is ~2.3e5 doubles and there are ~2000 logged steps, so keeping
every checkpoint is 3.7 GB.  Instead each logged step is **sketched**: a CountSketch
(feature hashing with random signs) maps R^P to R^D once and for all, so the Gram matrix of
the T logged points -- and therefore the covariance spectrum, and therefore the rank and
the participation ratio -- is preserved to O(sqrt(log T / D)).  At D = 1024 and T = 2000
that is about 9 %.  Two independent sketches are recorded so the sketch error can be
measured rather than trusted.

Two spaces are sketched side by side, because they answer different questions:

``param``     the flattened parameter vector.  "How many directions is the optimiser
              moving in?"
``fn_train``  the centred, L2-normalised probe logits, flattened.  "How many directions is
              the computed *function* moving in?"  Normalisation matters for the same
              reason as in ``prediction_improved/probe.py``: the raw logit scale is
              mechanically coupled to weight decay, so a signal read off it cannot be
              distinguished from a detector of weight decay.

Nothing is decided here.  The sketch is written out and every statistic -- participation
ratio over sliding windows, detrended or not, of positions or of increments, at several
smoothing scales -- is computed offline in ``analyze_rank.py``, where it can be changed
without retraining.

**RNG discipline.** ``grok.tasks`` seeds one global torch stream that the train/val split,
the weight initialisation and the mini-batch order all continue, so a single extra draw
changes the run and can destroy grokking.  The hash, the signs and the probe indices come
from NumPy; every forward pass is wrapped in a save/restore of the torch RNG state.
``verify_noninvasive.py`` checks that the training log is bit-identical with and without
this observer.
"""

import time

import numpy as np
import torch

EPS = 1e-12


def _count_sketch(n_in, n_out, seed):
    """(bucket index, +/-1 sign) for a CountSketch R^n_in -> R^n_out."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_out, size=n_in)
    sign = rng.integers(0, 2, size=n_in) * 2.0 - 1.0
    return idx, sign


def _normalize_logits(logits):
    """Centre over classes and L2-normalise per example -- see the module docstring."""
    centred = logits - logits.mean(dim=-1, keepdim=True)
    return centred / (torch.linalg.norm(centred, dim=-1, keepdim=True) + EPS)


class RankProbe:
    """Observer for :func:`grok.loop.train`.  See the module docstring."""

    def __init__(self, dim=1024, n_sketch=2, n_probe=256, seed=0, source="train",
                 progress_every=0):
        self.dim = dim
        self.n_sketch = n_sketch
        self.n_probe = n_probe
        self.seed = seed
        self.source = source
        self.progress_every = progress_every

        self._param_sketch = []        # list of (idx tensor, sign tensor)
        self._fn_sketch = []
        self._probe_inputs = None
        self._steps = []
        self._z = []                   # per step: (n_sketch, dim) parameter sketch
        self._zf = []                  # per step: (n_sketch, dim) function sketch
        self._scalars = {"param_norm": [], "param_step": [], "fn_step": []}
        self._prev_theta = None
        self._prev_fn = None
        self._started = None
        self._log_index = 0
        self.n_params = None

    # -- hooks -------------------------------------------------------------
    def on_start(self, task, model, config, device):
        rng = np.random.default_rng(self.seed)          # never the torch global stream
        self._started = time.perf_counter()

        theta = self._flat_params(model)
        self.n_params = int(theta.numel())
        for s in range(self.n_sketch):
            idx, sign = _count_sketch(self.n_params, self.dim, int(rng.integers(2 ** 31)))
            self._param_sketch.append((
                torch.from_numpy(idx).to(device=theta.device, dtype=torch.long),
                torch.from_numpy(sign).to(device=theta.device, dtype=theta.dtype)))

        pool = {"train": task.X_train, "val": task.X_val}[self.source]
        n = min(self.n_probe, len(pool))
        index = np.sort(rng.choice(len(pool), size=n, replace=False))
        self._probe_inputs = pool[torch.from_numpy(index).to(pool.device)]
        self._fn_seeds = [int(rng.integers(2 ** 31)) for _ in range(self.n_sketch)]

    def on_log(self, step, model):
        was_training = model.training
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            model.eval()
            self._record(step, model)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            if was_training:
                model.train()

    def on_finish(self):
        pass

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _flat_params(model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def _apply(self, vec, sketches):
        out = torch.zeros(len(sketches), self.dim, dtype=vec.dtype, device=vec.device)
        for s, (idx, sign) in enumerate(sketches):
            out[s].index_add_(0, idx, sign * vec)
        return out

    @torch.no_grad()
    def _record(self, step, model):
        theta = self._flat_params(model)
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

        self._log_index += 1
        if self.progress_every and self._log_index % self.progress_every == 0:
            dt = time.perf_counter() - self._started
            print(f"    [rank_probe] {self._log_index} rows, step {step}, {dt:.0f}s",
                  flush=True)

    # -- output ------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path,
            step=np.asarray(self._steps, dtype=np.int64),
            z=np.asarray(self._z, dtype=np.float64),        # (T, n_sketch, dim)
            zf=np.asarray(self._zf, dtype=np.float64),
            n_params=self.n_params, dim=self.dim, n_sketch=self.n_sketch,
            source=self.source, n_probe=self.n_probe, seed=self.seed,
            **{k: np.asarray(v, dtype=np.float64) for k, v in self._scalars.items()})
        return path

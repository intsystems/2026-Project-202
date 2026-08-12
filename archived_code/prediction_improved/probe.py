"""Function-space observables: projections of normalized probe logits.

The observable proposed in ``method.md``. At each logged step the model is applied,
in ``eval`` mode, to a *fixed* probe set. Each example's logit vector is centered
over classes and L2-normalized, the whole matrix is flattened, and the result is
projected onto a few fixed random unit directions -- one scalar time series each.

Why normalized. The weight norm is mechanically coupled to weight decay, so a signal
read off it cannot be told apart from a detector of weight decay itself (the audit's
central confound). Centering removes the common logit shift and normalization removes
per-example scale, which is the mode weight decay drives; what is left moves only when
the *direction* of the model's outputs moves, i.e. when the computed function changes.

Two probe sets are recorded side by side, because the choice is a real experimental
fork and costs almost nothing to keep open:

``train``
    Inputs sampled from the training split. Fully validation-agnostic -- the headline
    claim rests on this one.
``val``
    Held-out *inputs* only; labels are never touched, so the criterion stays label-free,
    but the setting becomes transductive. On these finite algebraic tasks the
    generalizing circuit shows up most sharply on unseen inputs, so this is expected to
    carry the stronger signal; it is reported separately, never mixed into the above.

Recorded per logged step, per probe set: ``R`` projections plus ``velocity``, the median
per-example movement of the normalized logits since the previous logged step (the ``N0``
null model of ``method.md`` -- the simplest thing that detects function drift, which the
geometry layer has to beat to justify itself). The full normalized logit matrix is
snapshotted every ``snapshot_every`` logged steps so any other statistic -- a different
velocity horizon, PCA, per-example series -- can be computed offline.

**RNG discipline.** ``grok.tasks`` seeds the global torch RNG and the train/val split,
the weight initialisation and the mini-batch order all continue that one stream, so a
single extra draw changes the run and can destroy grokking. Nothing here draws from it:
probe indices and projection directions come from NumPy, and every forward pass is
wrapped in a save/restore of the torch RNG state so the guarantee holds even if a future
model adds dropout.
"""

import time

import numpy as np
import torch

EPS = 1e-12


class LogitProbe:
    """Observer for :func:`grok.loop.train`. See the module docstring."""

    def __init__(self, n_probe=256, n_projections=16, seed=0, snapshot_every=20,
                 sources=("train", "val"), progress_every=0):
        self.n_probe = n_probe
        self.n_projections = n_projections
        self.seed = seed
        self.snapshot_every = snapshot_every
        self.sources = tuple(sources)
        # Heartbeat every N logged rows: these runs are long and detached, and the
        # training loop writes nothing until it finishes, so without this a job in
        # flight is indistinguishable from a hung one.
        self.progress_every = progress_every
        self._started = None

        self._inputs = {}          # source -> (n_probe, n_ctx) int64 tensor
        self._directions = {}      # source -> (R, n_probe * C) tensor, built lazily
        self._direction_seeds = {}  # source -> seed for the above
        self._previous = {}        # source -> (n_probe, C) tensor, last normalized logits
        self._rows = {"step": []}
        self._snapshots = {name: [] for name in self.sources}
        self._snapshot_steps = []
        self._log_index = 0

    # -- hooks -------------------------------------------------------------

    def on_start(self, task, model, config, device):
        """Pin the probe inputs and the projection directions, before step 0."""
        rng = np.random.default_rng(self.seed)          # never the torch global stream
        pools = {"train": task.X_train, "val": task.X_val}

        unknown = [name for name in self.sources if name not in pools]
        if unknown:
            raise ValueError(f"unknown probe source(s) {unknown}. Known: {sorted(pools)}")

        self._started = time.perf_counter()
        for name in self.sources:
            pool = pools[name]
            if len(pool) < self.n_probe:
                raise ValueError(
                    f"{name} split has {len(pool)} examples, fewer than n_probe="
                    f"{self.n_probe}; lower n_probe or raise the split size"
                )
            index = np.sort(rng.choice(len(pool), size=self.n_probe, replace=False))
            self._inputs[name] = pool[torch.from_numpy(index).to(pool.device)]

            # The directions live in R^(n_probe * C), and C is the model's output
            # width -- `vocab_size`, not `num_classes`: these models emit one logit
            # for the "=" token too. Deriving it from an actual forward pass instead
            # of from the task keeps this correct for any head. Only the seed is
            # fixed here, so the directions stay reproducible and order-independent.
            self._direction_seeds[name] = int(rng.integers(2 ** 31))

            for r in range(self.n_projections):
                self._rows[f"{name}_p{r:02d}"] = []
            self._rows[f"{name}_velocity"] = []

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

    @torch.no_grad()
    def _record(self, step, model):
        take_snapshot = self._log_index % self.snapshot_every == 0
        self._rows["step"].append(step)

        for name in self.sources:
            normalized = _normalize_logits(model(self._inputs[name]))
            flat = normalized.reshape(-1)

            if name not in self._directions:
                self._directions[name] = _random_directions(
                    self.n_projections, flat.numel(), self._direction_seeds[name],
                    device=flat.device, dtype=flat.dtype,
                )

            projections = self._directions[name] @ flat
            for r, value in enumerate(projections.tolist()):
                self._rows[f"{name}_p{r:02d}"].append(value)

            previous = self._previous.get(name)
            velocity = (
                float("nan") if previous is None
                else float(torch.linalg.norm(normalized - previous, dim=1).median())
            )
            self._rows[f"{name}_velocity"].append(velocity)
            self._previous[name] = normalized.clone()

            if take_snapshot:
                self._snapshots[name].append(
                    normalized.detach().to("cpu", torch.float32).numpy()
                )

        if take_snapshot:
            self._snapshot_steps.append(step)
        self._log_index += 1

        if self.progress_every and self._log_index % self.progress_every == 0:
            elapsed = time.perf_counter() - (self._started or time.perf_counter())
            rate = self._log_index / elapsed if elapsed > 0 else float("nan")
            print(f"    step {step:>7}  rows={self._log_index}  "
                  f"{elapsed:6.0f}s  {rate:.1f} rows/s", flush=True)

    # -- output ------------------------------------------------------------

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self._rows)

    def save(self, csv_path, npz_path=None):
        """Write the per-step series to CSV and the snapshots to ``.npz``."""
        from pathlib import Path

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(csv_path, index=False)

        if npz_path is not None and self._snapshot_steps:
            npz_path = Path(npz_path)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                steps=np.asarray(self._snapshot_steps),
                **{name: np.stack(frames) for name, frames in self._snapshots.items()
                   if frames},
            )
        return csv_path


def _random_directions(count, width, seed, device, dtype):
    """``count`` unit vectors in ``R^width``, drawn from NumPy so the global torch
    RNG stream -- shared by the split, the init and the batch order -- is untouched."""
    directions = np.random.default_rng(seed).standard_normal((count, width))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return torch.from_numpy(directions).to(device=device, dtype=dtype)


def _normalize_logits(logits):
    """Center each row over classes, then scale it to unit L2 norm."""
    centered = logits - logits.mean(dim=1, keepdim=True)
    return centered / (centered.norm(dim=1, keepdim=True) + EPS)

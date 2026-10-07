"""The trajectory sketch, and the discipline that keeps it from changing the run.

Section 7 measures the trajectory itself rather than a one-dimensional log of it, so what
has to be recorded is the whole parameter vector at every logged step. Compressed by
``countsketch``, that is 4096 floats a step against 226,816, a factor of 55, and the
participation ratio the article reads survives the compression (appendix I).

Two spaces are recorded, because they answer different questions:

``parameters``  the flat parameter vector: how many directions is the optimiser moving in?
``function``    the probe logits on a fixed subset of the training split, centred over
                classes and normalised to unit length: how many directions is the computed
                *function* moving in? The normalisation is not cosmetic. Raw logit scale is
                mechanically coupled to weight decay, so a signal read off it could not be
                told apart from a measurement of weight decay.

**Why the observer is safe to attach.** The task module seeds one global torch stream that
the train/validation split, the initial weights and the mini-batch order all continue. One
stray draw changes the initial weights, and initial weights decide whether these runs
generalise at all. So the hashes and the probe subset come from NumPy, and every probe
forward pass runs inside :func:`preserve_torch_rng`, which restores the CPU and CUDA states
afterwards. ``tests/test_sketch.py`` trains the same configuration with and without the
probe and requires every logged column to be bit-identical; the archived check ran on the
CPU only, so the test here exercises the CUDA branch as well when a GPU is present.

**One departure from the archived probe.** The archived code drew the parameter hash seeds,
then the probe subset, then the function hash seeds, all from one generator, so changing
``dim`` moved which examples were probed. Here the hashes come from the sketch's own
generator and the probe subset from a separate one. The hashes therefore differ from the
archived ``.npz`` files, which are regenerable and untracked; nothing downstream depends on
a particular hash, since every statistic is averaged over the independent families and
their disagreement is what appendix I reports as the error.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .countsketch import CountSketch

EPS = 1e-12


@contextlib.contextmanager
def preserve_torch_rng(device: Any = None):
    """Save and restore the CPU and CUDA torch RNG state around a probe forward pass.

    This is what makes the sketch non-invasive. Without it, any draw taken inside the
    observer -- now or after someone adds a dropout layer or a randomised kernel -- would
    advance the same stream the mini-batch order comes from, and the observed run would
    stop being the run everyone else gets.

    ``device`` names the stream the block could perturb. The CUDA state is saved only when
    a CUDA device is in play: reading it costs a synchronisation, and a CPU forward pass
    cannot touch it. The availability check below asks whether a CUDA generator exists to
    be saved; it is not a device choice, which belongs to ``actdim.runtime.device``.
    """
    import torch

    cpu_state = torch.get_rng_state()
    on_cuda = str(device).startswith("cuda") if device is not None else True
    cuda_state = (torch.cuda.get_rng_state_all()
                  if on_cuda and torch.cuda.is_available() else None)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def normalise_logits(logits: Any) -> Any:
    """Centre each example over classes and scale it to unit length.

    Both halves matter. Centring removes the per-example offset that softmax ignores;
    normalising removes the overall scale, which weight decay drives directly.
    """
    centred = logits - logits.mean(dim=-1, keepdim=True)
    import torch

    return centred / (torch.linalg.norm(centred, dim=-1, keepdim=True) + EPS)


class TrajectorySketch:
    """CountSketch of the parameter and function vectors at every logged step.

    The parameter sketch is sized on construction, since the parameter count is known
    before training starts. The function sketch is sized on its first call, because the
    probe subset and the class count are not known until the first forward pass; its hash
    seeds are drawn on construction all the same, so the sketch is fully determined by
    ``seed`` however late it is built.
    """

    def __init__(self, n_params: int, dim: int = 1024, n_sketch: int = 2, seed: int = 0):
        self.n_params = int(n_params)
        self.dim = int(dim)
        self.n_sketch = int(n_sketch)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self._param_seeds = [int(rng.integers(2 ** 31)) for _ in range(self.n_sketch)]
        self._function_seeds = [int(rng.integers(2 ** 31)) for _ in range(self.n_sketch)]
        self._parameters = CountSketch(self.n_params, self.dim, self.n_sketch,
                                       seeds=self._param_seeds)
        self._function: Optional[CountSketch] = None

    @property
    def n_function(self) -> Optional[int]:
        """Width of the flattened probe-logit vector, once it is known."""
        return None if self._function is None else self._function.n_in

    def sketch_parameters(self, flat: Any) -> np.ndarray:
        """Sketch the flat parameter vector. Returns ``(n_sketch, dim)``."""
        return self._parameters.apply(flat).detach().cpu().numpy()

    def sketch_function(self, logits: Any) -> np.ndarray:
        """Sketch the probe logits, centred and L2-normalised first.

        ``logits`` is ``(n_probe, n_classes)``. The normalised rows are flattened into one
        vector, so the sketch sees the whole function on the probe subset rather than one
        example at a time.
        """
        flat = normalise_logits(logits).reshape(-1)
        if self._function is None:
            self._function = CountSketch(int(flat.numel()), self.dim, self.n_sketch,
                                         seeds=self._function_seeds)
        return self._function.apply(flat).detach().cpu().numpy()

    def metadata(self) -> Dict[str, Any]:
        """What the sketch was built with, to be stored beside it."""
        return {"n_params": self.n_params, "dim": self.dim, "n_sketch": self.n_sketch,
                "seed": self.seed}


class TrajectoryRecorder:
    """Observer for a training loop: sketches both spaces at every logged step.

    Implements the three optional hooks the loops offer -- ``on_start``, ``on_log``,
    ``on_finish`` -- and accumulates in memory. It writes nothing: :meth:`arrays` returns
    what to store, and the caller decides where that goes.
    """

    def __init__(self, dim: int = 1024, n_sketch: int = 2, n_probe: int = 256, seed: int = 0,
                 source: str = "train", progress_every: int = 0):
        self.dim = int(dim)
        self.n_sketch = int(n_sketch)
        self.n_probe = int(n_probe)
        self.seed = int(seed)
        self.source = source
        self.progress_every = int(progress_every)

        self.sketch: Optional[TrajectorySketch] = None
        self.n_params: Optional[int] = None
        self._device: Any = None
        self._probe_inputs: Any = None
        self._steps: List[int] = []
        self._z: List[np.ndarray] = []
        self._zf: List[np.ndarray] = []
        self._scalars: Dict[str, List[float]] = {"param_norm": [], "param_step": [],
                                                 "fn_step": []}
        self._previous_theta: Any = None
        self._previous_fn: Any = None
        self._started: Optional[float] = None
        self._rows = 0

    # -- hooks -----------------------------------------------------------------

    def on_start(self, task: Any, model: Any, config: Any, device: Any) -> None:
        import torch

        self._started = time.perf_counter()
        self._device = device
        theta = self._flat_parameters(model)
        self.n_params = int(theta.numel())
        self.sketch = TrajectorySketch(self.n_params, dim=self.dim, n_sketch=self.n_sketch,
                                       seed=self.seed)

        # A stream of its own for the probe subset, so that changing the sketch width does
        # not silently change which examples the function is read on.
        rng = np.random.default_rng([self.seed, 1])
        pool = {"train": task.X_train, "val": task.X_val}[self.source]
        size = min(self.n_probe, len(pool))
        index = np.sort(rng.choice(len(pool), size=size, replace=False))
        self._probe_inputs = pool[torch.from_numpy(index).to(pool.device)]

    def on_log(self, step: int, model: Any) -> None:
        was_training = model.training
        with preserve_torch_rng(self._device):
            try:
                model.eval()
                self._record(step, model)
            finally:
                if was_training:
                    model.train()

    def on_finish(self) -> None:
        pass

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _flat_parameters(model: Any) -> Any:
        import torch

        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def _record(self, step: int, model: Any) -> None:
        import torch

        with torch.no_grad():
            theta = self._flat_parameters(model)
            self._steps.append(int(step))
            self._z.append(self.sketch.sketch_parameters(theta))
            self._scalars["param_norm"].append(float(torch.linalg.norm(theta)))
            self._scalars["param_step"].append(
                float("nan") if self._previous_theta is None
                else float(torch.linalg.norm(theta - self._previous_theta)))
            self._previous_theta = theta.clone()

            logits = model(self._probe_inputs)
            self._zf.append(self.sketch.sketch_function(logits))
            fn = normalise_logits(logits).reshape(-1)
            self._scalars["fn_step"].append(
                float("nan") if self._previous_fn is None
                else float(torch.linalg.norm(fn - self._previous_fn)))
            self._previous_fn = fn.clone()

        self._rows += 1
        if self.progress_every and self._rows % self.progress_every == 0:
            elapsed = time.perf_counter() - (self._started or time.perf_counter())
            print(f"    [sketch] {self._rows} rows, step {step}, {elapsed:.0f}s", flush=True)

    # -- output ----------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """The sketch's own metadata plus what the recorder chose."""
        base = dict(self.sketch.metadata()) if self.sketch is not None else {
            "n_params": self.n_params, "dim": self.dim, "n_sketch": self.n_sketch,
            "seed": self.seed}
        base.update({"source": self.source, "n_probe": self.n_probe,
                     "n_function": (self.sketch.n_function if self.sketch else None)})
        return base

    def arrays(self) -> Dict[str, Any]:
        """Everything to store, ready for ``np.savez_compressed`` or ``ctx.store.array``.

        ``z`` and ``zf`` are ``(T, n_sketch, dim)``. The metadata travels in the same file
        as zero-dimensional arrays, so a stored sketch can always say how wide it is and
        what it was taken of without a second file to lose.
        """
        payload: Dict[str, Any] = {
            "step": np.asarray(self._steps, dtype=np.int64),
            "z": np.asarray(self._z, dtype=np.float64),
            "zf": np.asarray(self._zf, dtype=np.float64),
        }
        payload.update({k: np.asarray(v, dtype=np.float64) for k, v in self._scalars.items()})
        for key, value in self.metadata().items():
            payload[key] = np.asarray(value if value is not None else -1)
        return payload

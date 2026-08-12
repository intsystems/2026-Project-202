"""The transformer training loop of appendix O, and the configuration that describes it.

One entry point, :func:`train`, driven entirely by a :class:`TransformerConfig`. Every row
of the log is one *logged* optimisation step, and ``log_every`` sets the sampling rate of
everything read off it afterwards, so it must stay constant within a run.

What the loop must not be talked out of:

*Mini-batches of 256.* The source configuration is full batch. The departure is deliberate
and is what places these runs in the stochastic regime the article's diagnostics assign
them to; see appendix O.

*Double precision.* On a T4, float64 runs at a thirty-second of the float32 rate, which is
why a run takes minutes. It is not an optimisation opportunity: the milestones move.

*The double optimiser step.* ``double_step`` calls ``optimizer.step()`` twice per batch with
no intervening ``zero_grad``, so the same gradient is applied twice and the AdamW moments
advance twice. This reproduces a duplicated update in earlier released ``S_5`` logs.
Appendix O states it, and the effective learning rate and decay of those two runs are about
double their nominal values because of it. It stays behind a flag, set on ``s5_wd1`` and
``s5_wd0`` and nowhere else, and the gradient probe is read *between* the two steps, which
is where the published columns were read.

**Two ``mod_wd1`` logs exist and they are not the same series.** The re-trained log follows
the earlier canonical one to 1e-14 for 198 rows and then diverges through float64 rounding,
ending 2.79 apart on the parameter norm. Its generalisation step is 13,700 where the
canonical log's is 13,810. Appendix O and section 7.2 use the re-trained log, at 13,700.
See ``actdim.training.runs`` for the full statement.
"""
from __future__ import annotations

import contextlib
import dataclasses
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..runtime import device as device_mod

OBSERVABLES = (
    "step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm",
    "grad_norm", "embed_grad_norm", "grad_cosine",
)
"""Everything the loop knows how to record, one CSV column each."""

GRAD_OBSERVABLES = ("grad_norm", "embed_grad_norm", "grad_cosine")
"""Columns that need the gradient probe, which costs a flatten of every gradient."""

BASE_COLUMNS = ("step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm")
"""The modular-addition column set."""

FULL_COLUMNS = BASE_COLUMNS + GRAD_OBSERVABLES
"""The ``S_5`` column set: the gradient diagnostics on top of the base ones."""

DTYPES = ("float32", "float64")
"""The dtypes a run may use. Appendix O's runs are all the second one."""


@dataclass(frozen=True)
class TransformerConfig:
    """One transformer training run. The defaults are the appendix O configuration."""

    key: str = "custom"
    description: str = ""
    provenance: str = ""
    """How the archived tree produced this run, for ``docs/experiments.md``."""

    # -- data ------------------------------------------------------------------
    task: str = "modular_addition"
    p: int = 113
    n: int = 5
    fraction: float = 0.3
    max_pairs: Optional[int] = None

    # -- model -----------------------------------------------------------------
    d_model: int = 128
    d_mlp: int = 512
    d_head: int = 32
    num_heads: int = 4
    high_variance_init: bool = False

    # -- optimisation ----------------------------------------------------------
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.98)
    momentum: float = 0.0
    max_steps: int = 20000
    batch_size: Optional[int] = 256      # None -> full batch
    val_batch_size: Optional[int] = 512  # None -> the whole validation split

    # -- logging ---------------------------------------------------------------
    log_every: int = 10
    val_every: Optional[int] = None
    """How often to evaluate the validation split, in optimisation steps.

    ``None`` means every logged step, which is what the published runs did. The cheap
    series and the validation pass are decoupled because the pass is the expensive half:
    ``log_every=1, val_every=10`` reconstructs the loss and the parameter norm at every
    step while evaluating at the old rate. Rows without a fresh evaluation carry NaN.
    """
    columns: Tuple[str, ...] = BASE_COLUMNS

    # -- reproducibility -------------------------------------------------------
    seed: Optional[int] = 42
    init_seed: Optional[int] = None
    """Reseed the global torch generator between the split and the initialisation.

    ``seed`` alone cannot separate the two things it controls: the task seeds the stream
    and draws the split from it, and the initialisation continues that same stream, so
    moving ``seed`` moves which examples are in the training set *and* what the initial
    weights are, together. ``init_seed`` restarts the stream between the two, so a run can
    be varied along one axis at a time. ``None`` leaves the stream alone.
    """
    dtype: str = "float64"
    device: str = "auto"

    # -- the faithfulness switch -----------------------------------------------
    double_step: bool = False
    """Call ``optimizer.step()`` twice per batch, with no ``zero_grad`` between them.

    A duplicated update in the notebooks that produced the released ``S_5`` logs: the same
    gradient is applied twice and the optimiser state advances twice, so the run sees about
    double the nominal learning rate and weight decay. Appendix O states this, and the
    regime those two runs are assigned to depends on it. Kept because the article's numbers
    cannot be reproduced without it; leave it off for anything new.
    """

    def __post_init__(self) -> None:
        unknown = [c for c in self.columns if c not in OBSERVABLES]
        if unknown:
            raise ValueError(f"unknown log column(s) {unknown}. Known: {list(OBSERVABLES)}")
        for required in ("step", "train_acc", "val_acc"):
            if required not in self.columns:
                raise ValueError(
                    f"'{required}' is required: every downstream reader indexes on step and "
                    f"locates memorisation and generalisation from the two accuracies")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.val_every is not None:
            if self.val_every < 1:
                raise ValueError(f"val_every must be >= 1, got {self.val_every}")
            if self.val_every % self.log_every:
                raise ValueError(
                    f"val_every ({self.val_every}) must be a multiple of log_every "
                    f"({self.log_every}), or validation never lands on a logged row")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must lie in (0, 1), got {self.fraction}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be float32 or float64, got '{self.dtype}'")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(f"unknown optimizer '{self.optimizer}'. Available: adamw, sgd")

    # -- derived ---------------------------------------------------------------

    @property
    def needs_grad_probe(self) -> bool:
        return any(c in self.columns for c in GRAD_OBSERVABLES)

    @property
    def expected_rows(self) -> int:
        return (self.max_steps + self.log_every - 1) // self.log_every

    def replace(self, **overrides: Any) -> "TransformerConfig":
        """A copy with fields overridden, each coerced to its declared type."""
        return replace(self, **{k: coerce(k, v) for k, v in overrides.items()})

    def to_dict(self) -> Dict[str, Any]:
        """The resolved configuration, for the provenance record.

        The archived tree never stored this beside a run's outputs, so a sketch could not
        be traced back to the overrides that produced it. Everything that reaches disk now
        goes through here.
        """
        out = dataclasses.asdict(self)
        out["betas"] = list(self.betas)
        out["columns"] = list(self.columns)
        return out

    def summary(self) -> str:
        batch = "full batch" if self.batch_size is None else f"batch {self.batch_size}"
        return (f"{self.task} / adamw lr={self.lr} wd={self.weight_decay} / {batch} / "
                f"{self.max_steps} steps, log every {self.log_every}"
                + (" / double step" if self.double_step else ""))


# -- override coercion ---------------------------------------------------------
# The command line and the experiment options hand over strings. One coercion table, so
# that ``--set batch_size=none`` means full batch everywhere and nowhere means "none".

_FIELDS = {f.name: f for f in fields(TransformerConfig)}


def _to_bool(raw: Any) -> bool:
    lowered = str(raw).strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got '{raw}'")


def _optional(cast):
    def inner(raw):
        if raw is None or str(raw).strip().lower() in ("none", "null", ""):
            return None
        return cast(raw)
    return inner


def _tuple_of(cast):
    def inner(raw):
        if isinstance(raw, (list, tuple)):
            return tuple(cast(item) for item in raw)
        return tuple(cast(item) for item in str(raw).split(",") if item.strip())
    return inner


_CASTS = {
    "max_pairs": _optional(int), "batch_size": _optional(int),
    "val_batch_size": _optional(int), "seed": _optional(int), "init_seed": _optional(int),
    "val_every": _optional(int), "betas": _tuple_of(float),
    "columns": _tuple_of(lambda s: str(s).strip()),
    "double_step": _to_bool, "high_variance_init": _to_bool,
}


def coerce(name: str, raw: Any) -> Any:
    """Cast one override to the declared type of ``TransformerConfig.<name>``."""
    if name not in _FIELDS:
        raise KeyError(f"unknown config field '{name}'. Known: {sorted(_FIELDS)}")
    if name in _CASTS:
        return _CASTS[name](raw)
    default = _FIELDS[name].default
    if isinstance(raw, type(default)) and not isinstance(default, str):
        return raw
    return type(default)(raw)


# -- the observables -----------------------------------------------------------


def weight_norm(model: Any) -> float:
    """``||w||_2`` over every parameter.

    Accumulated in Python floats before the square root, as the original did, so the column
    stays comparable digit for digit with the published logs.
    """
    return float(np.sqrt(sum(p.detach().pow(2).sum().item() for p in model.parameters())))


def accuracy(logits: Any, targets: Any) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean())


class GradientProbe:
    """Gradient norm, embedding gradient norm, and the step-to-step cosine.

    Updated after ``loss.backward()``. On a ``double_step`` run it is read between the two
    optimiser steps, which is where the published ``S_5`` columns were read; the cosine is
    0.0 on the first step, as in the source notebooks.
    """

    KEYS = GRAD_OBSERVABLES

    def __init__(self) -> None:
        self._previous: Any = None
        self.values: Dict[str, float] = dict.fromkeys(self.KEYS, 0.0)

    def update(self, model: Any) -> Dict[str, float]:
        import torch

        with torch.no_grad():
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                return self.values
            flat = torch.cat([g.reshape(-1) for g in grads])
            embed = getattr(model, "embedding_weight", None)
            embed_grad = None if embed is None else embed.grad
            self.values = {
                "grad_norm": float(torch.sqrt(sum(g.pow(2).sum() for g in grads))),
                "embed_grad_norm": 0.0 if embed_grad is None else float(embed_grad.norm(2)),
                "grad_cosine": 0.0 if self._previous is None else float(
                    torch.nn.functional.cosine_similarity(
                        flat.unsqueeze(0), self._previous.unsqueeze(0))),
            }
            self._previous = flat.clone()
        return self.values


# -- the loop ------------------------------------------------------------------


@dataclass
class TrainingRun:
    """What one run produced. Nothing here has been written anywhere."""

    log: pd.DataFrame
    config: Dict[str, Any]
    device: str
    seconds: float
    t_mem: Optional[int] = None
    t_gen: Optional[int] = None
    sketch: Any = None
    paths: Dict[str, Path] = field(default_factory=dict)

    def milestones(self) -> Tuple[Optional[int], Optional[int]]:
        return self.t_mem, self.t_gen


@contextlib.contextmanager
def default_dtype(name: str):
    """Set the global default dtype for the run, as the source notebooks did globally."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype({"float32": torch.float32, "float64": torch.float64}[name])
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_optimizer(config: TransformerConfig, model: Any) -> Any:
    import torch.optim as optim

    if config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay, betas=tuple(config.betas))
    return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                     weight_decay=config.weight_decay)


def _batches(task: Any, config: TransformerConfig):
    """An endless stream of training batches, or the whole split if ``batch_size`` is None."""
    from torch.utils.data import DataLoader, TensorDataset

    # The two cases are spelled out. The archived version put the full-batch loop first and
    # let the loader construction below sit unreachable behind it, which was correct only
    # for as long as nobody put a `break` in the loop above.
    if config.batch_size is None:
        while True:
            yield task.X_train, task.Y_train
    else:
        loader = DataLoader(
            TensorDataset(task.X_train, task.Y_train),
            batch_size=min(config.batch_size, len(task.X_train)),
            shuffle=True,
        )
        while True:
            for batch in loader:
                yield batch


def train(config: TransformerConfig, outdir: Optional[Any] = None, progress: bool = False,
          overwrite: bool = False, observer: Any = None, batch_hook: Any = None) -> TrainingRun:
    """Run ``config`` and return the log, the resolved configuration and the milestones.

    ``outdir`` is the only thing this function writes to, and it comes from the caller: an
    experiment module decides where results live, never this module. ``None`` keeps
    everything in memory, which is what the tests use.

    ``observer`` is a side-channel recorder -- in practice
    ``actdim.sketch.probe.TrajectoryRecorder``. It is offered three optional hooks:
    ``on_start(task, model, config, device)``, ``on_log(step, model)`` and ``on_finish()``.
    It must not mutate the model or draw from the global torch generator; the split, the
    initial weights and the mini-batch order share that one stream, so one extra draw
    changes the run. ``tests/test_sketch.py`` enforces it.

    ``batch_hook(step, x, y) -> (x, y)`` may transform each batch before the optimiser sees
    it. Unlike ``observer`` this is *meant* to change the run: it exists to inject a known
    driver so that a method claiming to recover drivers can be tested against ground truth.
    """
    import torch
    import torch.nn as nn

    from ..models import transformer as model_mod
    from ..tasks import modular

    resolved_device = device_mod.resolve(config.device)
    torch_device = torch.device(resolved_device)

    paths: Dict[str, Path] = {}
    if outdir is not None:
        outdir = Path(outdir)
        paths["log"] = outdir / f"{config.key}_train.csv"
        if paths["log"].exists() and not overwrite:
            raise FileExistsError(
                f"{paths['log']} already exists (pass overwrite=True). These runs cost "
                f"minutes to hours; clobbering one silently is expensive.")

    columns = list(config.columns)
    logs: Dict[str, list] = {name: [] for name in columns}
    started = time.perf_counter()

    with default_dtype(config.dtype):
        task = modular.from_config(config, torch_device)
        if config.init_seed is not None:
            # Restart the stream between the split and the initialisation, so the two can
            # be varied one at a time. A no-op unless the field is set.
            torch.manual_seed(config.init_seed)
        model = model_mod.build(config, task.vocab_size, n_ctx=task.n_ctx).to(torch_device)
        optimizer = build_optimizer(config, model)
        criterion = nn.CrossEntropyLoss()
        probe = GradientProbe() if config.needs_grad_probe else None

        val_x = task.X_val if config.val_batch_size is None else task.X_val[:config.val_batch_size]
        val_y = task.Y_val if config.val_batch_size is None else task.Y_val[:config.val_batch_size]
        if len(val_x) == 0:
            raise ValueError("empty validation split -- lower `fraction`")

        if observer is not None and hasattr(observer, "on_start"):
            observer.on_start(task=task, model=model, config=config, device=torch_device)

        val_every = config.val_every or config.log_every
        stream = _batches(task, config)

        for step in range(config.max_steps):
            batch_x, batch_y = next(stream)
            if batch_hook is not None:
                batch_x, batch_y = batch_hook(step, batch_x, batch_y)

            model.train()
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            if probe is not None:
                probe.update(model)          # read between the two steps -- see double_step
            if config.double_step:
                optimizer.step()             # the duplicated update of appendix O

            if step % config.log_every == 0:
                row = _observe(model, criterion, columns, step, loss, logits, batch_y,
                               val_x, val_y, probe, evaluate_val=(step % val_every == 0))
                for name in columns:
                    logs[name].append(row[name])
                if observer is not None and hasattr(observer, "on_log"):
                    observer.on_log(step, model)      # the model is already in eval mode
                if progress and step % (config.log_every * 50) == 0:
                    print(f"[{config.key}] step {step}/{config.max_steps} "
                          f"loss={row.get('train_loss', float('nan')):.3f} "
                          f"val_acc={row.get('val_acc', float('nan')):.3f}", flush=True)

        if observer is not None and hasattr(observer, "on_finish"):
            observer.on_finish()

    frame = pd.DataFrame(logs, columns=columns)
    t_mem, t_gen = _milestones(frame)
    result = TrainingRun(log=frame, config=config.to_dict(), device=resolved_device,
                         seconds=time.perf_counter() - started, t_mem=t_mem, t_gen=t_gen,
                         sketch=observer, paths=paths)

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths["log"], index=False)
        if observer is not None and hasattr(observer, "arrays"):
            paths["sketch"] = outdir / f"{config.key}_sketch.npz"
            np.savez_compressed(paths["sketch"], **observer.arrays())
    return result


def _observe(model: Any, criterion: Any, columns: Sequence[str], step: int, loss: Any,
             logits: Any, targets: Any, val_x: Any, val_y: Any, probe: Any,
             evaluate_val: bool = True) -> Dict[str, Any]:
    """One log row. Only the requested columns are computed.

    ``evaluate_val=False`` skips the validation pass and writes NaN. The pass runs under
    ``eval()`` and consumes no randomness, so skipping it cannot perturb training.
    """
    import torch

    with torch.no_grad():
        model.eval()
        row: Dict[str, Any] = {"step": step}

        if "train_loss" in columns:
            row["train_loss"] = loss.item()
        if "train_acc" in columns:
            row["train_acc"] = accuracy(logits, targets)

        if evaluate_val:
            val_logits = model(val_x)
            row["val_acc"] = accuracy(val_logits, val_y)
            if "val_loss" in columns:
                row["val_loss"] = criterion(val_logits, val_y).item()
        else:
            row["val_acc"] = float("nan")
            if "val_loss" in columns:
                row["val_loss"] = float("nan")

        if "weight_norm" in columns:
            row["weight_norm"] = weight_norm(model)
        if probe is not None:
            row.update({k: v for k, v in probe.values.items() if k in columns})
    return row


def _milestones(frame: pd.DataFrame, threshold: float = 0.95) -> Tuple[Optional[int], Optional[int]]:
    """Memorisation and generalisation, as appendix O defines them.

    The first logged step at which the training and the validation accuracy respectively
    reach the threshold, with no persistence requirement. Training accuracy is the current
    mini-batch's and validation accuracy is a fixed subset's, so both carry a sampling error
    of a few tenths of a per cent, and the step is resolved only to the logging stride. The
    extended reruns of appendix O instead require the threshold to hold over a sustained
    block; the two rules agree to within 70 steps on memorisation everywhere here.
    """
    from ..sketch.analysis import milestones

    return milestones(frame, threshold=threshold)


# -- the entry point an experiment calls ---------------------------------------


def run(ctx: Any, key: str, outdir: Optional[Any] = None, sketch: bool = False,
        overrides: Optional[Dict[str, Any]] = None, progress: bool = False,
        **sketch_options: Any) -> TrainingRun:
    """Train a registered run inside an experiment, recording what it was.

    The resolved configuration reaches ``ctx.config`` before a single step is taken. The
    archived tree stored no configuration beside its outputs, so a stored sketch could not
    be traced to the overrides that produced it; that is the hole this closes.

    The device comes from ``ctx``, which got it from ``actdim.runtime.device`` -- never from
    a local availability check.
    """
    from . import runs as registry

    config = registry.get(key)
    merged = dict(overrides or {})
    merged.setdefault("device", ctx.device)
    config = config.replace(**merged)

    recorder = None
    if sketch:
        from ..sketch.probe import TrajectoryRecorder

        recorder = TrajectoryRecorder(**sketch_options)

    ctx.config(**{f"run.{k}": v for k, v in config.to_dict().items()})
    result = train(config, outdir=outdir, progress=progress, observer=recorder)
    ctx.note(f"milestones.{key}", {"t_mem": result.t_mem, "t_gen": result.t_gen})
    if recorder is not None:
        ctx.note(f"sketch.{key}", recorder.metadata())
    return result

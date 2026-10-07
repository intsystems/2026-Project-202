"""One training loop for the quadratic perceptron, with the options each campaign needs.

The archived tree had two loops. One logged at a stride and wrote the spectral probes;
the other logged at every optimiser step and measured the sharpness, and to get those
two options it re-declared the column list, re-implemented the evaluation byte for byte
and re-wrote the loop -- some eighty lines of deliberate copy. The duplication is not
free. The evaluation is where the loss convention lives (the mean over every element
including the class axis, so that the initial loss is ``1/p``), and a change made in one
copy and not the other moves the two campaigns' logs onto different scales without
saying so anywhere in the output.

So there is one loop, and the differences are arguments to it:

* ``log_every = 1`` gives the per-step log appendix Q needs, because the oscillation at
  the edge of stability is the two-cycle of the unstable mode and a stride of ten does
  not blur it, it aliases it away.
* ``sharpness_every`` turns on the Hessian power iteration of
  ``actdim.training.eos``. Its start vector comes from its own stream, so switching it
  on cannot move the trajectory it is measuring.
* ``obs_every`` turns on the spectral probes of appendix M, which cost an SVD and are
  not wanted at a per-step stride.

Nothing here writes a file. ``train`` returns the rows; ``train_registered`` is the
entry point that takes a ``Context`` from its caller and writes through its store.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.perceptron import (QuadraticPerceptron, flat_parameters, n_parameters,
                                 spectra, weight_norm)
from ..runtime.determinism import stream_seed
from ..runtime.device import resolve
from ..tasks import arithmetic, polynomials

TRAIN_COLUMNS: Tuple[str, ...] = (
    "step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm")
"""The column set every downstream reader indexes on, in order.

The estimator reads a training log by these names. Renaming one is a breaking change to
every figure that reads a trajectory, so the set is declared once and shared by the
strided campaign and the per-step one rather than restated in each.
"""


# -- configuration -------------------------------------------------------------

@dataclass(frozen=True)
class PerceptronConfig:
    """One training run. The defaults are appendix O's setting for these rows: full
    batch gradient descent, no regularisation of any kind, quadratic activation, MSE on
    one-hot targets."""

    key: str = "custom"
    description: str = ""

    # -- data ------------------------------------------------------------------
    task: str = "add"                   # a key of tasks.arithmetic or tasks.polynomials
    p: int = 97                         # the modulus; it need not be prime
    n_vars: int = 2                     # operands, so D = n_vars * p
    fraction: float = 0.5               # alpha, the training fraction of Eq. (3)
    split_seed: int = 420               # draws the train/validation split, nothing else

    # -- model -----------------------------------------------------------------
    width: int = 500                    # N
    activation: str = "quadratic"
    init_seed: int = 1                  # draws W1 and W2, nothing else
    batch_seed: int = 10_001            # draws the mini-batch order; unused at full batch

    # -- optimisation ----------------------------------------------------------
    optimizer: str = "gd"               # gd | adam | adamw
    lr: float = 3e4
    weight_decay: float = 0.0
    momentum: float = 0.0               # gd only
    betas: Tuple[float, float] = (0.9, 0.98)
    max_steps: int = 100_000
    batch_size: Optional[int] = None    # None is full batch, which is appendix O's choice

    # -- logging ---------------------------------------------------------------
    log_every: int = 10                 # rows of the training log
    obs_every: int = 100                # rows of the spectral probes; 0 turns them off
    sharpness_every: int = 0            # steps between sharpness measurements; 0 is off
    sharpness_iters: int = 30           # power iterations per measurement
    sharpness_tol: float = 1e-4
    n_snapshots: int = 21               # log-spaced full weight dumps, float32
    progress_every: int = 5_000

    # -- numerics --------------------------------------------------------------
    dtype: str = "float32"
    device: str = "auto"                # resolved by actdim.runtime.device, never here

    def __post_init__(self):
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must lie in (0, 1), got {self.fraction}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be at least 1, got {self.log_every}")
        # Both strides have to land on logged steps, or the probe series and the loss
        # series never share a step and cannot be joined afterwards.
        for name in ("obs_every", "sharpness_every"):
            value = getattr(self, name)
            if value and value % self.log_every:
                raise ValueError(
                    f"{name} ({value}) must be a multiple of log_every "
                    f"({self.log_every}), else the two logs never share a step")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32 or float64, got {self.dtype!r}")

    @property
    def n_params(self) -> int:
        return n_parameters(self.p, self.width, self.n_vars)

    @property
    def batch(self) -> str:
        """How the run is batched, as a string that is never null.

        The archived metadata recorded ``batch_size = None`` for a full-batch run and
        then null-filled the field entirely, so the full-batch and mini-batch sketch
        campaigns had identical records although the batch rule was the only difference
        between them. A string that says ``full batch`` cannot be confused with a
        missing value.
        """
        return "full batch" if self.batch_size is None else f"minibatch {self.batch_size}"

    def summary(self) -> str:
        return (f"{self.task} mod {self.p} / N={self.width} {self.activation} / "
                f"{self.optimizer} lr={self.lr:g} wd={self.weight_decay:g} / "
                f"{self.batch} / alpha={self.fraction} / {self.max_steps} steps")

    def with_overrides(self, overrides: Dict[str, Any]) -> "PerceptronConfig":
        return replace(self, **{k: _coerce(k, v) for k, v in overrides.items()})

    def resolved(self) -> Dict[str, Any]:
        """Everything that was actually used, for the provenance record.

        The archived campaign rebuilt its summary file from the *current* registry, so
        editing a registry entry silently rewrote the recorded configuration of logs
        produced under the old one, and every wall-clock time was lost. This is the
        configuration object the run held, not a lookup, and ``train_registered`` hands
        it to ``ctx.config`` so the run records itself.
        """
        out = asdict(self)
        out["device"] = resolve(self.device)
        out["batch"] = self.batch
        out["n_params"] = self.n_params
        out["sharpness_seed"] = (stream_seed(self.init_seed, "sharpness_start")
                                 if self.sharpness_every else None)
        return out


_FIELDS = {f.name: f for f in fields(PerceptronConfig)}


def _coerce(name: str, raw: Any) -> Any:
    """Cast a ``key=value`` override to the declared type of the field."""
    if name not in _FIELDS:
        raise KeyError(f"unknown config field {name!r}. Known: {sorted(_FIELDS)}")
    if name == "betas":
        if isinstance(raw, (list, tuple)):
            return tuple(float(v) for v in raw)
        return tuple(float(v) for v in str(raw).split(",") if v.strip())
    if name == "batch_size":
        if raw is None or str(raw).strip().lower() in ("none", "null", ""):
            return None
        return int(raw)
    default = _FIELDS[name].default
    if isinstance(raw, type(default)) and not isinstance(default, str):
        return raw
    return type(default)(raw)


def label_function(cfg: PerceptronConfig) -> Callable[..., np.ndarray]:
    """The label function of a config, from whichever task table defines it.

    The arithmetic tasks are modulus-independent and the polynomial evaluators close
    over ``p``, so this must be called after every override has been applied: resolving
    it first and then setting ``p = 23`` would keep emitting labels for ``p = 97``.
    """
    try:
        return arithmetic.get(cfg.task)
    except KeyError:
        return polynomials.evaluator(cfg.task, cfg.p)


def describe(cfg: PerceptronConfig) -> str:
    if cfg.description:
        return cfg.description
    try:
        return arithmetic.describe(cfg.task)
    except KeyError:
        return polynomials.describe(cfg.task, cfg.p)


# -- data ----------------------------------------------------------------------

def build_dataset(cfg: PerceptronConfig,
                  fn: Optional[Callable[..., np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """All ``p ** n_vars`` operand tuples, one-hot encoded, split into train and val.

    The split is drawn from a generator seeded by ``split_seed`` alone, so changing the
    initialisation cannot change which examples are held out. That is the comparison
    appendix O's matched pairs need: two runs that differ only in the label function
    must see the same held-out set.
    """
    fn = fn or label_function(cfg)
    grids = np.meshgrid(*[np.arange(cfg.p) for _ in range(cfg.n_vars)], indexing="ij")
    operands = [g.reshape(-1) for g in grids]
    labels = np.asarray(fn(*operands), dtype=np.int64) % cfg.p

    m = labels.size
    x = np.zeros((m, cfg.n_vars * cfg.p), dtype=np.float32)
    for v, col in enumerate(operands):
        x[np.arange(m), v * cfg.p + col] = 1.0

    order = np.random.default_rng(cfg.split_seed).permutation(m)
    n_train = int(cfg.fraction * m)
    return {"x_train": x[order[:n_train]], "y_train": labels[order[:n_train]],
            "x_val": x[order[n_train:]], "y_val": labels[order[n_train:]]}


# -- the loop ------------------------------------------------------------------

def evaluate(model: QuadraticPerceptron, x: torch.Tensor, y: torch.Tensor,
             chunk: int = 4096) -> Tuple[float, float]:
    """Mean-reduced MSE against one-hot targets, and argmax accuracy.

    The reduction is the mean over *all* elements including the class axis, which is
    what ``nn.MSELoss`` does and what makes the initial loss ``1/p``. Summing over the
    class axis instead would inflate every loss by a factor of ``p``, and no reader of
    the log could tell which convention produced it. There is one copy of this, so the
    per-step campaign and the strided one cannot drift apart.
    """
    loss_sum, correct, n = 0.0, 0, x.shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            xb, yb = x[i:i + chunk], y[i:i + chunk]
            out = model(xb)
            target = torch.zeros_like(out)
            target[torch.arange(yb.shape[0], device=yb.device), yb] = 1.0
            loss_sum += float(((out - target) ** 2).sum())
            correct += int((out.argmax(dim=1) == yb).sum())
    return loss_sum / (n * model.p), correct / n


def _make_optimizer(model: torch.nn.Module, cfg: PerceptronConfig):
    if cfg.optimizer == "gd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                                 weight_decay=cfg.weight_decay)
    raise KeyError(f"unknown optimizer {cfg.optimizer!r}")


def _snapshot(parameter: Any) -> np.ndarray:
    """One parameter tensor as an array that will not change when training continues."""
    import torch

    return np.array(parameter.detach().to(torch.float32).cpu().numpy(), copy=True)


def _snapshot_steps(cfg: PerceptronConfig) -> set:
    if cfg.n_snapshots < 2:
        return set()
    # max(2, ...): geomspace with a single point returns the start and drops the end, so
    # n_snapshots = 2 would give {0, 1} and no end-of-training weights at all.
    n = max(2, cfg.n_snapshots - 1)
    steps = np.unique(np.round(np.geomspace(1.0, float(cfg.max_steps), n)).astype(int))
    return {int(s) for s in steps} | {0}


@dataclass
class TrainingRun:
    """What one run produced. No paths: the caller decides what to keep."""

    config: Dict[str, Any]
    device: str
    train_rows: List[Dict[str, Any]] = field(default_factory=list)
    obs_rows: List[Dict[str, Any]] = field(default_factory=list)
    sharp_rows: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: Dict[str, np.ndarray] = field(default_factory=dict)
    diverged_at: Optional[int] = None
    seconds: float = 0.0

    @property
    def summary(self) -> Dict[str, Any]:
        return grok_summary(self.train_rows, diverged_at=self.diverged_at)


def train(cfg: PerceptronConfig, fn: Optional[Callable[..., np.ndarray]] = None,
          on_row: Optional[Callable[[Dict[str, Any]], None]] = None,
          observer: Optional[Any] = None, verbose: bool = False) -> TrainingRun:
    """Train one config and return its rows.

    ``observer`` is an optional object with ``on_start(model, x_train)`` and
    ``on_log(step, model)``, called on every logged step. It exists so that the
    trajectory sketch can be recorded without this module knowing what a sketch is, and
    so that a run with the observer attached takes the same optimisation path as one
    without: nothing here reads it back.
    """
    device = resolve(cfg.device)
    tdtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    fn = fn or label_function(cfg)

    data = build_dataset(cfg, fn)
    def to(a, d):
        return torch.as_tensor(a, dtype=d, device=device)
    x_tr, y_tr = to(data["x_train"], tdtype), to(data["y_train"], torch.long)
    x_va, y_va = to(data["x_val"], tdtype), to(data["y_val"], torch.long)

    gen = torch.Generator(device=device).manual_seed(cfg.init_seed)
    model = QuadraticPerceptron(cfg.p, cfg.width, cfg.n_vars, cfg.activation,
                                generator=gen, dtype=tdtype, device=device)
    opt = _make_optimizer(model, cfg)

    target_tr = torch.zeros(x_tr.shape[0], cfg.p, dtype=tdtype, device=device)
    target_tr[torch.arange(x_tr.shape[0], device=device), y_tr] = 1.0

    # Its own seed, not init_seed plus a constant: tying the two makes it impossible to
    # vary the initialisation at a fixed batch order, which is the comparison the
    # mini-batch arm of the sketch campaign needs.
    batch_gen = torch.Generator(device=device).manual_seed(cfg.batch_seed)

    sharp_gen = None
    if cfg.sharpness_every:
        # The power-iteration start vector comes from a stream that nothing else draws
        # from, so turning the measurement on cannot move the trajectory it measures.
        # LEGACY_OFFSETS carries sharpness_start = 7,000,000, which is the offset the
        # archived campaign used, so this reproduces its stream exactly.
        sharp_gen = torch.Generator(device=device).manual_seed(
            stream_seed(cfg.init_seed, "sharpness_start"))

    snap_at = _snapshot_steps(cfg)
    if observer is not None:
        observer.on_start(model, x_tr)

    run = TrainingRun(config=cfg.resolved(), device=device)
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            tr_loss, tr_acc = evaluate(model, x_tr, y_tr)
            va_loss, va_acc = evaluate(model, x_va, y_va)
            row = dict(step=step, train_loss=tr_loss, val_loss=va_loss,
                       train_acc=tr_acc, val_acc=va_acc,
                       weight_norm=weight_norm(model))
            run.train_rows.append(row)
            if on_row is not None:
                on_row(row)
            if observer is not None:
                observer.on_log(step, model)
            if cfg.obs_every and (step % cfg.obs_every == 0 or step == cfg.max_steps):
                run.obs_rows.append(dict(step=step, **spectra(model)))
            if cfg.sharpness_every and step % cfg.sharpness_every == 0:
                from .eos import sharpness  # local: eos imports this module for the loop
                lam, used, rel = sharpness(model, x_tr, target_tr,
                                           iters=cfg.sharpness_iters,
                                           tol=cfg.sharpness_tol, generator=sharp_gen)
                run.sharp_rows.append(dict(step=step, lam_max=lam,
                                           eta_lam_over_2=cfg.lr * lam / 2.0,
                                           power_iters=used, power_rel_change=rel))
            if verbose and cfg.progress_every and step % cfg.progress_every == 0:
                print(f"  step {step:>7d}  train {tr_loss:.3e}/{tr_acc:6.2%}  "
                      f"val {va_loss:.3e}/{va_acc:6.2%}  "
                      f"|W|={row['weight_norm']:.3f}  {time.time() - t0:6.0f}s",
                      flush=True)

        if step in snap_at:
            # Copied, not viewed. `.to(dtype).cpu().numpy()` is a no-op chain when the
            # model is already float32 on the CPU, and returns an array sharing memory
            # with the live parameter: every snapshot then aliases the same buffer and all
            # twenty-one come out equal to the final weights. The file has the right
            # shape, the right keys and one array, which is why it went unnoticed.
            run.snapshots[f"W1_{step}"] = _snapshot(model.W1)
            run.snapshots[f"W2_{step}"] = _snapshot(model.W2)

        if step == cfg.max_steps:
            break

        if cfg.batch_size is None:
            xb, tb = x_tr, target_tr
        else:
            idx = torch.randint(0, x_tr.shape[0], (cfg.batch_size,),
                                generator=batch_gen, device=device)
            xb, tb = x_tr[idx], target_tr[idx]

        opt.zero_grad(set_to_none=True)
        loss = ((model(xb) - tb) ** 2).mean()
        loss.backward()
        opt.step()

        if not torch.isfinite(loss.detach()):
            # Recorded, not merely printed. Every milestone this run reports is
            # censored downstream on this field, because a run that blew up has no
            # generalisation step to report and no post-transition segment to slice.
            run.diverged_at = step
            if verbose:
                print(f"  diverged at step {step}", flush=True)
            break

    run.seconds = round(time.time() - t0, 3)
    return run


# -- what a finished run says ---------------------------------------------------

def grok_summary(train_rows: Sequence[Dict[str, Any]], thresh: float = 0.95,
                 diverged_at: Optional[int] = None) -> Dict[str, Any]:
    """When each split first crossed ``thresh`` accuracy, and where the run ended.

    ``t_grok`` is the first logged step at which validation accuracy reaches the
    threshold, and ``None`` means the run never generalised, which for the perturbed
    polynomials is the expected answer rather than a failure.

    **A diverged run reports no milestones.** The archived campaign let one report
    ``t_grok = 463`` on a run that blew up at step 567 with a final validation accuracy
    of 1 per cent: a real crossing, followed by a real explosion, and an analysis that
    keyed on it went looking for post-transition structure in a 567-step record. The
    crossings are kept under ``*_before_divergence`` so nothing is hidden, and the
    milestone fields are ``None`` so that a slicer finds nothing to slice.
    """
    if not train_rows:
        raise ValueError("a run with no logged rows has nothing to summarise")

    def first(col: str) -> Optional[int]:
        for r in train_rows:
            if r[col] >= thresh:
                return int(r["step"])
        return None

    last = train_rows[-1]
    out: Dict[str, Any] = dict(
        t_memorise=first("train_acc"), t_grok=first("val_acc"),
        final_train_acc=last["train_acc"], final_val_acc=last["val_acc"],
        final_train_loss=last["train_loss"], final_val_loss=last["val_loss"],
        best_val_acc=max(r["val_acc"] for r in train_rows),
        peak_val_loss=max(r["val_loss"] for r in train_rows),
        final_weight_norm=last["weight_norm"], steps=int(last["step"]),
        diverged_at=diverged_at,
    )
    if diverged_at is not None:
        out["t_memorise_before_divergence"] = out["t_memorise"]
        out["t_grok_before_divergence"] = out["t_grok"]
        out["t_memorise"] = None
        out["t_grok"] = None
    return out


# -- the trajectory sketch, as an observer --------------------------------------

class SketchRecorder:
    """Records the trajectory sketch on every logged step.

    The sketch itself lives in ``actdim.sketch.probe``; this is the adapter between it
    and the loop's observer hook, and it holds no sketching arithmetic of its own -- a
    second CountSketch would be a second thing to keep correct.

    What is recorded per logged step: the sketch of the flattened parameter vector, the
    sketch of the centred and normalised logits on a fixed set of training inputs, and
    three scalars (the parameter norm and the step lengths in parameter and function
    space). The array names are the ones the rank analysis reads.
    """

    def __init__(self, dim: int = 1024, n_sketch: int = 2, n_probe: int = 256,
                 seed: int = 0, progress_every: int = 0):
        self.dim, self.n_sketch, self.n_probe = dim, n_sketch, n_probe
        self.seed, self.progress_every = seed, progress_every
        self.n_params: Optional[int] = None
        self._sketch = None
        self._probe_inputs = None
        self._device = None
        self._steps: List[int] = []
        self._z: List[np.ndarray] = []
        self._zf: List[np.ndarray] = []
        self._scalars: Dict[str, List[float]] = {
            "param_norm": [], "param_step": [], "fn_step": []}
        self._prev_theta = None
        self._prev_fn = None
        self._started = 0.0

    def on_start(self, model: torch.nn.Module, x_train: torch.Tensor) -> None:
        # Imported here rather than at module scope so that a checkout without the
        # sketch package can still train; only a sketched run needs it.
        from ..sketch.probe import TrajectorySketch

        theta = flat_parameters(model)
        self.n_params = int(theta.numel())
        self._device = theta.device
        self._sketch = TrajectorySketch(self.n_params, dim=self.dim,
                                        n_sketch=self.n_sketch, seed=self.seed)
        # A stream of its own for the probe inputs: the sketch owns its hash seeds, and
        # drawing the inputs from the same generator would tie which examples are probed
        # to how many hash families the sketch happens to use.
        rng = np.random.default_rng(stream_seed(self.seed, "sketch_probe"))
        n = min(self.n_probe, int(x_train.shape[0]))
        index = np.sort(rng.choice(int(x_train.shape[0]), size=n, replace=False))
        self._probe_inputs = x_train[torch.from_numpy(index).to(x_train.device)]
        self._started = time.perf_counter()

    @torch.no_grad()
    def on_log(self, step: int, model: torch.nn.Module) -> None:
        from ..sketch.probe import normalise_logits, preserve_torch_rng

        theta = flat_parameters(model)
        self._steps.append(int(step))
        self._z.append(self._sketch.sketch_parameters(theta))
        self._scalars["param_norm"].append(float(torch.linalg.norm(theta)))
        self._scalars["param_step"].append(
            float("nan") if self._prev_theta is None
            else float(torch.linalg.norm(theta - self._prev_theta)))
        self._prev_theta = theta.clone()

        # The probe forward pass runs inside the guard even though this model has no
        # stochastic layer, so that non-invasiveness is a property of the code rather
        # than of an argument about the current architecture.
        with preserve_torch_rng(self._device):
            logits = model(self._probe_inputs)
        self._zf.append(self._sketch.sketch_function(logits))
        # Centred and L2-normalised, as the sketch itself is and as the archived probe
        # was. On raw logits this scalar measures the growth of the output scale rather
        # than a change in the function, and the two architectures' displacements would
        # not be on one axis -- appendix J compares them.
        fn = normalise_logits(logits).reshape(-1)
        self._scalars["fn_step"].append(
            float("nan") if self._prev_fn is None
            else float(torch.linalg.norm(fn - self._prev_fn)))
        self._prev_fn = fn.clone()

        if self.progress_every and len(self._steps) % self.progress_every == 0:
            print(f"    [sketch] {len(self._steps)} rows, step {step}, "
                  f"{time.perf_counter() - self._started:.0f}s", flush=True)

    def arrays(self) -> Dict[str, np.ndarray]:
        """The arrays to store, under the names the rank analysis reads."""
        out = {
            "step": np.asarray(self._steps, dtype=np.int64),
            "z": np.asarray(self._z, dtype=np.float64),    # (T, n_sketch, dim)
            "zf": np.asarray(self._zf, dtype=np.float64),
        }
        out.update({k: np.asarray(v, dtype=np.float64) for k, v in self._scalars.items()})
        return out

    def metadata(self) -> Dict[str, Any]:
        meta = {"n_params": self.n_params, "n_probe": self.n_probe, "seed": self.seed}
        if self._sketch is not None:
            meta.update(self._sketch.metadata())
        return meta


# -- the entry point ------------------------------------------------------------

def train_registered(ctx: Any, key: str, overrides: Optional[Dict[str, Any]] = None,
                     sketch: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Train one registered run, writing its logs through the caller's store.

    Files written, all under ``ctx.store``: ``<key>_train.csv`` always,
    ``<key>_obs.csv`` when the spectral probes are on, ``<key>_sharp.csv`` when the
    sharpness is measured, ``<key>_snapshots.npz`` when snapshots are asked for, and
    ``<key>_sketch.npz`` when ``sketch`` is set. Returns the run's record.

    The resolved configuration goes to ``ctx.config``, so the provenance beside the logs
    describes the run that produced them. The archived campaign instead rebuilt its
    summary from the registry after the fact, which meant an edit to a registry entry
    rewrote the recorded configuration of logs made under the old one, and lost every
    wall-clock time in the process.
    """
    # Local import: the registry imports this module for its config type, so importing
    # it at module scope would close a cycle.
    from . import runs_perceptron

    cfg = runs_perceptron.get(key)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    # The device is the runtime's decision, never this module's.
    cfg = replace(cfg, device=ctx.device)
    fn = label_function(cfg)

    recorder = None
    if sketch:
        recorder = SketchRecorder(seed=ctx.seed_for(f"sketch:{key}"), progress_every=500)

    if verbose:
        print(f"\n=== {key} ===\n{cfg.summary()}\n{describe(cfg)}", flush=True)

    # Rows are flushed as they are produced. A long run can lose its machine, and a
    # partial log that stops at step 60,000 is still a usable trajectory; a buffered one
    # that is lost whole is not.
    import csv

    train_csv = ctx.store.path(f"{key}_train.csv")
    handle = train_csv.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS))
    writer.writeheader()

    def on_row(row: Dict[str, Any]) -> None:
        writer.writerow(row)
        if row["step"] % 1000 == 0:
            handle.flush()

    try:
        run = train(cfg, fn, on_row=on_row, observer=recorder, verbose=verbose)
    finally:
        handle.close()
    ctx.store.adopt(train_csv)

    import pandas as pd

    if run.obs_rows:
        ctx.store.table(f"{key}_obs.csv", pd.DataFrame(run.obs_rows))
    if run.sharp_rows:
        ctx.store.table(f"{key}_sharp.csv", pd.DataFrame(run.sharp_rows))
    if run.snapshots:
        ctx.store.array(f"{key}_snapshots.npz", **run.snapshots)
    if recorder is not None:
        ctx.store.array(f"{key}_sketch.npz", **recorder.arrays())

    record: Dict[str, Any] = dict(key=key, **run.summary)
    record.update(rows=len(run.train_rows), seconds=run.seconds, device=run.device)
    # Every field that distinguishes one campaign from another, not only the ones that
    # name the task. Without the batch rule the full-batch and mini-batch sketch
    # campaigns wrote identical records, although the batch rule was the only thing
    # that differed between them.
    record.update({k: run.config[k] for k in (
        "task", "p", "width", "fraction", "optimizer", "lr", "weight_decay",
        "max_steps", "log_every", "batch_size", "batch", "init_seed", "split_seed",
        "dtype", "n_params")})
    if recorder is not None:
        record["sketch"] = recorder.metadata()

    ctx.config(**{key: run.config})
    if verbose:
        print(f"[{key}] memorised at {record['t_memorise']}, "
              f"generalised at {record['t_grok']}, final val acc "
              f"{record['final_val_acc']:.2%}, {record['seconds']:.0f}s", flush=True)
    return record


# -- what the sketch costs ------------------------------------------------------

def sketch_cost(cfg: PerceptronConfig, repeats: int = 2, dim: int = 1024,
                n_probe: int = 256, device: str = "auto") -> Dict[str, Any]:
    """Train the same configuration with and without the sketch, and time both.

    The article's practical claim for measuring a trajectory directly is that it is
    cheap: a CountSketch to 1024 dimensions at every logged step instead of the whole
    parameter vector. This measures that. The arms alternate so that a warm cache or a
    thermal ramp cannot favour one, and the logged losses are compared bit for bit,
    because a timing comparison between two runs that took different paths means
    nothing.

    Three things to keep in view when quoting the result.

    * The archived measurement reports a *negative* overhead: the probed arm came out
      3 per cent faster over two repeats of a 1500-step run, which is inside its own
      noise. A negative overhead is a bound, not a result. Raise ``repeats`` before
      quoting a figure with a sign.
    * It trains the perceptron, while the article describes the number at the logging
      stride the transformer runs use. The two have different parameter counts and
      different step costs, so the ratio does not transfer without saying so.
    * The archived record wrote ``device: "auto"``, the unresolved config value, so it
      could not say whether it had run on a GPU. The resolved device is recorded here.
    """
    resolved = resolve(device)
    cfg = replace(cfg, device=resolved)
    fn = label_function(cfg)

    bare: List[float] = []
    probed: List[float] = []
    reference: Optional[np.ndarray] = None
    for i in range(repeats):
        order = ("bare", "probed") if i % 2 == 0 else ("probed", "bare")
        for which in order:
            observer = (SketchRecorder(dim=dim, n_probe=n_probe, seed=0)
                        if which == "probed" else None)
            started = time.perf_counter()
            run = train(cfg, fn, observer=observer, verbose=False)
            elapsed = time.perf_counter() - started
            (probed if which == "probed" else bare).append(elapsed)

            losses = np.array([r["train_loss"] for r in run.train_rows])
            if reference is None:
                reference = losses
            elif not (losses.shape == reference.shape
                      and np.array_equal(losses, reference)):
                raise AssertionError(
                    "the two arms did not take the same optimisation path; the timing "
                    "comparison is meaningless if they diverge")

    bare_median, probed_median = float(np.median(bare)), float(np.median(probed))
    n_logged = cfg.max_steps // cfg.log_every + 1
    sketched_floats = n_logged * dim * 2 * 2      # two spaces, two hash families
    full_floats = n_logged * cfg.n_params
    return {
        "steps": cfg.max_steps, "log_every": cfg.log_every, "n_logged": n_logged,
        "n_params": cfg.n_params, "sketch_dim": dim, "repeats": repeats,
        "seconds_bare": bare_median, "seconds_probed": probed_median,
        "overhead_frac": (probed_median - bare_median) / bare_median,
        "sketched_float32_MB": sketched_floats * 4 / 1e6,
        "full_float32_MB": full_floats * 4 / 1e6,
        "storage_ratio": full_floats / sketched_floats,
        "device": resolved,   # resolved, so the record can say where it ran
        "model": "perceptron",
    }

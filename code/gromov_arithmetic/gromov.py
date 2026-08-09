"""Gromov's two-layer MLP and its training loop, written from the paper's equations.

Reference: A. Gromov, *Grokking modular arithmetic*, arXiv:2301.02679, Sec. 2.

The architecture is Eqs. (1)-(2), with the mean-field second-layer normalisation
``1/N`` rather than ``1/sqrt(N)``; for the quadratic activation the whole network
collapses to Eq. (4)::

    f(x) = 1 / (D * N) * W2 (W1 x)^2 ,      W1: N x D,  W2: p x N,  D = S * p

No biases.  Inputs are one-hot encodings of the ``S`` operands concatenated, so
``D = S p``; the target is the one-hot encoding of the answer.  Weights are drawn
from ``N(0, 1)`` -- the normalisation lives in the forward pass, not in the init.

That convention is not cosmetic.  It is what makes the paper's own figure readable:
with ``W ~ N(0,1)`` and the ``1/(DN)`` prefactor the initial output is ~0, so the
MSE starts at ``1/p`` (0.0105 for p=97, exactly the value Fig. 0a shows), and the
normalised weight norm starts at 1.0 (again Fig. 0a).  The reference implementation
of Doshi et al. folds the normalisation into the init instead; that rescales the
loss landscape and, with it, the usable learning rate.  Mixing the two conventions
silently changes the dynamics, so this module implements one of them and says which.

The paper states no learning rate anywhere.  ``lr_sweep.py`` calibrates it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

TRAIN_COLUMNS = ("step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm")
"""Column set of ``dimension_recovery/results/extended/*_train.csv``.

Kept byte-compatible on purpose: the intrinsic-dimension pipeline
(``active_dimension/e5_real_logs.py``) indexes on exactly these names, so a log
produced here drops into it without an adapter.
"""


@dataclass(frozen=True)
class Config:
    """One training run.  Defaults are the paper's headline setup: full-batch GD,
    no regularisation of any kind, quadratic activation, MSE on one-hot targets."""

    key: str = "custom"
    description: str = ""

    # --- data ---------------------------------------------------------------
    task: str = "add"                   # key into the task table passed to build_dataset
    p: int = 97                         # modulus; need not be prime (Sec. 2)
    n_vars: int = 2                     # operands, so D = n_vars * p
    fraction: float = 0.5               # alpha of Eq. (3)
    split_seed: int = 420               # draws the train/val split only

    # --- model --------------------------------------------------------------
    width: int = 500                    # N
    activation: str = "quadratic"       # quadratic | quartic | relu | abs | gelu
    init_seed: int = 1                  # draws W1, W2 only
    batch_seed: int = 10_001            # draws the mini-batch order only; unused at full batch

    # --- optimisation -------------------------------------------------------
    optimizer: str = "gd"               # gd | adam | adamw
    lr: float = 3e4
    weight_decay: float = 0.0           # the paper's claim: none is necessary
    momentum: float = 0.0               # gd only
    betas: Tuple[float, float] = (0.9, 0.98)
    max_steps: int = 100_000
    batch_size: Optional[int] = None    # None -> full batch (the paper's choice)

    # --- logging ------------------------------------------------------------
    log_every: int = 10                 # rows of <key>_train.csv
    obs_every: int = 100                # rows of <key>_obs.csv (the expensive probes)
    n_snapshots: int = 21               # log-spaced full-weight dumps, float32
    progress_every: int = 5_000

    # --- numerics -----------------------------------------------------------
    dtype: str = "float32"
    device: str = "auto"

    def __post_init__(self):
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must lie in (0,1), got {self.fraction}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.obs_every % self.log_every:
            raise ValueError(
                f"obs_every ({self.obs_every}) must be a multiple of log_every "
                f"({self.log_every}), else the two CSVs never share a step")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32 or float64, got '{self.dtype}'")

    @property
    def n_params(self):
        return self.width * self.n_vars * self.p + self.p * self.width

    def with_overrides(self, overrides):
        return replace(self, **{k: _coerce(k, v) for k, v in overrides.items()})

    def summary(self):
        batch = "full batch" if self.batch_size is None else f"batch {self.batch_size}"
        return (f"{self.task} mod {self.p} / N={self.width} {self.activation} / "
                f"{self.optimizer} lr={self.lr:g} wd={self.weight_decay:g} / {batch} / "
                f"alpha={self.fraction} / {self.max_steps} steps")


_FIELDS = {f.name: f for f in fields(Config)}


def _coerce(name, raw):
    """Cast a ``--set key=value`` override to the declared type of the field."""
    if name not in _FIELDS:
        raise KeyError(f"unknown config field '{name}'. Known: {sorted(_FIELDS)}")
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


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_dataset(cfg: Config, fn: Callable[..., np.ndarray]):
    """All ``p ** n_vars`` operand tuples, one-hot encoded, split into train/val.

    ``fn`` maps integer arrays of shape ``(M,)`` -- one per operand -- to the answer
    in ``Z_p``.  The split is drawn from a generator seeded by ``split_seed`` alone,
    so changing the initialisation cannot change which examples are held out.
    """
    grids = np.meshgrid(*[np.arange(cfg.p) for _ in range(cfg.n_vars)], indexing="ij")
    operands = [g.reshape(-1) for g in grids]
    labels = np.asarray(fn(*operands), dtype=np.int64) % cfg.p

    m = labels.size
    x = np.zeros((m, cfg.n_vars * cfg.p), dtype=np.float32)
    for v, col in enumerate(operands):
        x[np.arange(m), v * cfg.p + col] = 1.0

    order = np.random.default_rng(cfg.split_seed).permutation(m)
    n_train = int(cfg.fraction * m)
    return {
        "x_train": x[order[:n_train]], "y_train": labels[order[:n_train]],
        "x_val": x[order[n_train:]], "y_val": labels[order[n_train:]],
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "quadratic": lambda h: h * h,
    "quartic": lambda h: h ** 4,
    "relu": torch.relu,
    "abs": torch.abs,
    "gelu": torch.nn.functional.gelu,
}


class GromovMLP(torch.nn.Module):
    """Eqs. (1)-(2): ``f(x) = (1/N) W2 phi( W1 x / sqrt(D) )``, no biases."""

    def __init__(self, p, width, n_vars=2, activation="quadratic", generator=None,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        d_in = n_vars * p
        kw = dict(generator=generator, dtype=dtype, device=device)
        self.W1 = torch.nn.Parameter(torch.randn(width, d_in, **kw))
        self.W2 = torch.nn.Parameter(torch.randn(p, width, **kw))
        self.phi = ACTIVATIONS[activation]
        self.d_in, self.width, self.p, self.n_vars = d_in, width, p, n_vars

    def forward(self, x):
        h = (x @ self.W1.T) / math.sqrt(self.d_in)
        return (self.phi(h) @ self.W2.T) / self.width


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------

def _ipr(power):
    """Inverse participation ratio of a normalised power spectrum, per row.

    ``IPR = sum_j P_j^2`` with ``sum_j P_j = 1``.  1.0 means one frequency carries
    everything (the analytic solution); ~1/p means the row is spectrally flat, which
    is what random init looks like.  This is the order parameter of Doshi et al.
    (arXiv:2406.03495, Eq. 3) and the cheapest signal that a periodic representation
    has formed.
    """
    total = power.sum(axis=-1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    q = power / total
    return float((q ** 2).sum(axis=-1).mean())


def _fourier_ipr(block):
    """Mean IPR over rows of a ``(rows, p)`` weight block."""
    return _ipr(np.abs(np.fft.rfft(block, axis=-1)) ** 2)


def _participation(sv):
    """Effective rank ``(sum s^2)^2 / sum s^4`` -- the linear participation ratio.

    Reported alongside MG in the dimension work because the exp10-12 audit found it
    recovers a known subspace dimension more reliably than the delay-embedding
    estimators do.  Here it is a per-step measure of how many directions of each
    weight matrix are actually carrying the map.
    """
    s2 = sv.astype(np.float64) ** 2
    denom = (s2 ** 2).sum()
    return float(s2.sum() ** 2 / denom) if denom > 0 else 0.0


def observables(model, cfg):
    """The expensive probes: spectra of both layers, in numpy, off the autograd tape."""
    w1 = model.W1.detach().to(torch.float64).cpu().numpy()
    w2 = model.W2.detach().to(torch.float64).cpu().numpy()

    out = {}
    for v in range(cfg.n_vars):
        out[f"ipr_u{v + 1}"] = _fourier_ipr(w1[:, v * cfg.p:(v + 1) * cfg.p])
    out["ipr_w"] = _fourier_ipr(w2.T)              # readout, per neuron, over q

    sv1 = np.linalg.svd(w1, compute_uv=False)
    sv2 = np.linalg.svd(w2, compute_uv=False)
    out["erank_w1"] = _participation(sv1)
    out["erank_w2"] = _participation(sv2)
    out["w1_norm"] = float(np.linalg.norm(w1) / math.sqrt(w1.size))
    out["w2_norm"] = float(np.linalg.norm(w2) / math.sqrt(w2.size))
    for i in range(5):
        out[f"sv1_{i}"] = float(sv1[i]) if i < sv1.size else float("nan")
        out[f"sv2_{i}"] = float(sv2[i]) if i < sv2.size else float("nan")
    return out


OBS_COLUMNS = None  # filled on first use by train(); depends on n_vars


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _evaluate(model, x, y, chunk=4096):
    """Mean-reduced MSE against one-hot targets, and argmax accuracy.

    The reduction is the mean over *all* elements including the class axis, which is
    what ``nn.MSELoss()`` does and what makes the initial loss ``1/p``.  Summing over
    the class axis instead would inflate every loss by a factor of p.
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


def _make_optimizer(model, cfg):
    if cfg.optimizer == "gd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                                 weight_decay=cfg.weight_decay)
    raise KeyError(f"unknown optimizer '{cfg.optimizer}'")


def _snapshot_steps(cfg):
    if cfg.n_snapshots < 2:
        return set()
    lo, hi = 1.0, float(cfg.max_steps)
    # max(2, ...): geomspace with a single point returns [lo] and silently drops the
    # endpoint, so n_snapshots=2 would give {0, 1} -- no end-of-training weights at all.
    n = max(2, cfg.n_snapshots - 1)
    steps = np.unique(np.round(np.geomspace(lo, hi, n)).astype(int))
    return set(int(s) for s in steps) | {0}


def train(cfg: Config, fn, on_row=None, verbose=True, observer=None):
    """Train one config.  Returns ``(train_rows, obs_rows, snapshots)``.

    ``train_rows`` carries exactly ``TRAIN_COLUMNS``; ``obs_rows`` carries the
    spectral probes at the coarser ``obs_every`` stride; ``snapshots`` is a dict of
    log-spaced float32 weight dumps for post-hoc analysis.

    ``observer`` is an optional object with ``on_start(model, x_train)`` and
    ``on_log(step, model)``, called on every logged step.  It exists so
    ``rank.py`` can record the trajectory sketch without this module having to know
    what a sketch is, and so that a run with the observer attached takes the same
    optimisation path as one without: nothing here consults it.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

    data = build_dataset(cfg, fn)
    to = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    x_tr, y_tr = to(data["x_train"], dtype), to(data["y_train"], torch.long)
    x_va, y_va = to(data["x_val"], dtype), to(data["y_val"], torch.long)

    gen = torch.Generator(device=device).manual_seed(cfg.init_seed)
    model = GromovMLP(cfg.p, cfg.width, cfg.n_vars, cfg.activation,
                      generator=gen, dtype=dtype, device=device)
    opt = _make_optimizer(model, cfg)

    target_tr = torch.zeros(x_tr.shape[0], cfg.p, dtype=dtype, device=device)
    target_tr[torch.arange(x_tr.shape[0], device=device), y_tr] = 1.0

    # Its own seed, not init_seed + constant: tying the two makes it impossible to vary
    # the initialisation at a fixed batch order, which is exactly the comparison the
    # mini-batch variant of these runs would need.
    batch_gen = torch.Generator(device=device).manual_seed(cfg.batch_seed)
    snap_at = _snapshot_steps(cfg)
    if observer is not None:
        observer.on_start(model, x_tr)

    train_rows, obs_rows, snapshots = [], [], {}
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            tr_loss, tr_acc = _evaluate(model, x_tr, y_tr)
            va_loss, va_acc = _evaluate(model, x_va, y_va)
            wn = float(torch.sqrt(
                (model.W1.detach() ** 2).sum() + (model.W2.detach() ** 2).sum()))
            wn /= math.sqrt(model.W1.numel() + model.W2.numel())
            row = dict(step=step, train_loss=tr_loss, val_loss=va_loss,
                       train_acc=tr_acc, val_acc=va_acc, weight_norm=wn)
            train_rows.append(row)
            if on_row is not None:
                on_row(row)
            if observer is not None:
                observer.on_log(step, model)
            if step % cfg.obs_every == 0 or step == cfg.max_steps:
                obs_rows.append(dict(step=step, **observables(model, cfg)))
            if verbose and cfg.progress_every and step % cfg.progress_every == 0:
                print(f"  step {step:>7d}  train {tr_loss:.3e}/{tr_acc:6.2%}  "
                      f"val {va_loss:.3e}/{va_acc:6.2%}  |W|={wn:.3f}  "
                      f"{time.time() - t0:6.0f}s", flush=True)

        if step in snap_at:
            snapshots[f"W1_{step}"] = model.W1.detach().to(torch.float32).cpu().numpy()
            snapshots[f"W2_{step}"] = model.W2.detach().to(torch.float32).cpu().numpy()

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
            print(f"  diverged at step {step} (loss={float(loss.detach())})", flush=True)
            break

    return train_rows, obs_rows, snapshots


# ---------------------------------------------------------------------------
# summary of a finished run
# ---------------------------------------------------------------------------

def grok_summary(train_rows, thresh=0.95):
    """When each split first crosses ``thresh`` accuracy, and the final values.

    ``t_grok`` is the first logged step with ``val_acc >= thresh``; ``None`` means the
    run never generalised, which for the non-learnable polynomials is the expected
    answer rather than a failure.
    """
    def first(col):
        for r in train_rows:
            if r[col] >= thresh:
                return r["step"]
        return None
    last = train_rows[-1]
    return dict(
        t_memorise=first("train_acc"), t_grok=first("val_acc"),
        final_train_acc=last["train_acc"], final_val_acc=last["val_acc"],
        final_train_loss=last["train_loss"], final_val_loss=last["val_loss"],
        best_val_acc=max(r["val_acc"] for r in train_rows),
        peak_val_loss=max(r["val_loss"] for r in train_rows),
        final_weight_norm=last["weight_norm"], steps=last["step"],
    )

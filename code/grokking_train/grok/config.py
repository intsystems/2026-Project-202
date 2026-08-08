"""Declarative description of one training run.

A :class:`RunConfig` is the single source of truth for a run: which algebraic
task, which architecture, which optimizer, how often to log and which columns to
write.  ``runs.py`` is a registry of the configurations that produced the CSVs in
``../grokking_analysis/grokking_logs/``.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

OBSERVABLES = (
    "step",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "weight_norm",
    "grad_norm",
    "embed_grad_norm",
    "grad_cosine",
)
"""Everything the training loop knows how to record, one CSV column each."""

GRAD_OBSERVABLES = ("grad_norm", "embed_grad_norm", "grad_cosine")
"""Columns that require the (expensive) gradient probe -- see ``metrics.GradientProbe``."""

BASE_COLUMNS = ("step", "train_loss", "val_loss", "train_acc", "val_acc", "weight_norm")
"""Column set of the modular-addition logs (Fig. 1 of the paper)."""

FULL_COLUMNS = BASE_COLUMNS + GRAD_OBSERVABLES
"""Column set of the ``S_5`` logs (Figs. 2-3): adds the gradient diagnostics."""

BASELINE_COLUMNS = ("step", "train_loss", "train_acc", "val_loss", "val_acc")
"""Column set of the full-batch baseline log (App. B): no weight norm, different order."""


@dataclass(frozen=True)
class RunConfig:
    """One training run.  Field defaults describe the mini-batch Omnigrok setup."""

    key: str = "custom"
    description: str = ""

    # --- data ---------------------------------------------------------------
    task: str = "modular_addition"      # key into tasks.TASKS
    p: int = 113                        # modulus, for modular addition
    n: int = 5                          # degree, for the symmetric group S_n
    fraction: float = 0.3               # share of the pairs used for training
    max_pairs: Optional[int] = None     # subsample the product set (needed for large n)

    # --- model --------------------------------------------------------------
    model: str = "omnigrok"             # key into models.MODELS
    d_model: int = 128
    d_mlp: int = 512
    d_head: int = 32
    num_heads: int = 4
    n_layers: int = 1                   # "encoder" model only
    high_variance_init: bool = False    # "omnigrok" model only

    # --- optimization -------------------------------------------------------
    optimizer: str = "adamw"            # "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.98)
    momentum: float = 0.0               # "sgd" only
    max_steps: int = 20000
    batch_size: Optional[int] = 256     # None -> full batch
    val_batch_size: Optional[int] = 512  # None -> the whole validation split

    # --- logging ------------------------------------------------------------
    log_every: int = 10
    val_every: Optional[int] = None
    """How often to evaluate the validation split, in optimisation steps.

    ``None`` means "every logged step", which is the historical behaviour and keeps every
    existing run bit-identical.  Reconstructing the dynamics of ``train_loss`` and
    ``weight_norm`` calls for ``log_every = 1``, but those two are nearly free while a
    validation pass is not, so this decouples them: ``log_every=1, val_every=10`` logs the
    cheap series every step and evaluates the split at the old rate.  Rows without a fresh
    evaluation carry NaN in the validation columns.  Must be a multiple of ``log_every``.
    """
    columns: Tuple[str, ...] = BASE_COLUMNS
    csv: str = "{key}_logs.csv"         # ``str.format``-ed with the config fields

    # --- reproducibility ----------------------------------------------------
    seed: Optional[int] = 42            # None -> do not touch the global torch RNG
    init_seed: Optional[int] = None
    """Reseed the global torch RNG *after* the task is built, before the model is.

    ``seed`` alone cannot separate the two things it controls: ``tasks`` seeds the
    stream and draws the train/val split from it, and the model initialisation then
    continues that same stream, so changing ``seed`` changes *which examples are in
    the training set* and *what the initial weights are* together.

    Setting ``init_seed`` restarts the stream between those two steps, making the
    initialisation (and the mini-batch order that follows it) depend only on
    ``init_seed`` and the split only on ``seed``. That is what lets a run be varied
    along one axis at a time -- see ``../prediction_improved/sweep.py``.

    ``None`` (the default) leaves the stream alone, so every existing run is
    unaffected, bit for bit.
    """
    dtype: str = "float64"              # "float32" | "float64"
    device: str = "auto"                # "auto" | "cpu" | "cuda:0" | ...

    # --- faithfulness switch ------------------------------------------------
    double_step: bool = False
    """Call ``optimizer.step()`` twice per batch.

    A bug in ``Grokking/generator_logs_to_S_5_with_stochastic*.ipynb`` that the
    published ``S_5`` logs were produced with: the same gradient is applied twice
    and the optimizer state is advanced twice per batch, so the run sees roughly
    double the intended learning rate and weight decay.  Kept behind a flag
    because the article's numbers cannot be reproduced without it.  Leave it off
    for new runs.
    """

    def __post_init__(self):
        unknown = [c for c in self.columns if c not in OBSERVABLES]
        if unknown:
            raise ValueError(f"unknown log column(s) {unknown}. Known: {list(OBSERVABLES)}")
        if "step" not in self.columns:
            raise ValueError("'step' is required -- the analysis package indexes on it")
        for required in ("train_acc", "val_acc"):
            if required not in self.columns:
                raise ValueError(
                    f"'{required}' is required: edm.load_logs() rejects logs without it"
                )
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.val_every is not None:
            if self.val_every < 1:
                raise ValueError(f"val_every must be >= 1, got {self.val_every}")
            if self.val_every % self.log_every:
                raise ValueError(
                    f"val_every ({self.val_every}) must be a multiple of "
                    f"log_every ({self.log_every}), else validation never lands on a "
                    f"logged row")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must lie in (0, 1), got {self.fraction}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32 or float64, got '{self.dtype}'")

    # --- derived ------------------------------------------------------------

    @property
    def needs_grad_probe(self):
        return any(c in self.columns for c in GRAD_OBSERVABLES)

    @property
    def csv_name(self):
        """The output file name, with ``{p}`` / ``{n}`` / ``{key}`` ... substituted."""
        return self.csv.format(**{f.name: getattr(self, f.name) for f in fields(self)})

    @property
    def expected_rows(self):
        """How many rows the CSV will have (one per logged step)."""
        return (self.max_steps + self.log_every - 1) // self.log_every

    def with_overrides(self, overrides):
        """Return a copy with ``{field: raw_string_or_value}`` applied and coerced."""
        return replace(self, **{k: coerce(k, v) for k, v in overrides.items()})

    def summary(self):
        wd = f"WD={self.weight_decay}"
        batch = "full batch" if self.batch_size is None else f"batch {self.batch_size}"
        return (f"{self.task} / {self.model} / {self.optimizer} lr={self.lr} {wd} / "
                f"{batch} / {self.max_steps} steps, log every {self.log_every}")


# --------------------------------------------------------------------------
# Coercion of ``--set key=value`` command-line overrides
# --------------------------------------------------------------------------

_FIELD_TYPES = {f.name: f for f in fields(RunConfig)}


def _to_bool(raw):
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
    "max_pairs": _optional(int),
    "batch_size": _optional(int),
    "val_batch_size": _optional(int),
    "seed": _optional(int),
    "init_seed": _optional(int),
    "val_every": _optional(int),
    "betas": _tuple_of(float),
    "columns": _tuple_of(lambda s: str(s).strip()),
    "double_step": _to_bool,
    "high_variance_init": _to_bool,
}


def coerce(name, raw):
    """Cast a raw override to the declared type of ``RunConfig.<name>``."""
    if name not in _FIELD_TYPES:
        raise KeyError(f"unknown config field '{name}'. Known: {sorted(_FIELD_TYPES)}")
    if name in _CASTS:
        return _CASTS[name](raw)
    default = _FIELD_TYPES[name].default
    if isinstance(raw, type(default)) and not isinstance(default, str):
        return raw
    return type(default)(raw)

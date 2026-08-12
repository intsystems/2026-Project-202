"""Training runs that produce the grokking logs analysed by ``../grokking_analysis``.

The pipeline is: :class:`~grok.config.RunConfig` -> :func:`~grok.loop.train` ->
one CSV per run -> ``edm.load_logs`` on the other side.  ``runs.py`` in the parent
directory registers the configurations behind the published figures.

:mod:`grok.groups` (the finite-group algebra) and :mod:`grok.config` (the run
description) are pure NumPy and import without torch, so the algebra and the
registry can be checked on a machine that has no deep-learning stack.  Everything
else is imported lazily, on first attribute access.
"""

import importlib

from .config import (
    BASE_COLUMNS,
    BASELINE_COLUMNS,
    FULL_COLUMNS,
    GRAD_OBSERVABLES,
    OBSERVABLES,
    RunConfig,
)
from .groups import SymmetricGroup, minimal_faithful_dimension, permutations, rank

_LAZY = {
    "EncoderTransformer": "models",
    "GradientProbe": "metrics",
    "MODELS": "models",
    "NandaTransformer": "models",
    "OmnigrokTransformer": "models",   # deprecated alias of NandaTransformer
    "TASKS": "tasks",
    "Task": "tasks",
    "accuracy": "metrics",
    "build_optimizer": "loop",
    "default_dtype": "loop",
    "modular_addition": "tasks",
    "resolve_device": "loop",
    "symmetric_group": "tasks",
    "train": "loop",
    "weight_norm": "metrics",
}


def __getattr__(name):
    """Import the torch-backed half of the package on demand (PEP 562)."""
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "BASELINE_COLUMNS",
    "BASE_COLUMNS",
    "FULL_COLUMNS",
    "GRAD_OBSERVABLES",
    "OBSERVABLES",
    "RunConfig",
    "SymmetricGroup",
    "minimal_faithful_dimension",
    "permutations",
    "rank",
    *sorted(_LAZY),
]

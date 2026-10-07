"""The experiments, one module per group.

Importing this package registers nothing; ``load_all()`` imports every sibling module,
and each of those registers its experiments through the ``@experiment`` decorator. The
split keeps ``python -m actdim list`` from importing torch when nothing needs it.

Groups:

    calib   the two frozen estimator configurations
    sys     the systems whose active dimension is fixed by construction (section 5)
    valid   the conditions under which the estimate may be read (section 6)
    train   the training campaigns of appendix O
    grok    the application to delayed generalisation (section 7)
    check   the checks and the cost measurements
    paper   figures and the table audit
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import List

_LOADED = False


def load_all() -> List[str]:
    """Import every experiment module exactly once."""
    global _LOADED
    if _LOADED:
        return []
    names = []
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")
        names.append(module.name)
    _LOADED = True
    return names

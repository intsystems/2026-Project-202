"""Re-export the training core from ``../gromov_arithmetic``.

The two papers share one architecture, so they share one implementation; duplicating
it would let the two folders drift apart, and the whole point of the polynomial runs
is that they are the *same* network as the arithmetic runs.  This module is only the
import path, isolated here so the sys.path edit happens in exactly one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "gromov_arithmetic"
if not (_CORE / "gromov.py").exists():
    raise ImportError(
        f"cannot find gromov.py under {_CORE}. The polynomial runs reuse the core "
        f"from ../gromov_arithmetic; both folders must be present.")
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from gromov import (  # noqa: E402,F401
    Config, GromovMLP, TRAIN_COLUMNS, build_dataset, grok_summary, observables, train,
)

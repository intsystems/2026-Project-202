"""Counting the active degrees of freedom of a training run.

The code behind the article: the estimator, the systems whose active dimension is known
by construction, the two training settings, the trajectory sketch, and the analysis that
turns their logs into the article's numbers and figures.

Start at ``docs/architecture.md`` for the module map, ``docs/experiments.md`` for the
table that maps every section of the article to the experiment that produced it, and
``docs/reproduce.md`` for how to regenerate the lot.

    python -m actdim list
    python -m actdim run sys.matrix
"""
from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]

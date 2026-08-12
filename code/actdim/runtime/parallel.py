"""Parallel map that returns results in the order they were asked for.

The archived experiments collected from ``as_completed``, so the row order of every raw
CSV depended on which worker happened to finish first. The values were stable and the
files still could not be diffed against a re-run. Ordering here is by input index, always.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, List, Optional, Sequence

from .determinism import pin_blas_threads


def default_jobs(requested: Optional[int] = None) -> int:
    """How many workers to use.

    Leaves a core free so the machine stays usable, and never returns more workers than
    there are cores: over-subscribing a BLAS-heavy workload makes it slower, not faster.
    """
    if requested and requested > 0:
        return requested
    cores = os.cpu_count() or 1
    return max(1, cores - 1)


def _init_worker() -> None:
    pin_blas_threads(1)


def map_ordered(fn: Callable[[Any], Any], items: Sequence[Any], jobs: int = 1,
                desc: str = "", progress: bool = True) -> List[Any]:
    """Apply ``fn`` to each item, in parallel, returning results in input order.

    Falls back to a serial loop when ``jobs <= 1`` or there is one item, which keeps
    stack traces readable while developing and makes ``--jobs 1`` a real debugging tool.
    """
    items = list(items)
    if not items:
        return []
    jobs = min(default_jobs(jobs), len(items))

    if jobs <= 1:
        return [fn(item) for item in _tick(items, desc, progress)]

    from concurrent.futures import ProcessPoolExecutor

    results: List[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        from concurrent.futures import as_completed

        for done in _tick(as_completed(futures), desc, progress, total=len(items)):
            index = futures[done]
            results[index] = done.result()
    return results


def _tick(iterable: Iterable[Any], desc: str, progress: bool, total: Optional[int] = None):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc or None, total=total, leave=False)

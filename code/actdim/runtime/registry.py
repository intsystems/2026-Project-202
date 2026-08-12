"""The catalogue of experiments, and the order they run in.

Every number in the article is produced by exactly one registered experiment. An
experiment declares what it needs, what it writes, which of those the article reads, and
roughly what it costs, so that ``actdim plan`` can order a full regeneration and say what
it will take before starting.

The archived tree had no such catalogue. Its nearest equivalent ran five of twenty-three
scripts, two of them commented out mid-chain, and the real dependency order had to be
recovered by reading imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CPU = "cpu"
GPU = "gpu"


@dataclass(frozen=True)
class Experiment:
    """One experiment: a unit of work with a stable id."""

    id: str
    title: str
    fn: Callable[["Context"], None]  # noqa: F821 -- see actdim.runtime.context
    paper: Tuple[str, ...] = ()
    device: str = CPU
    minutes: float = 1.0
    needs: Tuple[str, ...] = ()
    promotes: Tuple[str, ...] = ()
    tier: int = 0
    notes: str = ""

    @property
    def group(self) -> str:
        return self.id.split(".")[0]


REGISTRY: Dict[str, Experiment] = {}


def experiment(
    id: str,
    title: str,
    paper: Sequence[str] = (),
    device: str = CPU,
    minutes: float = 1.0,
    needs: Sequence[str] = (),
    promotes: Sequence[str] = (),
    tier: int = 0,
    notes: str = "",
) -> Callable[[Callable], Callable]:
    """Register the decorated function as an experiment.

    The function is returned unchanged, so it stays directly callable in a test.
    """

    def decorate(fn: Callable) -> Callable:
        if id in REGISTRY:
            raise ValueError(f"duplicate experiment id: {id}")
        REGISTRY[id] = Experiment(
            id=id, title=title, fn=fn, paper=tuple(paper), device=device,
            minutes=minutes, needs=tuple(needs), promotes=tuple(promotes),
            tier=tier, notes=notes,
        )
        return fn

    return decorate


def load() -> Dict[str, Experiment]:
    """Import every experiment module, populating the registry."""
    from .. import experiments  # noqa: F401  -- importing it registers everything

    experiments.load_all()
    return REGISTRY


def get(id: str) -> Experiment:
    if id not in REGISTRY:
        load()
    if id not in REGISTRY:
        raise KeyError(f"no such experiment: {id}")
    return REGISTRY[id]


def select(patterns: Iterable[str]) -> List[Experiment]:
    """Resolve ids or prefixes to experiments, keeping registration order.

    ``sys`` selects every ``sys.*`` experiment; ``sys.matrix`` selects just that one.
    """
    load()
    wanted: List[Experiment] = []
    for pattern in patterns:
        matches = [e for e in REGISTRY.values()
                   if e.id == pattern or e.id.startswith(pattern.rstrip(".") + ".")]
        if not matches:
            raise KeyError(f"no experiment matches {pattern!r}")
        for match in matches:
            if match not in wanted:
                wanted.append(match)
    return wanted


def order(targets: Sequence[Experiment], include_needs: bool = True) -> List[Experiment]:
    """Order the targets so that every prerequisite runs before what needs it.

    Among steps that are ready at the same moment, the lower tier goes first, and ties
    inside a tier break by id. Dependency always wins over tier: the two disagreed once --
    the article's figures were drawn at a point where the section 6 experiments had not
    run, and three of the twelve came out of stale data reporting themselves clean -- and
    the fix is to state the dependency, not to reorder around it.

    Raises on a cycle rather than looping, and names the steps involved.
    """
    load()
    wanted: Dict[str, Experiment] = {}

    def collect(exp: Experiment, path: Tuple[str, ...]) -> None:
        if exp.id in path:
            raise ValueError("dependency cycle: " + " -> ".join(path + (exp.id,)))
        if exp.id in wanted:
            return
        wanted[exp.id] = exp
        if include_needs:
            for need in exp.needs:
                collect(get(need), path + (exp.id,))

    for target in targets:
        collect(target, ())

    # Kahn's algorithm, taking the lowest (tier, id) among whatever is ready. Needs that
    # fall outside the selection -- when --no-deps is given -- are not waited on.
    remaining = {
        id: {need for need in exp.needs if need in wanted} if include_needs else set()
        for id, exp in wanted.items()
    }
    resolved: List[Experiment] = []
    while remaining:
        ready = [id for id, needs in remaining.items() if not needs]
        if not ready:
            raise ValueError("dependency cycle among: " + ", ".join(sorted(remaining)))
        chosen = min(ready, key=lambda id: (wanted[id].tier, id))
        resolved.append(wanted.pop(chosen))
        remaining.pop(chosen)
        for needs in remaining.values():
            needs.discard(chosen)
    return resolved


def cost(experiments: Sequence[Experiment]) -> Dict[str, float]:
    """Estimated minutes, split by device."""
    return {
        "cpu_minutes": sum(e.minutes for e in experiments if e.device == CPU),
        "gpu_minutes": sum(e.minutes for e in experiments if e.device == GPU),
        "total_minutes": sum(e.minutes for e in experiments),
    }

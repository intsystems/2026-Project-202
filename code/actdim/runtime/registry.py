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
    """Topologically order the targets, with their prerequisites first.

    Raises on a cycle rather than looping, and names the cycle.
    """
    load()
    resolved: List[Experiment] = []
    seen: Dict[str, int] = {}  # 1 = in progress, 2 = done

    def visit(exp: Experiment, path: Tuple[str, ...]) -> None:
        state = seen.get(exp.id, 0)
        if state == 2:
            return
        if state == 1:
            raise ValueError("dependency cycle: " + " -> ".join(path + (exp.id,)))
        seen[exp.id] = 1
        if include_needs:
            for need in exp.needs:
                visit(get(need), path + (exp.id,))
        seen[exp.id] = 2
        resolved.append(exp)

    for target in targets:
        visit(target, ())
    return resolved


def cost(experiments: Sequence[Experiment]) -> Dict[str, float]:
    """Estimated minutes, split by device."""
    return {
        "cpu_minutes": sum(e.minutes for e in experiments if e.device == CPU),
        "gpu_minutes": sum(e.minutes for e in experiments if e.device == GPU),
        "total_minutes": sum(e.minutes for e in experiments),
    }

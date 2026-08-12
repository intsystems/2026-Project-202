"""What an experiment is handed when it runs.

An experiment function takes one argument, a ``Context``, and writes through
``ctx.store``. It reads an upstream experiment's output through ``ctx.input``, never by
building a path itself, so that a missing prerequisite is reported as a missing
prerequisite rather than as a file-not-found from three frames down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from . import device as device_mod
from .determinism import rng, seed_map, stream_seed
from .provenance import Provenance
from .store import Store, data_root, runs_root


@dataclass
class Context:
    """The run environment of one experiment."""

    experiment: str
    store: Store
    device: str = "cpu"
    jobs: int = 1
    seed: int = 0
    fast: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    # -- randomness ------------------------------------------------------------

    def rng(self, role: str):
        """A NumPy generator for a named stream, recorded in the provenance."""
        self.store.provenance.seeds.setdefault(role, stream_seed(self.seed, role))
        return rng(self.seed, role)

    def seed_for(self, role: str) -> int:
        self.store.provenance.seeds.setdefault(role, stream_seed(self.seed, role))
        return stream_seed(self.seed, role)

    def declare_seeds(self, *roles: str) -> Dict[str, int]:
        seeds = seed_map(self.seed, roles)
        self.store.provenance.seeds.update(seeds)
        return seeds

    # -- inputs ----------------------------------------------------------------

    def input(self, experiment: str, name: str) -> Path:
        """The path to an upstream experiment's output.

        Looks in ``runs/`` first, then in the tracked ``data/`` tree, so that a fresh
        clone can rebuild the downstream half of the article from committed files without
        re-running the expensive upstream half.
        """
        candidates = [runs_root() / experiment / name,
                      data_root() / experiment / name,
                      data_root() / experiment / Path(name).name]
        for path in candidates:
            if path.exists():
                self.store.provenance.inputs.setdefault(
                    f"{experiment}/{name}", path.relative_to(path.parents[2]).as_posix())
                return path
        raise FileNotFoundError(
            f"{self.experiment} needs {name!r} from {experiment!r}, which has not been "
            f"produced.\nRun it first:  python -m actdim run {experiment}"
        )

    def input_dir(self, experiment: str) -> Path:
        for root in (runs_root(), data_root()):
            path = root / experiment
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"{self.experiment} needs the outputs of {experiment!r}, which has not run.\n"
            f"Run it first:  python -m actdim run {experiment}"
        )

    # -- convenience -----------------------------------------------------------

    def torch_device(self) -> Any:
        import torch

        return torch.device(self.device)

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)

    def note(self, key: str, value: Any) -> None:
        """Record something worth keeping beside the results."""
        self.store.provenance.notes[key] = value

    def config(self, **values: Any) -> None:
        """Record the resolved configuration this run used."""
        self.store.provenance.config.update(values)


def build(experiment: str, device: str = "auto", jobs: int = 1, seed: int = 0,
          fast: bool = False, options: Optional[Dict[str, Any]] = None,
          root: Optional[Path] = None) -> Context:
    """Create the context and its provenance record for one run."""
    resolved = device_mod.resolve(device)
    provenance = Provenance(experiment=experiment, device=resolved, jobs=jobs, fast=fast)
    provenance.notes["device_detail"] = device_mod.describe(device)
    provenance.start()
    store = Store(experiment, provenance, root=root)
    return Context(experiment=experiment, store=store, device=resolved, jobs=jobs,
                   seed=seed, fast=fast, options=dict(options or {}))

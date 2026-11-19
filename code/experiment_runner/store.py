"""Read bundled evidence and write the outputs of fresh experiment runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiment_runner.provenance import Provenance


def repo_root() -> Path:
    """The ``code/`` directory, whatever the working directory is."""
    return Path(__file__).resolve().parents[1]


def runs_root() -> Path:
    return repo_root() / "runs"


def bundled_file(experiment: str, name: str) -> Path:
    """Resolve a shipped training log or analysis table by its experiment ID."""
    folder = BUNDLED_EVIDENCE.get(experiment)
    if folder:
        logs = repo_root() / folder / "logs"
        suffixed = logs / f"{experiment.replace('.', '_')}__{name}"
        if suffixed.is_file():
            return suffixed
        plain = logs / name
        return plain.with_suffix(".csv") if name == "rank_milestones.json" else plain
    return repo_root() / "data" / experiment / name


BUNDLED_EVIDENCE = {'grok.diagnostics.logs': 'real/grokking_and_stability/transformer_grokking',
 'grok.diagnostics.perceptron': 'real/grokking_and_stability/perceptron_grokking',
 'grok.eos': 'real/grokking_and_stability/edge_of_stability',
 'grok.extended.outcomes': 'real/grokking_and_stability/transformer_grokking',
 'grok.matched.surrogate': 'real/grokking_and_stability/transformer_grokking',
 'grok.matched.window': 'real/grokking_and_stability/transformer_grokking',
 'grok.prwindow': 'real/grokking_and_stability/perceptron_grokking',
 'grok.rank.dip': 'real/grokking_and_stability/transformer_grokking',
 'grok.repr': 'real/grokking_and_stability/perceptron_grokking',
 'grok.repr.measured': 'real/grokking_and_stability/perceptron_grokking',
 'train.perceptron.arith': 'real/grokking_and_stability/perceptron_grokking',
 'train.perceptron.eos': 'real/grokking_and_stability/edge_of_stability',
 'train.perceptron.poly': 'real/grokking_and_stability/perceptron_grokking',
 'train.perceptron.sketched.fb': 'real/grokking_and_stability/perceptron_grokking',
 'train.perceptron.sketched.long': 'real/grokking_and_stability/perceptron_grokking',
 'train.perceptron.sketched.mb': 'real/grokking_and_stability/perceptron_grokking',
 'train.transformer.extended': 'real/grokking_and_stability/transformer_grokking',
 'train.transformer.p211': 'real/grokking_and_stability/transformer_grokking',
 'train.transformer.sketched': 'real/grokking_and_stability/transformer_grokking'}


class Store:
    """The output directory of one experiment run."""

    def __init__(self, experiment: str, provenance: Provenance, root: Optional[Path] = None):
        self.experiment = experiment
        self.provenance = provenance
        self.dir = (root or runs_root()) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- writers ---------------------------------------------------------------

    def table(self, name: str, frame: Any, **extra: Any) -> Path:
        """Write a DataFrame as CSV.

        Rows are written in the order given. Experiments that collect results from a
        process pool must sort before calling this: the archived tree wrote pool
        completion order, so re-running produced the same numbers in a different order
        and no diff could be taken.
        """
        path = self._path(name if name.endswith(".csv") else name + ".csv")
        frame.to_csv(path, index=False)
        self.provenance.record_output(path, self.dir, rows=int(len(frame)),
                                      columns=list(map(str, frame.columns)), **extra)
        return path

    def array(self, name: str, **arrays: np.ndarray) -> Path:
        """Write arrays as a compressed ``.npz``."""
        path = self._path(name if name.endswith(".npz") else name + ".npz")
        np.savez_compressed(path, **arrays)
        self.provenance.record_output(
            path, self.dir, arrays={k: list(v.shape) for k, v in arrays.items()
                                    if hasattr(v, "shape")})
        return path

    def json(self, name: str, obj: Any) -> Path:
        path = self._path(name if name.endswith(".json") else name + ".json")
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n",
                        encoding="utf-8")
        self.provenance.record_output(path, self.dir)
        return path

    def text(self, name: str, body: str) -> Path:
        path = self._path(name)
        path.write_text(body, encoding="utf-8")
        self.provenance.record_output(path, self.dir)
        return path

    def path(self, name: str) -> Path:
        """A path inside the run directory, for a writer this class does not wrap."""
        return self._path(name)

    def adopt(self, path: Path) -> None:
        """Record a file written by something other than this class."""
        self.provenance.record_output(path, self.dir)

    def _path(self, name: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # -- reading a previous run ------------------------------------------------

    def existing(self, name: str) -> Optional[Path]:
        path = self.dir / name
        return path if path.exists() else None

    def close(self, status: str = "ok") -> None:
        self.provenance.finish(status)
        self.provenance.write(self.dir / "provenance.json")


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


# -- the tracked half ----------------------------------------------------------

def is_plumbing_check(run_dir: Path) -> bool:
    """Was this run directory produced by a ``--fast`` run?

    A ``--fast`` run computes the smallest grid that exercises every branch: the right
    files, the right columns, the wrong numbers. Three places have to know the difference,
    and each of them was a real failure before they did. It must not be promoted into
    ``data/``; it must not satisfy a downstream experiment's input; and it must not count
    as "already run", or a smoke test at teatime silently removes an experiment from the
    overnight campaign.
    """
    record = run_dir / "provenance.json"
    if not record.exists():
        return False
    try:
        return bool(json.loads(record.read_text(encoding="utf-8")).get("fast"))
    except (OSError, ValueError):
        return False

"""Where results go, and how they are written.

Two directories, with different rules.

``runs/<id>/`` holds everything an experiment produces, is regenerable, and is not in
version control. Trajectories and sketches live here; they run to hundreds of megabytes.

``data/`` holds the small subset the article reads -- what the figures load and what the
tables quote. It is in version control, and every file in it is listed in
``data/manifest.json`` with its checksum and the experiment that produced it. Promotion
from one to the other is explicit: an experiment declares its paper-facing outputs and
``actdim promote`` copies them.

Writers go through this class so that every file is checksummed as it is written, rather
than by a later pass that can miss one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .provenance import Provenance, sha256


def repo_root() -> Path:
    """The ``code/`` directory, whatever the working directory is."""
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return repo_root() / "runs"


def data_root() -> Path:
    return repo_root() / "data"


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

    def figure(self, name: str, fig: Any, **kwargs: Any) -> Path:
        """Write a matplotlib figure. See ``actdim.figures`` for the article's rules."""
        path = self._path(name)
        fig.savefig(path, **kwargs)
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


MANIFEST = "manifest.json"


def load_manifest() -> Dict[str, Any]:
    path = data_root() / MANIFEST
    if not path.exists():
        return {"schema": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: Dict[str, Any]) -> Path:
    data_root().mkdir(parents=True, exist_ok=True)
    path = data_root() / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def promote(experiment: str, names: Iterable[str], subdir: Optional[str] = None) -> Dict[str, Any]:
    """Copy an experiment's declared outputs into ``data/`` and record them.

    The manifest entry names the producing experiment and the command that rebuilds it,
    so a file in ``data/`` can always be traced to the run that made it.
    """
    import shutil

    manifest = load_manifest()
    source_dir = runs_root() / experiment
    target_dir = data_root() / (subdir or experiment)
    target_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = source_dir / "provenance.json"
    provenance = (json.loads(provenance_path.read_text(encoding="utf-8"))
                  if provenance_path.exists() else {})

    promoted = {}
    for name in names:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"{experiment} declared {name} but did not write it")
        dst = target_dir / Path(name).name
        shutil.copy2(src, dst)
        rel = dst.relative_to(data_root()).as_posix()
        entry = {
            "experiment": experiment,
            "command": f"python -m actdim run {experiment}",
            "sha256": sha256(dst),
            "bytes": dst.stat().st_size,
            "git": provenance.get("git", {}),
            "produced_utc": provenance.get("finished_utc"),
        }
        manifest["files"][rel] = entry
        promoted[rel] = entry

    write_manifest(manifest)
    return promoted


def verify() -> Dict[str, list]:
    """Check every tracked file against its recorded checksum."""
    manifest = load_manifest()
    ok, changed, missing = [], [], []
    for rel, entry in sorted(manifest.get("files", {}).items()):
        path = data_root() / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != entry.get("sha256"):
            changed.append(rel)
        else:
            ok.append(rel)
    return {"ok": ok, "changed": changed, "missing": missing}

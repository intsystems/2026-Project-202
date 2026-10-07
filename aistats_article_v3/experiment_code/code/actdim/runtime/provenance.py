"""What produced a result file.

Every experiment writes one ``provenance.json`` beside its outputs. It records the
commit, the resolved configuration, the seeds, the device, the library versions and a
checksum of every file written. The old tree recorded none of this, and paid for it: two
result files were committed that their own scripts could no longer reproduce, and nothing
in the file said which code had written it.

The record is a plain dict on disk, so it stays readable without this package.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .. import __version__

SCHEMA = 1


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Checksum a file without reading it whole."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str, repo: Optional[Path] = None) -> Optional[str]:
    root = repo or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(repo: Optional[Path] = None) -> Dict[str, Any]:
    """The commit the code was at, and whether the tree was dirty.

    A dirty tree is not an error -- it is normal while developing -- but a result
    generated from one cannot be tied to a commit, so the flag is recorded.
    """
    sha = _git("rev-parse", "HEAD", repo=repo)
    status = _git("status", "--porcelain", repo=repo)
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
    }


def library_versions() -> Dict[str, str]:
    """Versions of every library whose behaviour can move a number.

    scikit-learn is here because the digits system reads its bundled dataset and trains
    an L-BFGS model whose low bits are library-sensitive; torch because the training
    runs are float64 and CPU and CUDA disagree in the last bits.
    """
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "joblib", "torch"):
        try:
            module = __import__(name)
        except Exception:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


@dataclass
class Provenance:
    """The record written beside a run's outputs."""

    experiment: str
    schema: int = SCHEMA
    package_version: str = __version__
    started_utc: str = ""
    finished_utc: str = ""
    wall_seconds: float = 0.0
    status: str = "running"
    device: str = "cpu"
    jobs: int = 1
    fast: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    git: Dict[str, Any] = field(default_factory=git_state)
    libraries: Dict[str, str] = field(default_factory=library_versions)
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    notes: Dict[str, Any] = field(default_factory=dict)

    _t0: float = field(default_factory=time.time, repr=False)

    def start(self) -> "Provenance":
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._t0 = time.time()
        return self

    def finish(self, status: str = "ok") -> "Provenance":
        self.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.wall_seconds = round(time.time() - self._t0, 3)
        self.status = status
        return self

    def record_output(self, path: Path, root: Path, **extra: Any) -> None:
        rel = path.relative_to(root).as_posix()
        self.outputs[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size, **extra}

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("_t0", None)
        return out

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

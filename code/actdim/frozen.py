"""The two frozen estimator configurations, loaded from disk rather than written in code.

The article uses two, one for each range of the active dimension. Each was selected once, on
data withheld from every later experiment, and then frozen; each is stored with the
copied unchanged from the calibration runs that produced them, so that the configuration a
result was computed at is a file with a checksum and not a literal somebody could edit.

    eight-direction   E_max 20, tau 4, m 20, autocorrelation Theiler, 8000 / 2000
    twenty-direction  E_max 40, tau 16, m 20, autocorrelation Theiler, 8000 / 4000

**Three pipelines override the window geometry and no estimator field.** The division of a
record into windows changes; the scoring of a window does not. Appendix C states them and
this module implements them, because in the archived tree the rule was copy-pasted into two
experiment scripts and re-derived by eye in a third:

* *the frozen configuration itself* -- window 8000, stride 2000 (eight directions) or 4000
  (twenty directions), used as loaded.
* *the constructed systems* -- window 8000, stride ``max(500, (n - window) // 6)``, which is
  3000 on the 26 000-sample records and 3666 on the 30 000-sample ones where there is no
  burn-in. It keeps the number of windows near ten whatever window length was frozen.
* *the training logs* -- window ``min(8000, max(2000, n // 3))``, stride 1000. The logs are
  12 000 samples, so the frozen window would leave three positions; a third of the record
  gives a usable number. The realised window there is 4000 samples, which is the 39 990
  optimiser steps the article's figures span.

Anything computed under an override is at the frozen configuration as regards the estimator
and not as regards the stride, and a result that quotes one should say so.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .estimator.calibration import Calibration
from .estimator.config import EstimatorConfig
from .runtime.store import data_root, runs_root

#: The file each configuration lives in, and the experiment that wrote it. A frozen
#: configuration is an output of its calibration run like any other, so it lives with that
#: run's other outputs rather than in a directory of its own -- one file, one producer, one
#: manifest entry. The names are the ones the calibration wrote, so the files still diff
#: against the archived tree.
EIGHT_DIRECTION = "frozen_config.json"
TWENTY_DIRECTION = "frozen_k20.json"

PRODUCER: Dict[str, str] = {
    EIGHT_DIRECTION: "calib.e8",
    TWENTY_DIRECTION: "calib.e20",
}


def frozen_path(name: str) -> Path:
    """Where a frozen configuration is read from: its calibration run's output.

    ``runs/`` first, so a fresh calibration is picked up without promoting it; then the
    tracked ``data/`` copy, which is what a clone has and what the article was written
    from.
    """
    experiment = PRODUCER.get(name)
    if experiment is None:
        raise KeyError(f"no such frozen configuration: {name!r}. "
                       f"Known: {', '.join(sorted(PRODUCER))}")
    for root in (runs_root(), data_root()):
        path = root / experiment / name
        if path.exists():
            return path
    return data_root() / experiment / name


def frozen_dir() -> Path:
    """Retained for callers that only want somewhere to look; prefer ``frozen_path``."""
    return data_root() / PRODUCER[EIGHT_DIRECTION]


def _read(name: str) -> Dict[str, Any]:
    path = frozen_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"the frozen configuration {name!r} is missing from {path.parent}.\n"
            f"It is tracked, so a fresh clone has it; otherwise produce it with\n"
            f"  python -m actdim run {PRODUCER[name]}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load(name: str = EIGHT_DIRECTION, **overrides: Any) -> EstimatorConfig:
    """One frozen configuration, with any window-geometry override applied.

    Overrides are applied here rather than by the caller so that they appear in the resolved
    configuration a run records, instead of being lost between the file and the estimate.
    """
    config = _read(name)["config"]
    return EstimatorConfig.from_dict(config).replace(**overrides)


def eight_direction(**overrides: Any) -> EstimatorConfig:
    """The configuration for active dimensions up to about eight."""
    return load(EIGHT_DIRECTION, **overrides)


def twenty_direction(**overrides: Any) -> EstimatorConfig:
    """The configuration for active dimensions up to twenty."""
    return load(TWENTY_DIRECTION, **overrides)


def selection(name: str = EIGHT_DIRECTION) -> Dict[str, Any]:
    """What the configuration was selected on: the seeds, the ranks, the score.

    Kept beside the configuration because a frozen setting is only meaningful with the split
    it was frozen on, and the two were recorded in the same file for that reason.
    """
    record = dict(_read(name))
    record.pop("config", None)
    record.pop("isotonic", None)
    return record


def calibration(observer: str, name: str = EIGHT_DIRECTION) -> Calibration:
    """The stored monotone map for one observer, rebuilt from its knots.

    Rebuilt, not refitted: the map belongs to the selection split, and refitting it on
    whatever data is at hand is how a held-out error becomes a training error.
    """
    maps = _read(name).get("isotonic", {})
    if observer not in maps:
        raise KeyError(
            f"{name} carries no calibration for {observer!r}. "
            f"It has: {', '.join(sorted(maps)) or 'none'}"
        )
    knots = maps[observer]
    return Calibration.from_points(knots["x"], knots["y"])


# -- the window-geometry overrides ---------------------------------------------


def constructed_geometry(cfg: EstimatorConfig, n: int) -> EstimatorConfig:
    """The stride the constructed-system pipelines use: about ten windows per record."""
    return cfg.replace(stride=max(500, (n - cfg.window) // 6))


def training_log_geometry(cfg: EstimatorConfig, n: int) -> EstimatorConfig:
    """The window and stride the training-log pipeline uses: a third of the record, by 1000."""
    return cfg.replace(window=min(cfg.window, max(2000, n // 3)), stride=1000)


def geometry_table() -> Tuple[Tuple[str, str, str], ...]:
    """Appendix C's table of window geometries, for a run to record or a test to check."""
    return (
        ("frozen configuration", "8000", "2000 (eight) / 4000 (twenty)"),
        ("constructed systems", "8000", "3000, or 3666"),
        ("training logs", "a third of the record", "1000"),
    )


def available(directory: Optional[Path] = None) -> Tuple[str, ...]:
    """The frozen configuration files present, for an error message worth reading."""
    root = directory or frozen_dir()
    return tuple(sorted(p.name for p in root.glob("*.json"))) if root.is_dir() else ()

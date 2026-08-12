"""What every analysis of a training log needs, in one place.

Section 7 reads training logs from four campaigns, and in the archived tree each script
that read one carried its own copy of the same four things: the set of runs, the window
geometry the article applies to a log, the interval a fall is measured over, and the fall
statistic itself. ``e9_matched_window.py``, ``e9_analyse.py`` and ``e10_surrogate.py``
each had all four, and they had already drifted -- one of them paired a surrogate's
values with the observed trace's window grid, which crashed on three of four fresh
invocations and silently mis-aligned the statistic on the fourth. A single copy is what
makes that class of drift impossible rather than unlikely.

Nothing here writes a file. The experiment modules decide what is stored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import frozen
from ..estimator.config import EstimatorConfig
# One implementation of each, imported rather than restated: the control pairing and the
# milestone rule are the same ones the direct measurement of section 7.2 uses, and the two
# analyses would stop being comparable the moment either had a second definition.
from ..sketch.analysis import CONTROL_REFERENCE, detrend as _detrend_columns, milestones
from ..training.runs import EXTENDED_RUNS as _EXTENDED, SKETCHED_RUNS as _SKETCHED

__all__ = [
    "CONTROL_OF", "GENERALISING", "MATCHED_WINDOW", "PERCEPTRON_ARITH", "PERCEPTRON_POLY",
    "PERCEPTRON_PROBE_COLUMNS", "POST", "PRE", "TRANSFORMER_EXTENDED",
    "TRANSFORMER_LOG_COLUMNS", "TRANSFORMER_SKETCHED",
    "article_geometry", "depth", "depth_profile", "detrend_series", "find_log",
    "first_sustained", "floor_offset", "load_log", "log_candidates", "log_stride",
    "matched_centres", "matched_windows", "milestone_map", "milestones",
    "require_sketch", "sketch_candidates", "transition_of",
]


# -- the runs -------------------------------------------------------------------
#
# Sorted, not in registration order, because every archived table was written by a
# ``sorted(glob(...))`` and a regenerated table has to line up with it row for row before
# `actdim diff` can say anything. Where a run set exists in the run registry it is taken
# from there; the polynomial labels are the one set that cannot be, and the reason is at
# `log_candidates`.

TRANSFORMER_SKETCHED: Tuple[str, ...] = tuple(sorted(_SKETCHED))
"""The six runs whose trajectory was stored: section 7.2 and the matched window of 7.3."""

TRANSFORMER_EXTENDED: Tuple[str, ...] = tuple(sorted(_EXTENDED))
"""The seven 120,000-step reruns: the diagnostics of 7.1 and the outcomes of appendix G."""

PERCEPTRON_ARITH: Tuple[str, ...] = ("a_add", "a_mul", "x_mix_quad", "x_no_grok")
"""The four arithmetic perceptron runs ``fig_map`` and ``fig_pairs`` draw.

Declared, not discovered. ``dimension_probe.py`` globbed every ``*_train.csv`` beside it,
which is seven runs in that directory, and the committed table covers these four: the
figure's legend reads "perceptron, full batch (10)" and a re-run of the archived command
would have made it 13. See docs/errata.md item 8.
"""

PERCEPTRON_POLY: Tuple[str, ...] = ("g_p1_p97", "g_p1x_p97", "g_p2_p97", "g_p2x_p97",
                                    "g_p3_p97", "g_p3x_p97")
"""The six polynomial runs at p = 97: three learnable and their three perturbed twins."""

PERCEPTRON_PROBE_COLUMNS: Tuple[str, ...] = ("train_loss", "weight_norm")
"""The two observers the committed probe covers, against the script's default of three.

The third was ``val_loss``, which on these runs is the quantity generalisation is defined
by; adding it back would put a circular observer into ``fig_map`` without saying so.
"""

TRANSFORMER_LOG_COLUMNS: Tuple[str, ...] = ("weight_norm", "train_loss", "val_loss",
                                            "train_acc", "val_acc")
"""Every 1-D column of a transformer log, in the order the archived atlas wrote them."""

CONTROL_OF: Mapping[str, str] = CONTROL_REFERENCE
"""Which run's generalisation step defines the window a control is measured in.

A control never generalises, so it has no window of its own, and measured over its whole
budget its fall is not comparable with a generalising run's.
"""

GENERALISING: Tuple[str, ...] = tuple(r for r in TRANSFORMER_SKETCHED
                                      if r not in CONTROL_OF)
"""The four sketched runs that generalise. The other two are the controls."""


# -- finding a log, and a sketch -------------------------------------------------


def log_candidates(run: str) -> Tuple[str, ...]:
    """The file names one run's training log may carry.

    The polynomial runs are the reason there is more than one. Every archived table, the
    article and ``fig_pairs`` call them ``g_p1_p97``; the port's registry keys them
    ``g_p1`` and carries the modulus in the configuration, so its trainer writes
    ``g_p1_train.csv``. Both names are accepted and the label written into the tables is
    the article's, so a regenerated table still lines up with the published one.
    """
    names = [f"{run}_train.csv"]
    if run.endswith("_p97"):
        names.append(f"{run[:-len('_p97')]}_train.csv")
    return tuple(names)


def sketch_candidates(run: str) -> Tuple[str, ...]:
    """The file names one run's trajectory sketch may carry.

    ``<run>_sketch.npz`` is what this package's trainers write. ``<run>_rank.npz`` is what
    the archived campaign wrote, and is accepted so that a sketch recovered from an old
    machine can be analysed without being renamed by hand.
    """
    return (f"{run}_sketch.npz", f"{run}_rank.npz")


def find_log(ctx: Any, experiment: str, run: str) -> Path:
    """The training log of one run, resolved through the upstream experiment that wrote it.

    Never a literal path: a missing log is reported as a missing prerequisite, naming the
    command that produces it, rather than as a file-not-found from inside a worker.
    """
    return _first_input(ctx, experiment, log_candidates(run),
                        what=f"the training log of {run!r}")


def require_sketch(ctx: Any, experiment: str, run: str) -> Path:
    """The trajectory sketch of one run, or a refusal that says what to run.

    The archived sketches behind appendix H's windows and appendix J's window-length sweep
    were never kept (docs/errata.md items 15 and 16), and the scripts that consumed them
    exited quietly when they found none -- which is how a table comes to be committed that
    its own command produces empty. This raises instead, and names the GPU campaign that
    rebuilds the input.
    """
    try:
        return _first_input(ctx, experiment, sketch_candidates(run),
                            what=f"the trajectory sketch of {run!r}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{ctx.experiment} needs the trajectory sketch of {run!r}, and no copy of it "
            f"exists: the archived sketches were not kept, so this table cannot be "
            f"rebuilt from the repository at any cost (docs/errata.md items 15 and 16).\n"
            f"Re-train the runs that produce it, on a GPU:\n"
            f"  python -m actdim run {experiment}\n"
            f"then:\n"
            f"  python -m actdim run {ctx.experiment}"
        ) from None


def _first_input(ctx: Any, experiment: str, names: Sequence[str], what: str) -> Path:
    errors: List[str] = []
    for name in names:
        try:
            return ctx.input(experiment, name)
        except FileNotFoundError as error:
            errors.append(str(error))
    raise FileNotFoundError(
        f"{ctx.experiment} needs {what} from {experiment!r}, under one of "
        f"{', '.join(names)}.\nRun it first:  python -m actdim run {experiment}"
    )


def load_log(path: Any) -> pd.DataFrame:
    """One training log, as a frame. The one place a log is read from disk."""
    return pd.read_csv(Path(path))


def log_stride(frame: pd.DataFrame) -> int:
    """Optimiser steps between logged rows, measured from the log rather than assumed.

    The transformer runs log every 10 steps and the ``S_5`` pair every 5, and an analysis
    that assumes one of the two silently doubles or halves every window it cuts. The
    median is taken because a log that was resumed can carry one irregular gap.
    """
    step = np.asarray(frame["step"], dtype=np.float64)
    if len(step) < 2:
        return 1
    return int(np.median(np.diff(step)))


# -- milestones ------------------------------------------------------------------


def milestone_map(path: Any) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """``{run: (t_mem, t_gen)}`` from a ``rank_milestones.json``.

    The archived matched-window analysis carried those numbers as two literal dictionaries
    in two files, so editing a milestone meant editing it twice and a log re-trained under
    a different seed would have been scored against the old run's transition. They are
    read from the file the direct measurement wrote instead, which is the same file the
    figures align on.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    for record in records:
        out[str(record["run"])] = (_maybe_int(record.get("t_mem")),
                                   _maybe_int(record.get("t_gen")))
    return out


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else int(value)


def transition_of(run: str, milestones_by_run: Mapping[str, Tuple[Any, Any]]) -> float:
    """The generalisation step a run is measured against, its match's where it has none."""
    own = milestones_by_run.get(run, (None, None))[1]
    if own is not None:
        return float(own)
    matched = CONTROL_OF.get(run)
    if matched is None:
        return float("nan")
    other = milestones_by_run.get(matched, (None, None))[1]
    return float("nan") if other is None else float(other)


def first_sustained(steps: Any, values: Any, threshold: float = 0.95,
                    window: int = 5) -> Optional[int]:
    """The step from which a rolling accuracy stays above the threshold to the end.

    Appendix G's rule, and deliberately not :func:`milestones`. The 120,000-step reruns
    are asked whether a run *ended up* generalising, so a run that touches the threshold
    and falls back has not; :func:`milestones` reports the first crossing, which is the
    right question for a run whose transition is already known to have happened. The two
    agree on every run that generalises and disagree on exactly the runs the appendix
    exists to reclassify. Returns ``None`` when the threshold is never held.
    """
    steps = np.asarray(steps)
    smoothed = pd.Series(np.asarray(values, dtype=np.float64)).rolling(
        window, min_periods=1, center=True).mean().to_numpy()
    above = smoothed >= threshold
    index: Optional[int] = None
    for j in range(len(above) - 1, -1, -1):
        if not above[j]:
            break
        index = j
    return None if index is None else int(steps[index])


# -- the window geometry the article applies to a log ----------------------------


def article_geometry(n_samples: int, base: Optional[EstimatorConfig] = None,
                     ) -> EstimatorConfig:
    """The frozen configuration with the training-log window geometry applied.

    Window ``min(8000, max(2000, n // 3))``, stride 1000, and no estimator field touched.
    The logs are 12,000 samples, so the frozen 8,000-sample window would leave three
    positions; a third of the record gives a usable number, and the realised window on the
    extended reruns is 4,000 samples, the 39,990 optimiser steps the figures span.

    Anything computed under this override is at the frozen configuration as regards the
    estimator and not as regards the stride, and a result quoting one should say so.
    """
    return frozen.training_log_geometry(base or frozen.eight_direction(), int(n_samples))


# -- the fall statistic ----------------------------------------------------------

PRE: Tuple[int, int] = (-3000, -1000)
"""The interval, relative to the transition, the level a fall is measured from is taken over."""

POST: Tuple[int, int] = (-1000, 2000)
"""The interval the floor is looked for in.

It is where the direct measurement's own minimum falls in all four generalising runs, and
it is fixed before any estimate is read: choosing it on the outcome is the failure
requirement 2 exists to prevent.
"""


def depth(t: Any, y: Any, centre: float, pre: Tuple[int, int] = PRE,
          post: Tuple[int, int] = POST) -> float:
    """``D(c)``: the pre-transition level over the post-transition floor.

    The grid and the values are passed together and their lengths are checked, because the
    archived surrogate control passed the observed trace's grid with a surrogate's values.
    A surrogate that produced one constant window returned a shorter array, so the call
    either raised or -- when the lengths happened to match because a different window had
    been dropped -- compared two traces sampled at different instants.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError(
            f"grid {t.shape} and values {y.shape} disagree: a trace has been paired with "
            f"another trace's window grid")
    if not np.isfinite(centre):
        return float("nan")
    before = (t >= centre + pre[0]) & (t <= centre + pre[1])
    after = (t >= centre + post[0]) & (t <= centre + post[1])
    if before.sum() < 3 or after.sum() < 3:
        return float("nan")
    # Checked before reducing, because the median and the minimum of an all-NaN selection
    # are a RuntimeWarning apiece, and under this package's test settings a warning is an
    # error. The archived tree let three of them reach a committed table.
    level = float(np.nanmedian(y[before])) if np.isfinite(y[before]).any() else float("nan")
    floor = float(np.nanmin(y[after])) if np.isfinite(y[after]).any() else float("nan")
    if not np.isfinite(level) or not np.isfinite(floor) or floor <= 0:
        return float("nan")
    return level / floor


def depth_profile(t: Any, y: Any, centres: Any, pre: Tuple[int, int] = PRE,
                  post: Tuple[int, int] = POST) -> np.ndarray:
    """``D`` at every candidate centre, which is what a depth is read against.

    A fall that is deep but no deeper than the rest of the run is not a signature, and the
    only way to say so is to compute the same statistic everywhere it is defined.
    """
    return np.array([depth(t, y, float(c), pre, post) for c in np.asarray(centres)],
                    dtype=np.float64)


def floor_offset(t: Any, y: Any, centre: float,
                 post: Tuple[int, int] = POST) -> float:
    """Where the post-transition minimum falls, relative to the transition."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(centre):
        return float("nan")
    after = (t >= centre + post[0]) & (t <= centre + post[1]) & np.isfinite(y)
    if after.sum() < 3:
        return float("nan")
    return float(t[after][int(np.argmin(y[after]))] - centre)


# -- the window matched to the transition ----------------------------------------

MATCHED_WINDOW = 60
"""Samples per window in the matched-window re-run of section 7.3.

Samples and not optimiser steps: the modular runs log every 10 steps and the ``S_5`` pair
every 5, so 60 samples is 600 steps in one and 300 in the other, which is what matches
each run to within two per cent of the width of the direct measurement's own window.
Fixing the step count instead would mismatch ``S_5`` by a factor of two.
"""


def detrend_series(v: Any) -> np.ndarray:
    """Remove the least-squares line from a scalar window.

    The same removal :math:`\\PR^{det}` performs per coordinate, so the two statistics are
    asked the same question. It is not the same amount of removal: on a 1024-dimensional
    sketch it takes one direction of many and on a scalar it takes most of the series,
    which is why appendix L reports the detrended arm beside the surrogate test rather
    than instead of it.

    Fitted in closed form through :mod:`actdim.sketch.analysis`, not with ``polyfit``,
    which raises from inside a worker on a window holding a non-finite sample.
    """
    v = np.asarray(v, dtype=np.float64)
    return _detrend_columns(v.reshape(-1, 1)).ravel()


def matched_centres(windows: pd.DataFrame, run: str) -> np.ndarray:
    """The midpoints of one run's direct-measurement windows, in optimiser steps.

    The grid is the direct measurement's own, so that the estimate and the participation
    ratio are two statistics of the same run at the same instants and "do they fall
    together" is a question about paired samples rather than about two plots.
    """
    group = windows[windows["run"] == run]
    mids = 0.5 * (group["right_step"].to_numpy(float) + group["left_step"].to_numpy(float))
    return np.sort(mids)


def matched_windows(x: Any, stride: int, centres: Any, window: int = MATCHED_WINDOW,
                    detrend: bool = False) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield ``(centre, values)`` for every usable window centred on the given grid.

    A window that runs off either end of the record, holds a non-finite sample, or has no
    spread is skipped, so the trace comes back on the grid it actually has. Every caller
    gets that grid with the values, which is the alignment defect of ``e10_surrogate``
    made impossible rather than documented.
    """
    x = np.asarray(x, dtype=np.float64)
    half = window // 2
    for centre in np.asarray(centres, dtype=np.float64):
        index = int(round(centre / stride))
        a, b = index - half, index + window - half
        if a < 0 or b > len(x):
            continue
        values = x[a:b]
        if not np.isfinite(values).all() or values.std() <= 1e-12:
            continue
        if detrend:
            values = detrend_series(values)
            if values.std() <= 1e-12:
                continue
        yield float(centre), values

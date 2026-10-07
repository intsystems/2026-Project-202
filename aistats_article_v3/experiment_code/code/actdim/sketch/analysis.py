"""What the sketched trajectory is asked, once training is over.

Nothing is decided while a run is training. The sketch is recorded and every statistic --
participation ratio over sliding windows, of positions or of increments, detrended or not,
at several smoothing scales -- is computed here, where it can be changed without retraining.

The statistics, per window and in each of the two spaces:

``PR_pos``      participation ratio of the window's position covariance. Dominated by
                whatever direction the trajectory is drifting along.
``PR_pos_det``  the same after removing a per-window linear trend. A steady drift is rank
                one and would otherwise mask everything else, so this is the honest form of
                the question, and it is one of the two the article reports.
``PR_step``     participation ratio of the increment covariance: how many directions the
                optimiser is exploring right now, trend-free by construction.
``PR_step<m>``  the same on increments block-averaged over ``m`` logged steps, which
                suppresses the mini-batch noise floor. ``PR_step5`` is the second statistic
                the article reports.

Two nulls are carried beside them, because a participation ratio can fall for reasons that
have nothing to do with rank: ``move``, the total displacement in the window, and ``pnorm``,
the parameter norm, which weight decay drives mechanically. The article's claim rests on the
rank recovering where the displacement does not.

Windows are labelled by their **centre**. Labelling by the right edge, as the first version
did, delays every feature by up to a whole window, which is enough to move a dip past the
event it belongs to.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -- window geometry -----------------------------------------------------------
# The archived tree ran two passes with different window lengths and recorded the fine
# one -- the pass the article reports -- only in the prose of a report file. Neither the
# script's defaults nor any Makefile mentioned it, so the published numbers could not be
# regenerated without reading the prose. Both passes are named here instead.


@dataclass(frozen=True)
class WindowGeometry:
    """How long a window is, how far apart windows sit, and what is averaged inside one.

    ``window`` and ``stride`` are counted in *logged rows*, not optimiser steps, so one
    geometry means the same thing on a run logged every 10 steps and on one logged every 5.
    At the article's geometry a window is sixty rows: 600 optimiser steps on the modular
    runs and 300 on ``S_5``.
    """

    name: str
    window: int
    stride: int
    smooth: Tuple[int, ...] = (5, 20)

    def usable_smoothing(self) -> Tuple[int, ...]:
        """The block sizes that fit, which is not all of them.

        A window of ``w`` rows yields ``w - 1`` increments, so a block size of ``m`` leaves
        ``(w - 1) // m`` blocks, and a participation ratio needs three points. At the
        article's sixty-row window a block of twenty leaves two, and the statistic is
        undefined. The archived pass emitted those columns anyway: eight all-NaN columns in
        the published window table, produced with a "Mean of empty slice" warning per
        window. Undefined columns are now simply not emitted.
        """
        return tuple(m for m in self.smooth if m > 1 and (self.window - 1) // m >= 3)

    def dropped_smoothing(self) -> Tuple[int, ...]:
        """The block sizes this geometry cannot support, for the caller to record."""
        return tuple(m for m in self.smooth if m not in self.usable_smoothing())

    def columns(self) -> List[str]:
        """Every column :func:`sliding` will produce, in order."""
        names = ["run", "left_step", "right_step", "centre"]
        for prefix in ("", "fn_"):
            for stat in ["pos", "pos_det", "step"] + [f"step{m}" for m in self.usable_smoothing()]:
                names += [f"{prefix}PR_{stat}", f"{prefix}PR_{stat}_sketchsd"]
        return names + ["move", "pnorm"]

    def stamp(self, name: str) -> str:
        """A file name stamped with this geometry.

        The archived collapse script took its results directory from the command line but
        pinned its figure directory beside the code, so the fine pass overwrote the coarse
        pass's figure and the coarse figure has been a copy of the fine one ever since. Two
        passes must not be able to write the same name; stamping is how they cannot.
        """
        stem, dot, suffix = name.partition(".")
        return f"{stem}_{self.name}{dot}{suffix}"


COARSE = WindowGeometry(name="coarse", window=200, stride=25)
"""The first pass: 2,000-step windows on the modular runs, 1,000 on ``S_5``. Superseded."""

FINE = WindowGeometry(name="fine", window=60, stride=10)
"""Sixty logged rows, which is 600 optimiser steps on the modular runs and 300 on ``S_5``."""

ARTICLE = FINE
"""The geometry section 7.2 and appendix I report. The coarse pass is kept for comparison."""


def output_paths(results_dir: Any, geometry: WindowGeometry) -> Dict[str, Path]:
    """Where one analysis pass's outputs go, derived from where its inputs came from.

    Every path is built from the results directory the caller passed and stamped with the
    geometry, so that pointing the analysis at a second directory moves all of its outputs
    and no pass can overwrite another's figure.
    """
    root = Path(results_dir)
    return {
        "windows": root / geometry.stamp("rank_windows.csv"),
        "summary": root / geometry.stamp("rank_summary.csv"),
        "collapse": root / geometry.stamp("rank_collapse.csv"),
        "controls": root / geometry.stamp("rank_collapse_controls.csv"),
        "controls_aligned": root / geometry.stamp("rank_collapse_controls_aligned.csv"),
        "figure": root / "figures" / geometry.stamp("rank_collapse.png"),
    }


# -- the statistics ------------------------------------------------------------


def pr(X: Any, tol: float = 1e-12) -> float:
    """Participation ratio ``(sum lambda)^2 / sum lambda^2`` of the row cloud of ``X``.

    One for a cloud along a single direction, ``r`` for ``r`` equally weighted ones. Returns
    NaN below three rows, where a covariance has no shape to report, and on a cloud that
    does not move.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return float("nan")
    s = np.linalg.svd(X, compute_uv=False) ** 2
    total = s.sum()
    return float(total * total / (s * s).sum()) if total > tol else float("nan")


def detrend(X: Any) -> np.ndarray:
    """Remove a per-coordinate linear trend and the mean.

    A steady drift is a rank-one contribution that would otherwise dominate every window.
    """
    X = np.asarray(X, dtype=float)
    t = np.arange(len(X), dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    beta = (t[:, None] * X).sum(0) / (t @ t)
    return X - np.outer(t, beta) - X.mean(0, keepdims=True)


def block_mean(X: Any, m: int) -> np.ndarray:
    """Average consecutive blocks of ``m`` rows, dropping the incomplete tail."""
    X = np.asarray(X, dtype=float)
    n = (len(X) // m) * m
    return X[:n].reshape(-1, m, X.shape[1]).mean(1) if n >= m else X[:0]


def _mean_and_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and spread over the hash families, skipping undefined values without warning.

    ``np.nanmean`` over an all-NaN slice is the "Mean of empty slice" warning the archived
    analysis emitted once per window. The filtering is explicit here, and an all-undefined
    statistic returns NaN rather than raising.
    """
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def milestones(log: Any, threshold: float = 0.95,
               sustain: int = 1) -> Tuple[Optional[int], Optional[int]]:
    """Memorisation and generalisation from a run's own log.

    ``sustain=1`` is appendix O's rule for every row but the extended reruns: the first
    logged step at which the training and the validation accuracy respectively reach the
    threshold, with no persistence requirement. ``sustain=5`` is the stricter rule the
    extended reruns use, a centred rolling mean over five logged rows. The two put
    memorisation within seventy steps of each other on every run in the table and differ by
    1,080 steps on one generalisation step.
    """
    frame = log if isinstance(log, pd.DataFrame) else pd.DataFrame(log)
    out: List[Optional[int]] = []
    for column in ("train_acc", "val_acc"):
        series = frame[column]
        if sustain > 1:
            series = series.rolling(sustain, min_periods=1, center=True).mean()
        hit = np.flatnonzero(series.to_numpy() >= threshold)
        out.append(int(frame["step"].iloc[hit[0]]) if len(hit) else None)
    return out[0], out[1]


# -- the sliding pass ----------------------------------------------------------


def sliding(sketch: Mapping[str, Any], geometry: WindowGeometry = ARTICLE,
            run: Optional[str] = None) -> pd.DataFrame:
    """Every statistic over every window of one run's sketch.

    ``sketch`` is what ``TrajectoryRecorder.arrays()`` returns, or a loaded ``.npz`` of it:
    ``step``, ``z`` and ``zf`` of shape ``(T, n_sketch, dim)``, and the scalars
    ``param_step`` and ``param_norm``.
    """
    step = np.asarray(sketch["step"])
    z, zf = np.asarray(sketch["z"]), np.asarray(sketch["zf"])
    move = np.asarray(sketch["param_step"], dtype=float)
    pnorm = np.asarray(sketch["param_norm"], dtype=float)
    smooth = geometry.usable_smoothing()

    rows: List[Dict[str, Any]] = []
    for a in range(0, len(step) - geometry.window + 1, geometry.stride):
        b = a + geometry.window
        record: Dict[str, Any] = {"left_step": int(step[a]), "right_step": int(step[b - 1])}
        record["centre"] = (record["left_step"] + record["right_step"]) / 2.0
        for prefix, arr in (("", z), ("fn_", zf)):
            per_sketch: Dict[str, List[float]] = {
                key: [] for key in ["pos", "pos_det", "step"] + [f"step{m}" for m in smooth]}
            for s in range(arr.shape[1]):
                W = arr[a:b, s, :]
                per_sketch["pos"].append(pr(W))
                per_sketch["pos_det"].append(pr(detrend(W)))
                D = np.diff(W, axis=0)
                per_sketch["step"].append(pr(D))
                for m in smooth:
                    per_sketch[f"step{m}"].append(pr(block_mean(D, m)))
            for key, values in per_sketch.items():
                mean, sd = _mean_and_sd(values)
                record[f"{prefix}PR_{key}"] = mean
                record[f"{prefix}PR_{key}_sketchsd"] = sd
        window_move = move[a:b]
        window_move = window_move[np.isfinite(window_move)]
        record["move"] = float(window_move.sum()) if len(window_move) else float("nan")
        window_norm = pnorm[a:b][np.isfinite(pnorm[a:b])]
        record["pnorm"] = float(window_norm.mean()) if len(window_norm) else float("nan")
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.insert(0, "run", run if run is not None else "run")
    return frame.reindex(columns=geometry.columns())


def _scalar(sketch: Mapping[str, Any], key: str) -> Optional[int]:
    """One stored scalar, or None when it was kept elsewhere.

    A recorder may carry its metadata in the same file as the arrays or in the run's
    provenance record. The summary reports what it finds rather than insisting on one of
    the two, so that both architectures' sketches can be summarised by this function.
    """
    if key not in sketch:
        return None
    return int(np.asarray(sketch[key]))


def summarise(run: str, log: Any, sketch: Mapping[str, Any], windows: pd.DataFrame,
              geometry: WindowGeometry = ARTICLE) -> Dict[str, Any]:
    """One row of the per-run summary: size, geometry, milestones, where it ended."""
    t_mem, t_gen = milestones(log)
    frame = log if isinstance(log, pd.DataFrame) else pd.DataFrame(log)
    return {
        "run": run,
        "n_rows": int(len(frame)),
        "n_params": _scalar(sketch, "n_params"),
        "dim": _scalar(sketch, "dim"),
        "n_sketch": _scalar(sketch, "n_sketch"),
        "geometry": geometry.name,
        "window": geometry.window,
        "stride": geometry.stride,
        "n_windows": int(len(windows)),
        "t_mem": t_mem,
        "t_gen": t_gen,
        "final_val_acc": float(frame["val_acc"].iloc[-1]),
    }


# -- the collapse ---------------------------------------------------------------
# The question section 7.2 settles is whether the participation ratio falls at
# generalisation, and whether that fall is anything more than the trajectory slowing down.
# The discriminating fact is that the rank recovers where the displacement does not, so
# every statistic below is reported beside ``move``.

STATS = ("fn_PR_pos_det", "fn_PR_step5", "PR_pos_det", "PR_step5")
"""The four the article reports, of the ten each pass computes."""

PLATEAU = 4000
"""Steps before generalisation the plateau level is taken over."""

RECOVERY = (3000, 6000)
"""Steps after generalisation the recovered level is taken over."""

CONTROL_REFERENCE = {"mod_wd0": "mod_wd1", "s5_wd0": "s5_wd1"}
"""Which run's generalisation step defines the window a control is measured in.

A control never generalises, so it has no window of its own; measured over its whole budget
its depth is not comparable with a grokking run's. Pairing each control with the run it
matches is what makes the two numbers like for like.
"""


def collapse(windows: pd.DataFrame, milestones_by_run: Mapping[str, Tuple[Any, Any]],
             stats: Iterable[str] = STATS) -> pd.DataFrame:
    """Where the collapse is and how deep, per generalising run.

    ``plateau`` is the median over the 4,000 steps before generalisation, ``dip`` the
    deepest window centre within 4,000 either side, ``offset`` its distance from
    generalisation, ``recovered`` the median over 3,000 to 6,000 steps after, and ``depth``
    the ratio of plateau to dip.
    """
    columns = list(stats) + ["move"]
    rows = []
    for run, (_, t_gen) in sorted(milestones_by_run.items()):
        if t_gen is None or not np.isfinite(float(t_gen)):
            continue
        group = windows[windows.run == run].sort_values("centre")
        for stat in columns:
            pre = group[(group.centre >= t_gen - PLATEAU) & (group.centre < t_gen)][stat]
            near = group[(group.centre >= t_gen - PLATEAU) & (group.centre <= t_gen + PLATEAU)]
            post = group[(group.centre >= t_gen + RECOVERY[0])
                         & (group.centre <= t_gen + RECOVERY[1])][stat]
            if not len(near) or not len(pre):
                continue
            deepest = near[stat].idxmin()
            rows.append({
                "run": run, "stat": stat, "plateau": pre.median(), "dip": near[stat].min(),
                "offset": near.centre[deepest] - t_gen,
                "recovered": post.median() if len(post) else float("nan"),
                "depth": pre.median() / max(near[stat].min(), 1e-9),
            })
    return pd.DataFrame(rows)


def collapse_controls(windows: pd.DataFrame, milestones_by_run: Mapping[str, Tuple[Any, Any]],
                      stats: Iterable[str] = STATS) -> pd.DataFrame:
    """The deepest dip anywhere in a control's budget, and the level it ends at."""
    columns = list(stats) + ["move"]
    rows = []
    for run, (_, t_gen) in sorted(milestones_by_run.items()):
        if t_gen is not None and np.isfinite(float(t_gen)):
            continue
        group = windows[windows.run == run].sort_values("centre")
        if not len(group):
            continue
        early = group[group.centre < group.centre.quantile(0.2)]
        for stat in columns:
            rows.append({
                "run": run, "stat": stat, "early": early[stat].median(),
                "min": group[stat].min(), "offset": group.centre[group[stat].idxmin()],
                "end": group[stat].iloc[-8:].median(),
                "depth": early[stat].median() / max(group[stat].min(), 1e-9),
            })
    return pd.DataFrame(rows)


def collapse_controls_aligned(windows: pd.DataFrame,
                              milestones_by_run: Mapping[str, Tuple[Any, Any]],
                              reference: Mapping[str, str] = CONTROL_REFERENCE,
                              stats: Iterable[str] = STATS) -> pd.DataFrame:
    """The controls again, measured in the window their matched run defines."""
    columns = list(stats) + ["move"]
    rows = []
    for run, matched in sorted(reference.items()):
        if run not in set(windows.run) or matched not in milestones_by_run:
            continue
        t_gen = milestones_by_run[matched][1]
        if t_gen is None:
            continue
        group = windows[windows.run == run].sort_values("centre")
        for stat in columns:
            pre = group[(group.centre >= t_gen - PLATEAU) & (group.centre < t_gen)][stat]
            near = group[(group.centre >= t_gen - PLATEAU)
                         & (group.centre <= t_gen + PLATEAU)][stat]
            if not len(pre) or not len(near):
                continue
            rows.append({"run": run, "reference": matched, "stat": stat,
                         "plateau": pre.median(), "dip": near.min(),
                         "depth": pre.median() / max(near.min(), 1e-9)})
    return pd.DataFrame(rows)

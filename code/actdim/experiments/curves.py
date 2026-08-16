"""The scalar logs themselves, and the estimate along them.

Every other experiment reports the estimate as a number per run. The article's claim is
about a *log*, and a reader who has never seen one cannot tell whether the number came from
a rich signal or from noise. This reduces the stored twenty-direction trajectories to three
small tables a figure can draw:

* the standardised log at four ranks, decimated to a plottable length;
* the estimate along that same log, window by window, beside the roughness and the linear
  participation ratio that are the two things it might be confused with;
* the delay reconstruction of one window as a pair of coordinates, which is the object the
  neighbour search actually measures and which the article otherwise only describes.

Nothing is simulated here. The trajectories are read from ``calib.e20`` exactly as
``valid.theiler.cap`` reads them, so the logs a figure draws are the logs that experiment
scored and not a fresh draw that happens to look similar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..runtime import CPU, Context, experiment

#: Ranks 1, 3, 5 and 8 are the four the eight-direction configuration was *not* selected on
#: (appendix C, ``tab:frozen``), so a figure drawn on them is out of sample in rank. Seed 0
#: is likewise withheld there. The point of the figure is what the estimator reads, but
#: there is no reason to illustrate it on the data it was tuned on.
CURVE_RANKS: Tuple[int, ...] = (1, 3, 5, 8)
CURVE_SEED = 0

#: The parameter norm, which is also the observer the grokking application uses. Choosing
#: the best observer for a picture and the ordinary one for the results would flatter it.
CURVE_OBSERVER = "w_fro"

#: A window of 8000 samples on a 10000-sample record leaves three positions at the frozen
#: stride of 2000. The stride is the one field appendix C allows an analysis to override,
#: and 100 gives twenty-one overlapping windows: enough to show that the level is a property
#: of the record rather than of where the window happened to fall.
CURVE_STRIDE = 100

#: Decimation for the raw trace. 10000 points at 1.3 inches wide is ink, not a curve.
SERIES_STRIDE = 8

#: One window's reconstruction, thinned the same way, per shape.
SHAPE_POINTS = 1400

#: arm, rank, and the words the figure puts over the panel.
SHAPES: Tuple[Tuple[str, int, str], ...] = (
    ("qp", 1, "one phase"),
    ("qp", 2, "two phases"),
    ("gd", 1, "a transient"),
    ("batch_proj", 5, "mini-batch noise"),
)

_STEM = re.compile(r"^(?P<arm>.+)_r(?P<rank>\d+)_s(?P<seed>\d+)$")


def _trajectories(ctx: Context) -> Dict[Tuple[str, int, int], Path]:
    """Every stored twenty-direction trajectory, keyed by arm, rank and seed."""
    from ..runtime.store import is_plumbing_check

    upstream = ctx.input_dir("calib.e20")
    if is_plumbing_check(upstream) and not ctx.fast:
        raise ValueError(
            f"{upstream} was produced by a --fast run, so its trajectories are a "
            f"plumbing check.\nRun it for real first:  python -m actdim run calib.e20")
    directory = upstream / "trajectories"
    found: Dict[Tuple[str, int, int], Path] = {}
    for path in sorted(directory.glob("*.npz")) if directory.is_dir() else []:
        stem = _STEM.match(path.stem)
        if stem:
            found[(stem.group("arm"), int(stem.group("rank")),
                   int(stem.group("seed")))] = path
    if not found:
        raise FileNotFoundError(
            f"no stored trajectories under {directory}.\n"
            f"Run it first:  python -m actdim run calib.e20")
    return found


def _load(path: Path, observer: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        series = np.asarray(stored[f"log__{observer}"], dtype=float)
        info = json.loads(str(stored["info"]))
    return series, info


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    spread = x.std()
    return (x - x.mean()) / spread if spread > 0 else x - x.mean()


@experiment(
    id="valid.curves",
    title="The scalar log at four ranks, the estimate along it, and one reconstruction",
    paper=("fig:signal", "fig:shapes", "app:nulls"),
    device=CPU,
    minutes=4,
    needs=("calib.e20",),
    promotes=("curve_series.csv", "curve_windows.csv", "curve_shapes.csv"),
    tier=1,
    notes="Reads the stored trajectories and computes nothing new about them: the "
          "estimator, its configuration and its dither are the frozen eight-direction "
          "ones, with the stride overridden as appendix C permits and no other field "
          "touched.",
)
def curves(ctx: Context) -> None:
    import pandas as pd

    from .. import frozen as frozen_mod
    from ..estimator.companions import delay_participation_ratio, roughness
    from ..estimator.embedding import reconstruct
    from ..estimator.mle import estimate_from

    stored = _trajectories(ctx)
    cfg = frozen_mod.eight_direction(stride=CURVE_STRIDE)

    ranks = list(CURVE_RANKS)
    shapes = list(SHAPES)
    if ctx.fast:
        ranks, shapes = ranks[:2], shapes[:2]

    ctx.config(ranks=ranks, seed=CURVE_SEED, observer=CURVE_OBSERVER,
               configuration=cfg.tag(), stride=cfg.stride, window=cfg.window,
               shapes=[f"{arm}_r{rank}" for arm, rank, _ in shapes])

    series_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []

    for rank in ranks:
        key = ("qp", rank, CURVE_SEED)
        if key not in stored:
            raise FileNotFoundError(
                f"no recurrent trajectory at rank {rank}, seed {CURVE_SEED}; "
                f"calib.e20 appears to have been run with --fast")
        raw, info = _load(stored[key], CURVE_OBSERVER)
        truth = float(info["traj_pr"])

        z = _standardise(raw)
        for index in range(0, len(z), SERIES_STRIDE):
            series_rows.append({"r": rank, "truth": truth, "sample": index,
                                "z": float(z[index])})

        if len(raw) < cfg.window:
            raise ValueError(
                f"the stored record is {len(raw)} samples and the window is {cfg.window}")
        for start in range(0, len(raw) - cfg.window + 1, cfg.stride):
            piece = raw[start:start + cfg.window]
            rec = reconstruct(piece, cfg, seed=CURVE_SEED)
            got = estimate_from(rec, cfg)
            window_rows.append({
                "r": rank, "truth": truth,
                "centre": start + 0.5 * cfg.window,
                "MG": got.MG, "LB": got.LB, "degenerate": bool(got.degenerate),
                "roughness": roughness(piece),
                "PRdelay": delay_participation_ratio(rec.points) if rec.usable else np.nan,
            })

    shape_rows: List[Dict[str, Any]] = []
    for arm, rank, label in shapes:
        key = (arm, rank, CURVE_SEED)
        if key not in stored:
            continue
        raw, info = _load(stored[key], CURVE_OBSERVER)
        rec = reconstruct(raw[:cfg.window], cfg, seed=CURVE_SEED)
        if not rec.usable:
            continue
        points = np.asarray(rec.points, dtype=float)
        # The first two delay coordinates, which is the plane the article's delay vector is
        # built in. A principal-component view would be prettier and would no longer be the
        # coordinates the neighbour search runs in.
        thin = max(1, len(points) // SHAPE_POINTS)
        for index in range(0, len(points), thin):
            shape_rows.append({"arm": arm, "r": rank, "label": label,
                               "regime": info["mode"],
                               "x": float(points[index, 0]),
                               "y": float(points[index, 1])})

    ctx.store.table("curve_series.csv", pd.DataFrame(series_rows))
    ctx.store.table("curve_windows.csv", pd.DataFrame(window_rows))
    ctx.store.table("curve_shapes.csv", pd.DataFrame(shape_rows))
    ctx.note("windows", len(window_rows))
    ctx.note("shapes", len({(row["arm"], row["r"]) for row in shape_rows}))

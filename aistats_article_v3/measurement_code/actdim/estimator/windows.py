"""Scoring one window, and sliding over a record.

A window is scored once and yields every statistic, sharing one reconstruction and one
neighbour query. A record is divided into windows by ``window`` and ``stride`` alone; nothing
about how a window is scored changes between pipelines, which is why the article can say that
three of them override the geometry and no estimator field.

Sliding traces are labelled by the window's **right edge**. A detection lag is only meaningful
if the label is the last sample the value could have used; labelling by the centre attributes
a change to a moment half a window before the estimator could have seen it.

Each window is standardised on its own, which is the first line of appendix A applied to the
series the estimator is handed. The archived pipelines standardised the whole record before
slicing it, so their windows carried whatever spread they had. The two agree to about 1e-13
where the windows are stationary; they differ by up to 3e-8 on the decaying transient, whose
second window has a sixth of the spread of its first, because the dither is then a sixth as
large relative to the data. Per-window standardisation is the one that makes the dither and
the distance floor mean the same thing in every window of a record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .companions import companion_statistics, delay_participation_ratio
from .config import NEIGHBOUR_BASED, EstimatorConfig
from .embedding import reconstruct
from .mle import estimate_from


def statistic_names(cfg: EstimatorConfig) -> Tuple[str, ...]:
    """Every statistic a scored window carries, in a stable order."""
    return (("MG", "LB", "TwoNN", "PRdelay")
            + tuple(f"specPR{b}" for b in cfg.spectral_bins)
            + ("roughness", "acorr"))


def score(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> Dict[str, Any]:
    """Every statistic on one window, sharing one reconstruction.

    A window too short to embed, or holding a non-finite sample, has no statistic at all and
    every field comes back NaN. A window with no spread keeps the companion statistics, which
    are defined there, and loses only what the neighbour search would have produced.
    """
    x = np.asarray(x, dtype=np.float64)
    names = statistic_names(cfg)
    out: Dict[str, Any] = {name: float("nan") for name in names}
    out.update(degenerate=True, frac_floor=float("nan"), frac_sumfloor=float("nan"),
               tau_used=float("nan"), theiler_used=float("nan"))

    rec = reconstruct(x, cfg, seed)
    if rec.reason in ("short", "nonfinite"):
        return out

    out.update(companion_statistics(x, cfg))
    if rec.reason == "flat":
        return out

    out["tau_used"] = float(rec.tau)
    out["theiler_used"] = float(rec.theiler)
    if rec.points.size:
        out["PRdelay"] = delay_participation_ratio(rec.points)
    out.update(estimate_from(rec, cfg).as_dict())
    return out


def window_starts(n: int, cfg: EstimatorConfig) -> List[int]:
    """Where each window begins. Empty when the record is shorter than one window."""
    return list(range(0, n - cfg.window + 1, cfg.stride))


def sliding(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0
            ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The trace of every statistic over a record. Returns ``(right_edges, {name: array})``.

    The value at ``right_edges[i]`` uses samples ``right_edges[i] - window + 1`` to
    ``right_edges[i]`` and nothing after them.
    """
    x = np.asarray(x, dtype=np.float64)
    starts = window_starts(len(x), cfg)
    records = [score(x[s:s + cfg.window], cfg, seed) for s in starts]
    right = np.array([s + cfg.window - 1 for s in starts], dtype=int)
    keys = list(statistic_names(cfg)) + ["degenerate", "frac_floor", "frac_sumfloor",
                                         "tau_used", "theiler_used"]
    traces = {k: np.array([float(r[k]) for r in records], dtype=float) for k in keys}
    return right, traces


def summarise(x: np.ndarray, cfg: EstimatorConfig, seed: int = 0) -> Dict[str, Any]:
    """Median and spread of each statistic over the sliding windows of one record.

    Only the neighbour-based estimates are dropped on a degenerate window. This fixes a
    defect in the archived summariser (``active_dimension/mg.py:241-245``), which filtered
    every statistic by the flag, including the roughness, the autocorrelation time and the
    spectral participation ratios. None of those touches the neighbour search and all of them
    are valid on a degenerate window; filtering them thinned the null columns exactly on the
    arms where the nulls matter most -- the transient arm reports every window degenerate at
    some settings, and its nulls were being computed and then thrown away.

    An empty record -- one shorter than a single window -- returns NaNs and a window count of
    zero. The archived version took the mean of an empty array here and emitted a
    RuntimeWarning into whatever was capturing standard output; three of them reached a
    committed table.
    """
    x = np.asarray(x, dtype=np.float64)
    starts = window_starts(len(x), cfg)
    names = statistic_names(cfg)
    records = [score(x[s:s + cfg.window], cfg, seed) for s in starts]

    out: Dict[str, Any] = {"n_windows": len(records),
                           "frac_degenerate": float("nan")}
    for name in names:
        out[name] = float("nan")
        out[name + "_sd"] = float("nan")
    out["tau_used"] = float("nan")
    out["theiler_used"] = float("nan")
    if not records:
        return out

    degenerate = np.array([bool(r["degenerate"]) for r in records])
    out["frac_degenerate"] = float(degenerate.mean())
    # The resolved lag and exclusion travel with the summary: without them a capped run and
    # an uncapped one are indistinguishable in the saved table.
    for key in ("tau_used", "theiler_used"):
        values = np.array([float(r[key]) for r in records])
        values = values[np.isfinite(values)]
        out[key] = float(np.median(values)) if len(values) else float("nan")

    for name in names:
        values = np.array([float(r[name]) for r in records])
        if name in NEIGHBOUR_BASED:
            values = values[~degenerate]
        values = values[np.isfinite(values)]
        if len(values):
            out[name] = float(np.median(values))
            out[name + "_sd"] = float(np.std(values))
    return out

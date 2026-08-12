"""Section 5: recovery on systems whose active dimension is fixed by construction.

One experiment per ladder row, all sharing the driver below, because the archived tree had
the same driver copied into five scripts with different constants and a scoring objective
that changed between them -- one weighted the inversion count, three weighted it
differently, and the last dropped it while still reporting inversions.

The protocol is the article's, and two parts of it are worth stating here because they are
what the numbers mean.

*The configuration is frozen.* Every system is scored at the configuration selected once,
on withheld data, in `calib.e8` and `calib.e20`. Nothing here searches over estimator
settings, and nothing searches over the system's own settings either: the archived
calibration grids swept the drive period and the learning rate alongside the estimator's
parameters, which chooses the data along with the method, and the article records that as
requirement 2 unmet for three rows.

*The split is in rank and in seed.* Ranks are withheld as well as seeds, because the
excitation geometry depends on the rank, so every seed at a given rank would share it and
withholding seeds alone would leave the rank in the training set. That argument only holds
now that the geometry varies with the seed at all; in the archived construction it did not,
which is errata item 1 and the reason section 5's numbers move.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..runtime import CPU, Context, experiment
from ..runtime.parallel import map_ordered

#: Ranks reserved for selection, and ranks the article scores on. Appendix C, table 7.
SELECTION_RANKS: Tuple[int, ...] = (2, 4, 6)
HELDOUT_RANKS: Tuple[int, ...] = (1, 3, 5, 8)
#: Seeds withheld from the eight-direction selection, which used 90, 91 and 92.
HELDOUT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3)

#: The wider ladder rows run to twenty directions; their configuration is the other frozen
#: one, whose selection ranks were 2, 6, 10, 14 and 18 and which withheld the other fifteen.
WIDE_RANKS: Tuple[int, ...] = tuple(r for r in range(1, 21) if r not in (2, 6, 10, 14, 18))
WIDE_SEEDS: Tuple[int, ...] = (0, 1, 2)


#: How long a record each constructed system produces. The frozen window is 8000 samples
#: and the constructed-system stride rule is ``(n - window) // 6``; table 9 of the article
#: gives that stride as 3000, so the record it was measured on was 26,000 samples. The
#: system modules default to 4000, which is shorter than one window and yields no estimate
#: at all -- the geometry has to be set here, where the protocol lives, rather than left to
#: a default written for a quick check.
RECORD = 26_000
FAST_RECORD = 9_000

#: The field each system config calls its record length. They disagree, and renaming one
#: would change a dataclass three other modules construct.
LENGTH_FIELD = {"matrix": "length"}
DEFAULT_LENGTH_FIELD = "window"


def _record_length_field(config_type) -> str:
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_type)}
    for candidate in ("length", "window"):
        if candidate in names:
            return candidate
    raise TypeError(f"{config_type.__name__} has no record-length field")


def _cell(args) -> Dict[str, Any]:
    """Simulate one (rank, seed) and score every observer it records.

    A module-level function, not a closure, so it can cross a process boundary.
    """
    system_id, k, seed, wide, fast = args

    from .. import frozen
    from ..estimator import windows
    from ..systems import spec

    entry = spec.get(system_id)
    field = _record_length_field(entry.config)
    length = FAST_RECORD if fast else RECORD
    config = entry.config(**{"k": k, field: length})
    result = entry.simulate(config, seed=seed)

    base = frozen.twenty_direction() if wide else frozen.eight_direction()
    cfg = frozen.constructed_geometry(base, result.length)

    rows: List[Dict[str, Any]] = []
    names = list(result.series)
    if fast:
        names = names[:2]
    for observer in names:
        stats = windows.summarise(result[observer], cfg, seed=seed)
        rows.append({
            "system": system_id, "k": k, "seed": seed, "observer": observer,
            "truth": float(result.truth.active_dimension),
            "verified": bool(result.truth.verified),
            "effective_rank": float(result.truth.measured.get("effective_rank", np.nan)),
            "resonance_margin": float(result.truth.measured.get("resonance_margin", np.nan)),
            **{key: value for key, value in stats.items() if not key.startswith("_")},
        })
    return {"rows": rows, "truth": result.truth.measured, "length": result.length}


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, without pulling in scipy for one number."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[good])).astype(float)
    ry = np.argsort(np.argsort(y[good])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def _inversions(truth: Sequence[float], estimate: Sequence[float]) -> int:
    """Pairs the estimate orders the wrong way round. Reported, never scored on.

    The archived objective weighted this term at 0.25 in one experiment, 0.15 in four, and
    at zero in the last, which still printed the count. A single number that changes
    meaning between rows of one table is worse than no number, so it is reported beside
    the error rather than folded into it.
    """
    t, e = np.asarray(truth, float), np.asarray(estimate, float)
    good = np.isfinite(t) & np.isfinite(e)
    t, e = t[good], e[good]
    return int(sum(1 for i in range(len(t)) for j in range(i + 1, len(t))
                   if (t[i] - t[j]) * (e[i] - e[j]) < 0))


def _rank_observers(frame) -> Any:
    """Per-observer error against the constructed truth, over the held-out cells."""
    import pandas as pd

    records = []
    for observer, group in frame.groupby("observer", sort=True):
        cell = group.groupby("k", sort=True).agg(
            truth=("truth", "first"), estimate=("MG", "median")).reset_index()
        error = (cell["estimate"] - cell["truth"]).abs()
        records.append({
            "observer": observer,
            "mae": float(error.mean()),
            "max_error": float(error.max()),
            "rho": _spearman(cell["truth"], cell["estimate"]),
            "inversions": _inversions(cell["truth"], cell["estimate"]),
            "n_cells": int(len(cell)),
            "roughness_rho": _spearman(
                cell["truth"], group.groupby("k", sort=True)["roughness"].median()),
        })
    ranked = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_ladder_row(ctx: Context, system_id: str, wide: bool = False) -> None:
    """Score one constructed system at the frozen configuration, held out in rank and seed."""
    import pandas as pd

    ranks = WIDE_RANKS if wide else HELDOUT_RANKS
    seeds = WIDE_SEEDS if wide else HELDOUT_SEEDS
    if ctx.fast:
        ranks, seeds = ranks[:2], seeds[:1]

    ctx.config(system=system_id, configuration="twenty-direction" if wide else "eight-direction",
               heldout_ranks=list(ranks), heldout_seeds=list(seeds))
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions")

    cells = [(system_id, k, seed, wide, ctx.fast) for k in ranks for seed in seeds]
    results = map_ordered(_cell, cells, jobs=ctx.jobs, desc=system_id)

    rows = [row for result in results for row in result["rows"]]
    raw = pd.DataFrame(rows)
    ctx.store.table("heldout_raw.csv", raw)

    summary = (raw.groupby(["observer", "k"], sort=True)
               .agg(truth=("truth", "first"), MG=("MG", "median"), LB=("LB", "median"),
                    PRdelay=("PRdelay", "median"), roughness=("roughness", "median"),
                    n=("MG", "size"))
               .reset_index())
    summary["error"] = (summary["MG"] - summary["truth"]).abs()
    ctx.store.table("heldout_summary.csv", summary)

    ranked = _rank_observers(raw)
    ctx.store.table("observer_ranking.csv", ranked)

    unverified = sorted(set(raw.loc[~raw["verified"], "k"]))
    if unverified:
        # Requirement 1: a construction whose excitation was not confirmed is not a
        # constructed truth, and its row should not be read as recovery.
        ctx.note("unverified_ranks", unverified)
    ctx.note("best_observer", ranked.iloc[0]["observer"] if len(ranked) else None)
    ctx.note("mae_best", float(ranked.iloc[0]["mae"]) if len(ranked) else None)


_ROWS = (
    ("sys.matrix", "matrix", "An oscillating diagonal matrix", ("sec:matrix", "tab:ladder"), False, 2),
    ("sys.linear", "regression.linear", "Online linear regression", ("tab:ladder",), True, 3),
    ("sys.logistic", "regression.logistic", "Logistic regression", ("tab:ladder",), True, 2),
    ("sys.decoder", "decoder", "A frozen nonlinear decoder", ("tab:ladder", "sec:silence"), True, 4),
    ("sys.subspace", "subspace", "A perceptron in a k-subspace", ("tab:ladder", "sec:silence"), True, 2),
    ("sys.digits.function", "digits_function", "Image data, function subspace",
     ("sec:digits", "tab:ladder"), False, 2),
)


def _make(experiment_id: str, system_id: str, title: str, paper, wide: bool, minutes: int):
    @experiment(
        id=experiment_id,
        title=f"{title}: recovery at the frozen configuration",
        paper=paper,
        device=CPU,
        minutes=minutes,
        promotes=("heldout_raw.csv", "heldout_summary.csv", "observer_ranking.csv"),
        tier=1,
        notes=f"Ladder row for the {system_id} system, held out in rank and in seed.",
    )
    def _run(ctx: Context, _system_id: str = system_id, _wide: bool = wide) -> None:
        run_ladder_row(ctx, _system_id, wide=_wide)

    _run.__name__ = experiment_id.replace(".", "_")
    return _run


for _args in _ROWS:
    _make(*_args)

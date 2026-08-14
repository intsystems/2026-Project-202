"""Section 6 and the system it is measured on: when may an estimate be read at all?

Section 5 establishes that the estimate recovers a constructed active dimension. This
module is the other half of that claim -- the conditions under which the recovery holds --
and every experiment in it is a way of asking whether some quantity that is *not* the
active dimension could have produced the same reading.

Two of the experiments here are named ``sys.*`` rather than ``valid.*``. They are the
parameter-subspace system's own runs, and section 6 is built entirely on them: the regime
figure, the observer ranking and the ground-truth table of appendix F all come from the
rank sweep, and the silence control is requirement 4 stated as a measurement. They live
beside the experiments that read them rather than beside the ladder driver, which they do
not use.

Four defects of the archived implementation are fixed here, and each is marked in the code
where it mattered.

*The Theiler exclusion is set through the configuration.* It was a mutable module global
that three worker scripts assigned to from inside their own processes, one of them for the
life of the worker. It sets a published number -- on the transient arm the rule asks for
about 1600 samples and the cap gives 150, so the estimate near 29 is the value at the cap
and not at the rule -- and a number that consequential cannot live somewhere any importer
can write to. It is now :attr:`actdim.estimator.config.EstimatorConfig.theiler_cap`, and
the exclusion sweep of appendix P measures the capped and the uncapped rule side by side.

*One trend-crossing count and one stride rule.* Three copies of the first and two of the
second existed. The count is :func:`actdim.estimator.diagnostics.trend_crossings`, in
closed form so that a window holding a NaN returns NaN rather than raising from inside a
worker; the stride is :func:`actdim.frozen.constructed_geometry`.

*Everything collected in parallel is written in input order.* The archived raw tables were
written in pool-completion order, so the values were stable and no re-run could be diffed
against them.

*The observer no longer moves with the embedding dimension.* See
:mod:`actdim.systems.synthetic`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..observers import CEILING_PANEL
from ..runtime import CPU, Context, experiment
from ..runtime.parallel import default_jobs, map_ordered
from .calibration import mae, shrink as _shrink
from .systems import _spearman as spearman

# ============================================================== shared measurement pieces

#: How high a rank correlation counts as an observer still ordering the ranks. Stated once
#: here rather than chosen inside the experiment that uses it.
TRACKS_RANK = 0.9


def _slope(x, y) -> float:
    """Least-squares slope of ``y`` on ``x``, in closed form.

    ``np.polyfit`` raises inside a worker on a column holding a NaN, which is errata item
    29 in its other guise; two moments and a division do not.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return float("nan")
    a, b = a[good], b[good]
    centred = a - a.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        return float("nan")
    return float(centred @ (b - b.mean()) / denominator)


def _pearson(a, b) -> float:
    """Correlation of two series, NaN where either has no spread."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < 3:
        return float("nan")
    x, y = x[good] - x[good].mean(), y[good] - y[good].mean()
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def _median(values) -> float:
    """Median over the finite entries, NaN when there are none.

    ``np.median`` of an empty selection is a RuntimeWarning and a NaN, and this package
    turns that warning into a test failure for a reason: three of them reached a committed
    table.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _frozen(**overrides):
    """The eight-direction configuration, with whatever this experiment overrides."""
    from .. import frozen

    return frozen.eight_direction(**overrides)


# ============================================================== the parameter-subspace sweep

#: The seven ranks of section 5.4. The four the calibration withheld are 1, 3, 5 and 8.
SWEEP_RANKS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
SWEEP_SEEDS: Tuple[int, ...] = (0, 1, 2, 3)
#: The two controls run on a subset of seeds: they answer a yes-or-no question about the
#: construction rather than measuring a recovery.
CONTROL_SEEDS: Tuple[int, ...] = (0, 1)
#: Held out from the calibration in rank as well as seed. Appendix C, table 7.
TEST_RANKS: Tuple[int, ...] = (1, 3, 5, 8)

SWEEP_RECORD, SWEEP_BURN = 26_000, 4_000
FAST_RECORD, FAST_BURN = 9_000, 1_000

#: The identifiability ratio needs a second embedding at twice ``E_max``. A neighbour
#: query in forty dimensions costs several times one in twenty -- measured, it would be
#: three quarters of this sweep -- so it is computed for one observer per family, on one
#: seed and two ranks per arm, which is enough to say whether a dimension is identifiable
#: in that arm at all.
IDENT_OBSERVERS: Tuple[str, ...] = ("w_fro", "g_fro", "c_proj1")
IDENT_RANKS: Tuple[int, ...] = (2, 6)

#: Every arm of section 5.4 and appendix F, as an override of the system's configuration.
#: The archived experiment built these in one function and three other experiments each
#: imported a different subset of it from that experiment module.
ARMS: Dict[str, Dict[str, Any]] = {
    "qp": dict(mode="qp", drive_amp=0.8, noise_amp=0.0),
    "qp_slow": dict(mode="qp", drive_amp=0.8, noise_amp=0.0, slow=True),
    "noise": dict(mode="noise", drive_amp=0.0, noise_amp=0.08),
    "batch": dict(mode="batch", drive_amp=0.0, noise_amp=0.0, batch=64),
    "batch_proj": dict(mode="batch_proj", drive_amp=0.0, noise_amp=3.0, batch=64),
    "mixed": dict(mode="mixed", drive_amp=0.8, noise_amp=0.02),
    "gd": dict(mode="gd", drive_amp=0.0, noise_amp=0.0, eta=0.006, precondition=False,
               displacement=1.0, transient=True),
    "qp_eta0": dict(mode="qp", drive_amp=0.8, noise_amp=0.0, eta_zero=True),
    "noise_eta0": dict(mode="noise", drive_amp=0.0, noise_amp=0.08, eta_zero=True),
    "qp_nopre": dict(mode="qp", drive_amp=0.8, noise_amp=0.0, precondition=False),
    "noise_nopre": dict(mode="noise", drive_amp=0.0, noise_amp=0.08, precondition=False),
}

#: The arms scored over every seed, and those scored on two. The last four exist to make a
#: property of the construction visible, not to measure a recovery.
MAIN_ARMS: Tuple[str, ...] = ("qp", "qp_slow", "noise", "batch", "batch_proj", "mixed",
                              "gd")
CONTROL_ARMS: Tuple[str, ...] = ("qp_eta0", "noise_eta0", "qp_nopre", "noise_nopre")


def sweep_config(arm: str, rank: int, record: int, burn: int, extra=()):
    """One arm of the parameter-subspace sweep at one rank."""
    from ..systems.digits_parameter import F_FAST, F_SLOW, ten_direction

    settings = dict(ARMS[arm])
    slow = settings.pop("slow", False)
    transient = settings.pop("transient", False)
    settings["f0"] = F_SLOW if slow else F_FAST
    if transient:
        # A transient has no stationary segment to burn into, so the whole record is kept
        # and the burn-in is spent making it: that is why the archived stride rule gives
        # 3666 here and 3000 everywhere else.
        settings.update(window=record + burn, burn=0)
    else:
        settings.update(window=record, burn=burn)
    settings.update(dict(extra))
    return ten_direction(k=rank, **settings)


def _sweep_cell(args) -> List[Dict[str, Any]]:
    """One (arm, rank, seed): simulate once, score every observer at the frozen setting."""
    arm, rank, seed, record, burn, panel, ident, extra = args

    from .. import frozen as frozen_mod
    from .. import observers as registry
    from ..estimator import windows
    from ..systems import digits_parameter

    simulation = digits_parameter.simulate(sweep_config(arm, rank, record, burn, extra),
                                           seed=seed)
    # The stride override of appendix C, table 9, and the only field this pipeline moves.
    cfg = frozen_mod.constructed_geometry(frozen_mod.eight_direction(), simulation.length)
    names = list(windows.statistic_names(cfg))

    rows: List[Dict[str, Any]] = []
    shared = {"arm": arm, "seed": seed, **simulation.info}
    for name in panel:
        series = simulation[name]
        spread = float(series.std())
        base = {**shared, "observer": name, "family": registry.get(name).family,
                "obs_sd": spread}
        if not np.isfinite(series).all() or spread <= 1e-12:
            rows.append({**base, "flat": True, "n_windows": 0,
                         "frac_degenerate": float("nan"),
                         **{key: float("nan") for key in names},
                         "MG_2E": float("nan"), "ident_ratio": float("nan")})
            continue
        record_row = {**base, **windows.summarise(series, cfg, seed=seed), "flat": False,
                      "MG_2E": float("nan"), "ident_ratio": float("nan")}
        if ident and name in IDENT_OBSERVERS:
            from ..estimator.diagnostics import ratio

            # Doubling max_E also doubles the delay span and therefore the exclusion the
            # autocorrelation rule asks for, which the cap then treats asymmetrically
            # between the two halves; appendix N states that and it is why the ratio is
            # reported rather than used as a threshold.
            doubled = windows.summarise(series, cfg.replace(max_E=2 * cfg.max_E),
                                        seed=seed)
            record_row["MG_2E"] = doubled["MG"]
            record_row["ident_ratio"] = ratio(record_row["MG"], doubled["MG"])
        rows.append(record_row)
    return rows


def _observer_scores(raw):
    """Per (arm, observer): does the estimate track the measured rank, and does anything else?

    Every null is reported beside the estimate, because the question is never whether the
    estimate correlates with the rank -- any compressive monotone function of the rank
    does -- but whether it carries information the roughness, the linear participation
    ratio and the spectral count do not.
    """
    import pandas as pd

    records = []
    for (arm, observer), group in raw.groupby(["arm", "observer"], sort=True):
        if bool(group["flat"].all()):
            continue
        cell = group.groupby("r", sort=True).agg(
            MG=("MG", "median"), truth=("traj_PR", "median"),
            rough=("roughness", "median"), prd=("PRdelay", "median"),
            spec=("specPR0", "median"), spec256=("specPR256", "median"),
            ident=("ident_ratio", "median"), degenerate=("frac_degenerate", "mean"),
            spread=("MG", "std")).reset_index()
        if cell["truth"].nunique() < 3:
            continue
        records.append({
            "arm": arm, "observer": observer, "family": group["family"].iloc[0],
            "rho_MG": spearman(cell["MG"], cell["truth"]),
            "rho_rough": spearman(cell["rough"], cell["truth"]),
            "rho_PRdelay": spearman(cell["prd"], cell["truth"]),
            "rho_specPR": spearman(cell["spec"], cell["truth"]),
            "mae_specPR": mae(cell["spec"], cell["truth"]),
            "rho_specPR256": spearman(cell["spec256"], cell["truth"]),
            "slope": _slope(cell["truth"], cell["MG"]),
            "mae_raw": mae(cell["MG"], cell["truth"]),
            "ident": _median(cell["ident"]),
            "seed_sd": _median(cell["spread"]),
            "degen": float(cell["degenerate"].mean()),
        })
    return pd.DataFrame(records)


def _calibration_pairs() -> Tuple[Dict[str, Dict[str, List[float]]], str]:
    """The (estimate, dimension) pairs the frozen calibration was fitted on.

    Read from the frozen configuration file, which is where a calibration belongs: the map
    and the split it was fitted on travel together, and refitting on whatever data is to
    hand is how a held-out error becomes a training error. Files written by this port carry
    the pairs; the archived file carries only the knots of the fitted isotonic map, and
    those are used as a stand-in with the substitution recorded.
    """
    from .. import frozen as frozen_mod

    stored = json.loads(frozen_mod.frozen_path(frozen_mod.EIGHT_DIRECTION)
                        .read_text(encoding="utf-8"))
    points = stored.get("calibration_points")
    if points:
        return points, "calibration_points"
    knots = stored.get("isotonic", {})
    return ({observer: {"estimate": knot["x"], "truth": knot["y"]}
             for observer, knot in knots.items()}, "isotonic knots")


def _calibrated_error(raw, ranks: Sequence[int]):
    """Absolute recovery on the withheld ranks, under each of the three calibrations.

    The calibration is fitted on a split disjoint from these ranks in both seed and rank,
    and it is never refitted here.
    """
    import pandas as pd

    from ..estimator.calibration import Calibration

    pairs, source = _calibration_pairs()
    records = []
    for (arm, observer), group in raw.groupby(["arm", "observer"], sort=True):
        if observer not in pairs or bool(group["flat"].all()):
            continue
        estimate = np.asarray(pairs[observer]["estimate"], dtype=float)
        truth = np.asarray(pairs[observer]["truth"], dtype=float)
        if len(estimate) < 3:
            continue
        test = group[group["r"].isin(ranks)]
        if test.empty:
            continue
        for kind in ("isotonic", "affine", "identity"):
            try:
                fitted = Calibration(kind).fit(estimate, truth)
            except (ValueError, np.linalg.LinAlgError):
                continue
            predicted = np.atleast_1d(fitted.predict(test["MG"].to_numpy(dtype=float)))
            difference = predicted - test["traj_PR"].to_numpy(dtype=float)
            finite = np.isfinite(difference)
            records.append({"arm": arm, "observer": observer, "calibration": kind,
                            "mae": mae(predicted, test["traj_PR"]),
                            "bias": float(difference[finite].mean()) if finite.any()
                                    else float("nan"),
                            "n": int(finite.sum())})
    return pd.DataFrame(records), source


@experiment(
    id="sys.digits.parameter",
    title="A constrained head on image data: nine excitation cases, twelve observers",
    paper=("sec:digits", "tab:ladder", "tab:gt", "fig:regimes", "fig:observers"),
    device=CPU,
    minutes=143,
    promotes=("sweep_raw.csv", "observer_scores.csv", "calibrated_mae.csv",
              "ground_truth_PR.csv"),
    tier=1,
    notes="The article's most-cited system. Two hours on eight cores; check the --fast "
          "path before starting it unattended.",
)
def digits_parameter(ctx: Context) -> None:
    import pandas as pd

    from ..observers import PAPER_TWELVE

    panel = PAPER_TWELVE
    ranks, seeds, control_seeds = SWEEP_RANKS, SWEEP_SEEDS, CONTROL_SEEDS
    record, burn = SWEEP_RECORD, SWEEP_BURN
    main, controls = MAIN_ARMS, CONTROL_ARMS
    ident_ranks = IDENT_RANKS
    if ctx.fast:
        panel = ("w_fro", "g_fro", "loss_full")
        ranks, seeds, control_seeds = (2, 4, 6), (0,), (0,)
        record, burn = FAST_RECORD, FAST_BURN
        main, controls = ("qp", "gd"), ("qp_eta0",)
        ident_ranks = (2,)

    ctx.config(arms=list(main) + list(controls), ranks=list(ranks), seeds=list(seeds),
               observers=list(panel), record=record, burn=burn,
               geometry="frozen eight-direction, stride max(500, (n - window) // 6)")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions", "adapter",
                      "rotation")

    extra = _shrink(ctx.fast)
    cells = [(arm, rank, seed, record, burn, panel,
              seed == seeds[0] and rank in ident_ranks, extra)
             for arm in main for rank in ranks for seed in seeds]
    cells += [(arm, rank, seed, record, burn, panel, False, extra)
              for arm in controls for rank in ranks for seed in control_seeds]
    collected = map_ordered(_sweep_cell, cells, jobs=ctx.jobs, desc="sys.digits.parameter")
    raw = pd.DataFrame([row for cell in collected for row in cell])
    ctx.store.table("sweep_raw.csv", raw)

    truth = (raw.drop_duplicates(["arm", "r", "seed"])
             .pivot_table(index="arm", columns="r", values="traj_PR", aggfunc="median")
             .reset_index())
    ctx.store.table("ground_truth_PR.csv", truth)

    ctx.store.table("observer_scores.csv", _observer_scores(raw))
    calibrated, source = _calibrated_error(raw, [r for r in TEST_RANKS if r in ranks]
                                           or list(ranks))
    ctx.store.table("calibrated_mae.csv", calibrated)
    ctx.note("calibration_source", source)
    ctx.note("unverified_arms", sorted(set(raw.loc[~raw["flat"], "arm"])
                                       & {"batch", "gd"}))


# ============================================================== the silence control

#: The ranks and seeds the control is run at. Enough ranks for a rank correlation to mean
#: something and few enough that seven systems fit in the quarter of an hour the check is
#: worth: what is being asked is whether an observer still moves at all, not how accurately
#: it recovers.
SILENCE_RANKS: Tuple[int, ...] = (2, 4, 6, 8)
SILENCE_SEEDS: Tuple[int, ...] = (0, 1)
#: Exactly one frozen window per record. The question here is whether an observer still
#: moves and still orders the ranks with the optimiser switched off, and a sliding trace
#: answers it no better than a single window while costing six times as much across seven
#: systems, two arms and twelve observers.
SILENCE_RECORD = 8_000

#: A silent series counts as still moving when its spread reaches this fraction of the
#: trained run's. The floor has to clear the measurement jitter: every constructed system
#: adds noise of :data:`actdim.systems.spec.DEFAULT_JITTER`, a millionth of the spread, to
#: each recorded series, so a frozen observer arrives here with a spread of exactly that
#: and not of zero. An observer that really still reads the drive moves by a tenth of its
#: trained spread or more, so three decades of clearance costs nothing and nothing turns on
#: where in that gap the line sits.
SILENT_SPREAD_FLOOR = 1e-3


def _silence_config(system_id: str, rank: int, record: int, silent: bool, extra=()):
    """One constructed system at one rank, trained or silenced.

    Only the record length and the learning rate move. Every other setting is the
    system's own, including its burn-in, because the control has to differ from the run it
    is a control for in exactly one thing.
    """
    import dataclasses

    from ..systems import spec

    entry = spec.get(system_id)
    fields = {f.name for f in dataclasses.fields(entry.config)}
    settings: Dict[str, Any] = {"k": rank}
    if "window" in fields:
        settings["window"] = record
    if "length" in fields:
        settings["length"] = record
    if system_id == "digits_parameter":
        # The backbone overrides belong to that system alone. Applying them by field name
        # across the ladder would hand the decoder a tuple where it wants a width, which
        # is what several of the archived scripts did to each other with shared constants.
        settings.update(dict(extra))
    if silent:
        # The parameter-subspace system says the same thing with a flag, because setting
        # its learning rate to zero would also divide by zero in the gain equalisation --
        # the drive has to stay exactly as it was for the control to be a control.
        if "eta_zero" in fields:
            settings["eta_zero"] = True
        elif "eta" in fields:
            settings["eta"] = 0.0
        else:
            return None
    return entry.configure(**settings)


def _silence_cell(args) -> List[Dict[str, Any]]:
    """One (system, rank, seed) run twice: with the optimiser, and with it silenced."""
    system_id, rank, seed, record, extra = args

    from .. import frozen as frozen_mod
    from ..estimator import windows
    from ..systems import spec

    entry = spec.get(system_id)
    trained_config = _silence_config(system_id, rank, record, False, extra)
    silent_config = _silence_config(system_id, rank, record, True, extra)
    trained = entry.simulate(trained_config, seed=seed)
    silent = entry.simulate(silent_config, seed=seed)
    cfg = frozen_mod.constructed_geometry(frozen_mod.eight_direction(), trained.length)

    rows: List[Dict[str, Any]] = []
    for name in trained.series:
        left, right = trained[name], silent[name]
        spread, silent_spread = float(left.std()), float(right.std())
        moves = silent_spread > SILENT_SPREAD_FLOOR * max(spread, 1e-300)
        rows.append({
            "system": system_id, "k": rank, "seed": seed, "observer": name,
            "applicable": True,
            "truth": float(trained.truth.active_dimension),
            "sd_trained": spread, "sd_silent": silent_spread,
            "sd_ratio": silent_spread / spread if spread > 0.0 else float("nan"),
            "moves_when_silent": bool(moves),
            "series_correlation": _pearson(left, right),
            "MG_trained": windows.summarise(left, cfg, seed=seed)["MG"],
            "MG_silent": (windows.summarise(right, cfg, seed=seed)["MG"] if moves
                          else float("nan")),
        })
    return rows


def _claims_state_only(observer: str) -> bool:
    """Is this observer declared to be a function of the optimiser state alone?

    Requirement 4 is about that claim, so the verdict has to know which observers make it.
    Eleven of the article's twelve do; ``loss_step`` does not, because it contains the
    instantaneous drive weights, and it is kept in the panel precisely so that the
    contamination is visible rather than assumed away. Every other constructed system's
    observers are functions of the optimiser state and none of them declares an exception.
    """
    from .. import observers as registry

    return registry.REGISTRY[observer].state_only if observer in registry.REGISTRY else True


def _silence_verdict(raw):
    """Per (system, observer): does the reading survive the optimiser being switched off?

    A reading *survives the control* when, at zero learning rate, the series still varies
    and its estimate still orders the ranks. Both halves are needed: a series that stops
    moving carries nothing, and one that moves without ordering the ranks is reading the
    drive's amplitude rather than its dimension.

    Surviving is a failure of requirement 4 only for an observer that claims to be a
    function of the optimiser state. The two are kept in separate columns so that the one
    observer designed to survive does not by itself condemn the system it belongs to.
    """
    import pandas as pd

    records = []
    for (system, observer), group in raw.groupby(["system", "observer"], sort=True):
        claims = _claims_state_only(observer)
        if not bool(group["applicable"].iloc[0]):
            records.append({"system": system, "observer": observer, "applicable": False,
                            "claims_state_only": claims, "n_cells": int(len(group)),
                            "moves_when_silent": float("nan"),
                            "sd_ratio": float("nan"),
                            "series_correlation": float("nan"),
                            "rho_trained": float("nan"), "rho_silent": float("nan"),
                            "mae_trained": float("nan"), "mae_silent": float("nan"),
                            "MG_trained": float("nan"), "MG_silent": float("nan"),
                            "survives_silence": False, "fails_requirement_4": False})
            continue
        cell = group.groupby("k", sort=True).agg(
            trained=("MG_trained", "median"), silent=("MG_silent", "median"),
            truth=("truth", "first")).reset_index()
        moving = float(group["moves_when_silent"].mean())
        rho_silent = spearman(cell["silent"], cell["k"])
        survives = bool(moving > 0.5 and np.isfinite(rho_silent)
                        and rho_silent >= TRACKS_RANK)
        records.append({
            "system": system, "observer": observer, "applicable": True,
            "claims_state_only": claims, "n_cells": int(len(group)),
            "moves_when_silent": moving,
            "sd_ratio": _median(group["sd_ratio"]),
            "series_correlation": _median(group["series_correlation"]),
            "rho_trained": spearman(cell["trained"], cell["k"]),
            "rho_silent": rho_silent,
            "mae_trained": mae(cell["trained"], cell["truth"]),
            "mae_silent": mae(cell["silent"], cell["truth"]),
            "MG_trained": _median(cell["trained"]),
            "MG_silent": _median(cell["silent"]),
            "survives_silence": survives,
            "fails_requirement_4": bool(survives and claims),
        })
    verdict = pd.DataFrame(records)
    if verdict.empty:
        return verdict
    verdict["system_invalidated"] = (
        verdict.groupby("system")["fails_requirement_4"].transform("any"))
    return verdict


@experiment(
    id="sys.silence",
    title="The zero-learning-rate control: which systems does it invalidate?",
    paper=("sec:silence", "tab:ladder"),
    device=CPU,
    minutes=12,
    promotes=("silence.csv",),
    tier=1,
    notes="New. Section 5.3 says the control invalidated two of six systems and "
          "tab:ladder marks two of them, but no such arm exists in the archived tree and "
          "no result file records one (errata 10). This runs it.",
)
def silence(ctx: Context) -> None:
    import pandas as pd

    from ..systems import spec

    systems = [name for name in spec.LADDER if name in spec.load()]
    ranks, seeds = SILENCE_RANKS, SILENCE_SEEDS
    record = SILENCE_RECORD
    if ctx.fast:
        ranks, seeds = (2, 3, 4), (0,)

    ctx.config(systems=systems, ranks=list(ranks), seeds=list(seeds), record=record,
               rule="an observer fails requirement 4 when its silent series still varies "
                    "and its silent estimate still orders the ranks")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions", "adapter")

    extra = _shrink(ctx.fast)
    runnable, skipped = [], []
    for system_id in systems:
        if _silence_config(system_id, ranks[0], record, True, extra) is None:
            skipped.append(system_id)
        else:
            runnable.append(system_id)

    cells = [(system_id, rank, seed, record, extra)
             for system_id in runnable for rank in ranks for seed in seeds]
    collected = map_ordered(_silence_cell, cells, jobs=ctx.jobs, desc="sys.silence")
    rows = [row for cell in collected for row in cell]

    # A system with no learning rate has no control to run: the oscillating diagonal
    # matrix has no optimiser at all, which is exactly why section 5.1 says a negative
    # result there would be decisive. It is listed rather than dropped, so the table
    # covers every rung and says which ones the question can even be put to.
    for system_id in skipped:
        entry = spec.get(system_id)
        example = entry.simulate(
            _silence_config(system_id, ranks[0], record, False, extra), seed=seeds[0])
        for name in example.series:
            rows.append({"system": system_id, "k": ranks[0], "seed": seeds[0],
                         "observer": name, "applicable": False,
                         "truth": float(example.truth.active_dimension),
                         "sd_trained": float(example[name].std()),
                         "sd_silent": float("nan"), "sd_ratio": float("nan"),
                         "moves_when_silent": False,
                         "series_correlation": float("nan"),
                         "MG_trained": float("nan"), "MG_silent": float("nan")})

    verdict = _silence_verdict(pd.DataFrame(rows))
    ctx.store.table("silence.csv", verdict)
    ctx.store.table("silence_cells.csv", pd.DataFrame(rows))

    failing = verdict[verdict["fails_requirement_4"]]
    surviving = verdict[verdict["survives_silence"] & ~verdict["fails_requirement_4"]]
    ctx.note("no_learning_rate", skipped)
    ctx.note("invalidated", sorted(set(failing["system"])))
    ctx.note("observers_that_fail",
             {system: sorted(group["observer"])
              for system, group in failing.groupby("system")})
    # Recorded separately: an observer that is not claimed to be a function of the
    # optimiser state is meant to survive this, and its surviving is not a defect.
    ctx.note("survive_by_design",
             {system: sorted(group["observer"])
              for system, group in surviving.groupby("system")})


# ============================================================== valid.regime

ATLAS_RANKS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
ATLAS_SEEDS: Tuple[int, ...] = (0, 1, 2)
ATLAS_OBSERVERS: Tuple[str, ...] = ("generic", "norm")
ATLAS_EMBEDDINGS: Tuple[int, ...] = (10, 20)
#: Two base periods, so that "cycles of the slowest mode" and "number of samples" can be
#: told apart: 200 cycles at a period of 400 and 50 at a period of 100 are both N = 20000.
ATLAS_PERIODS: Tuple[float, ...] = (1 / 400.0, 1 / 100.0)
ATLAS_CYCLES: Tuple[int, ...] = (50, 200, 800)
ATLAS_TAU_C: Tuple[float, ...] = (50.0, 200.0, 1000.0)
ATLAS_LENGTHS: Tuple[int, ...] = (20_000, 60_000)
ATLAS_MAX_SAMPLES = 100_000


def _atlas_cell(args) -> Dict[str, Any]:
    family, rank, length, seed, observer, max_e, options = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..systems import synthetic

    state, meta = synthetic.generate(family, rank, length, seed, **dict(options))
    series = synthetic.observe(state, seed, observer)
    hard, effective = synthetic.state_rank(state)
    cfg = EstimatorConfig(max_E=max_e, tau=1, k_neighbors=5, theiler="embedding",
                          window=length, stride=length)
    scored = windows.score(series, cfg, seed=seed)
    row = {"family": family, "r": rank, "N": length, "seed": seed, "observer": observer,
           "max_E": max_e, "state_rank": hard, "state_PR": effective,
           "margin": float(meta.get("margin", np.nan)),
           "innov_ratio": float(meta.get("innov_ratio", np.nan)),
           "tau_c": float(meta.get("tau_c", np.nan)),
           # The base period, which the archived table did not carry and which two of its
           # cells needed: 200 cycles at a period of 400 and 50 cycles at a period of 100
           # are both 20,000 samples, so a key without it merges two different systems and
           # the identifiability ratio below is then a ratio of two averages over them.
           "f0": float(dict(options).get("f0", np.nan))}
    row.update({name: scored[name] for name in windows.statistic_names(cfg)})
    row["degenerate"] = bool(scored["degenerate"])
    return row


def _atlas_jobs(fast: bool):
    import itertools

    ranks = ATLAS_RANKS if not fast else (1, 3)
    seeds = ATLAS_SEEDS if not fast else (0,)
    periods = ATLAS_PERIODS if not fast else (1 / 100.0,)
    cycles = ATLAS_CYCLES if not fast else (50,)
    tau_cs = ATLAS_TAU_C if not fast else (200.0,)
    lengths = ATLAS_LENGTHS if not fast else (4_000,)
    observers = ATLAS_OBSERVERS if not fast else ("generic",)

    jobs = []
    for rank, f0, cycle, seed, observer, max_e in itertools.product(
            ranks, periods, cycles, seeds, observers, ATLAS_EMBEDDINGS):
        length = int(cycle / f0)
        if length > ATLAS_MAX_SAMPLES:
            continue
        jobs.append(("qp", rank, length, seed, observer, max_e, (("f0", f0),)))
    for rank, tau_c, length, seed, max_e in itertools.product(
            ranks, tau_cs, lengths, seeds[:2], ATLAS_EMBEDDINGS):
        jobs.append(("ou", rank, length, seed, "generic", max_e, (("tau_c", tau_c),)))
        jobs.append(("colored", rank, length, seed, "generic", max_e,
                     (("tau_c", tau_c), ("order", 3))))
    return jobs


@experiment(
    id="valid.regime",
    title="The identifiability atlas: which dynamical regime lets the rank be recovered",
    paper=("sec:regime", "fig:regimes"),
    device=CPU,
    minutes=14,
    promotes=("atlas_raw.csv", "identifiability_ratio.csv"),
    tier=1,
    notes="Nothing about a network is asked here. Three families with the same nominal "
          "rank and three different geometries, at two embedding dimensions.",
)
def regime(ctx: Context) -> None:
    import pandas as pd

    from ..estimator.diagnostics import ratio

    jobs = _atlas_jobs(ctx.fast)
    ctx.config(families=["qp", "ou", "colored"], embeddings=list(ATLAS_EMBEDDINGS),
               cells=len(jobs),
               note="both embeddings of a cell see the identical series, which the "
                    "archived seeding did not give them")
    ctx.declare_seeds("drive_phases", "ou_innovations", "coloured_innovations",
                      "observer_projection")

    rows = map_ordered(_atlas_cell, jobs, jobs=ctx.jobs, desc="valid.regime")
    raw = pd.DataFrame(rows)
    ctx.store.table("atlas_raw.csv", raw)

    # The identifiability ratio: near one the estimate is a property of the data, near two
    # it is a property of the embedding space and no dimension is identifiable at this
    # sample size. Grouped with dropna=False because the torus arm has no correlation time
    # and would otherwise vanish from the table entirely.
    key = ["family", "r", "N", "seed", "observer", "tau_c", "f0"]
    wide = (raw.groupby(key + ["max_E"], dropna=False)["MG"].median()
            .unstack("max_E").reset_index())
    low, high = ATLAS_EMBEDDINGS
    wide["ident_ratio"] = [ratio(a, b) for a, b in zip(wide[low], wide[high])]
    ctx.store.table("identifiability_ratio.csv", wide)
    ctx.note("median_ratio", {family: float(np.nanmedian(group["ident_ratio"]))
                              for family, group in wide.groupby("family")
                              if np.isfinite(group["ident_ratio"]).any()})


# ============================================================== valid.tau

TAU_RANKS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
TAU_GRID: Tuple[Any, ...] = (1, 2, 4, 8, 16, 32, "acorr")
TAU_PERIODS: Tuple[int, ...] = (16, 400)
TAU_LENGTH = 24_000
TAU_SEEDS: Tuple[int, ...] = (0, 1)


def _tau_cell(args) -> Dict[str, Any]:
    period, rank, tau, max_e, seed, length = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..systems import synthetic

    state, meta = synthetic.quasiperiodic(rank, length, seed, f0=1.0 / period)
    series = synthetic.observe(state, seed, "generic")
    cfg = EstimatorConfig(max_E=max_e, tau=tau, k_neighbors=5, theiler="embedding",
                          window=length, stride=length)
    scored = windows.score(series, cfg, seed=seed)
    used = scored["tau_used"]
    span = (max_e - 1) * (used if np.isfinite(used) and used > 0 else 1.0)
    return {"period": period, "r": rank, "tau": tau, "max_E": max_e, "seed": seed,
            "tau_used": used, "theiler_used": scored["theiler_used"],
            "span": span, "span_periods": span / period,
            "MG": scored["MG"], "PRdelay": scored["PRdelay"],
            "specPR0": scored["specPR0"], "specPR256": scored["specPR256"],
            "roughness": scored["roughness"],
            "degenerate": bool(scored["degenerate"]),
            "margin": float(meta["margin"])}


@experiment(
    id="valid.tau",
    title="How much of the saturation is the estimator and how much is the delay lag",
    paper=("sec:tau", "tab:tau", "fig:tau"),
    device=CPU,
    minutes=11,
    promotes=("tau_sensitivity.csv",),
    tier=1,
    notes="The system does not change across this sweep. What moves is the estimator's "
          "own free parameter, and it moves the estimate by an order of magnitude.",
)
def tau(ctx: Context) -> None:
    import itertools

    import pandas as pd

    ranks, taus, periods = TAU_RANKS, TAU_GRID, TAU_PERIODS
    embeddings, seeds, length = (10, 20), TAU_SEEDS, TAU_LENGTH
    if ctx.fast:
        ranks, taus = (1, 4), (1, 8, "acorr")
        seeds, length = (0,), 6_000

    ctx.config(periods=list(periods), ranks=list(ranks), taus=[str(t) for t in taus],
               embeddings=list(embeddings), seeds=list(seeds), length=length)
    ctx.declare_seeds("drive_phases", "observer_projection")

    cells = [(period, rank, tau_value, max_e, seed, length)
             for period, rank, tau_value, max_e, seed
             in itertools.product(periods, ranks, taus, embeddings, seeds)]
    rows = map_ordered(_tau_cell, cells, jobs=ctx.jobs, desc="valid.tau")
    frame = pd.DataFrame(rows)
    ctx.store.table("tau_sensitivity.csv", frame)

    fixed = frame[frame["tau"] != "acorr"]
    ctx.note("spread_across_tau",
             {int(rank): float(group["MG"].max() - group["MG"].min())
              for rank, group in fixed[fixed["max_E"] == max(embeddings)].groupby("r")})


# ============================================================== valid.nuisance

NUISANCE_RANK = 4
NUISANCE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
NUISANCE_RECORD, NUISANCE_BURN = 22_000, 4_000
#: Each control is a way the earlier reports' "dimension drop" could have been produced by
#: something that is not a change of dimension.
NUISANCE_CONTROLS: Tuple[str, ...] = ("baseline", "obs_scale", "amp_ramp", "lr_step",
                                      "noise_step", "rotate", "freq_half", "freq_double")
#: ``rotate`` acts through the coordinates, so only the fixed parameter projection can see
#: it; scoring the other eleven observers would dilute its rate to zero by construction.
ROTATION_OBSERVERS: Tuple[str, ...] = ("c_proj1",)
#: ``noise_step`` multiplies a noise amplitude that is zero on the recurrent arm, so the
#: run would be bit-identical to the baseline; ``amp_ramp`` and the two band shifts act on
#: a drive the stochastic arm does not have.
NUISANCE_SKIP: Tuple[Tuple[str, str], ...] = (("qp", "noise_step"),
                                              ("noise", "amp_ramp"),
                                              ("noise", "freq_half"),
                                              ("noise", "freq_double"))


def _nuisance_setup(control: str, mode: str, rank: int, record: int, burn: int, extra=()):
    """One control's system configuration and its per-step schedules."""
    from ..systems.digits_parameter import F_FAST, Schedules, ten_direction

    settings: Dict[str, Any] = dict(k=rank, window=record, burn=burn, eta=0.15,
                                    precondition=True, f0=F_FAST)
    settings.update(dict(mode="qp", drive_amp=0.8, noise_amp=0.0) if mode == "qp"
                    else dict(mode="noise", drive_amp=0.0, noise_amp=0.08))
    length = record + burn
    t = np.arange(length, dtype=float)
    schedules = Schedules()
    if control == "obs_scale":
        schedules = Schedules(observer_gain=np.exp(np.log(10.0) * t / length))
    elif control == "amp_ramp":
        schedules = Schedules(amplitude=1.0 + 3.0 * t / length)
    elif control == "lr_step":
        values = np.ones(length)
        values[length // 2:] = 0.5
        schedules = Schedules(learning_rate=values)
    elif control == "noise_step":
        values = np.ones(length)
        values[length // 2:] = 3.0
        schedules = Schedules(noise=values)
    elif control == "rotate":
        settings["rotate"] = True
    elif control == "freq_half":
        settings["f0"] = F_FAST / 2.0
    elif control == "freq_double":
        settings["f0"] = F_FAST * 2.0
    elif control != "baseline":
        raise ValueError(f"unknown control {control!r}")
    settings.update(dict(extra))
    return ten_direction(**settings), schedules


def _nuisance_cell(args) -> List[Dict[str, Any]]:
    control, mode, seed, rank, record, burn, panel, extra = args

    from .. import frozen as frozen_mod
    from ..estimator import windows
    from ..systems import digits_parameter

    config, schedules = _nuisance_setup(control, mode, rank, record, burn, extra)
    simulation = digits_parameter.simulate(config, seed=seed, schedules=schedules)
    # The frozen window geometry, unmodified: this experiment asks whether a nuisance
    # moves the estimate at the setting the article quotes, so it must not also move the
    # stride.
    cfg = frozen_mod.eight_direction()

    rows = []
    for name in panel:
        series = simulation[name]
        if float(series.std()) <= 1e-12:
            continue
        _, traces = windows.sliding(series, cfg, seed=seed)
        usable = (~(traces["degenerate"] > 0.5)) & np.isfinite(traces["MG"])
        count = len(traces["MG"])
        first = usable & (np.arange(count) < count // 2)
        second = usable & (np.arange(count) >= count // 2)
        rows.append({
            "control": control, "mode": mode, "seed": seed, "observer": name,
            "mg_all": _median(traces["MG"][usable]),
            "mg_first": _median(traces["MG"][first]),
            "mg_second": _median(traces["MG"][second]),
            "rough_all": _median(traces["roughness"][usable]),
            "acorr_all": _median(traces["acorr"][usable]),
            "prd_all": _median(traces["PRdelay"][usable]),
            "spec_all": _median(traces["specPR0"][usable]),
            "traj_PR": simulation.info["traj_PR"],
            "n_windows": int(usable.sum()),
        })
    return rows


def _nuisance_score(raw, rotation_observers: Sequence[str]):
    """Two statistics, two nulls, and a threshold measured from the baseline runs.

    ``within`` is a paired half-to-half difference inside one run; ``between`` compares a
    run's level with the baseline level and additionally carries the seed-to-seed
    variance. Sharing one threshold would give the two arms of the disjunction different
    and unmeasured sizes, so each gets its own 97.5th percentile, which is the Bonferroni
    correction for a nominal five per cent overall.

    The baseline's own ``between`` null is computed leave-one-out, so that a run does not
    help define the reference it is then compared against and shrink its own distance to
    it.
    """
    import pandas as pd

    frame = raw[~((raw["control"] == "rotate")
                  & (~raw["observer"].isin(rotation_observers)))].copy()
    frame["within"] = (frame["mg_second"] - frame["mg_first"]).abs()

    baseline = frame[frame["control"] == "baseline"]
    level = baseline.groupby(["mode", "observer"])["mg_all"].median().rename("base_level")
    scored = frame.merge(level, on=["mode", "observer"], how="left")
    scored["between"] = (scored["mg_all"] - scored["base_level"]).abs()

    leave_one_out = []
    for (mode, observer), group in baseline.groupby(["mode", "observer"]):
        values = group["mg_all"].to_numpy(dtype=float)
        for index in range(len(values)):
            others = np.delete(values, index)
            if len(others):
                leave_one_out.append({"mode": mode, "observer": observer,
                                      "loo": abs(values[index] - _median(others))})
    if not leave_one_out:
        raise ValueError("the baseline arm produced no runs, so no threshold can be set")
    thresholds_within = (baseline.groupby(["mode", "observer"])["within"]
                         .quantile(0.975).rename("delta_within"))
    thresholds_between = (pd.DataFrame(leave_one_out).groupby(["mode", "observer"])["loo"]
                          .quantile(0.975).rename("delta_between"))
    scored = (scored.merge(thresholds_within, on=["mode", "observer"], how="left")
              .merge(thresholds_between, on=["mode", "observer"], how="left"))
    scored["fires_within"] = scored["within"] > scored["delta_within"]
    scored["fires_between"] = scored["between"] > scored["delta_between"]
    scored["fires"] = scored["fires_within"] | scored["fires_between"]
    return scored


@experiment(
    id="valid.nuisance",
    title="Seven factors that change without the active dimension changing",
    paper=("sec:nuisance", "tab:controls"),
    device=CPU,
    minutes=34,
    promotes=("controls_raw.csv", "controls_scored.csv"),
    tier=1,
    notes="The false-alarm threshold is estimated from the baseline runs rather than "
          "guessed, and the two statistics get one threshold each.",
)
def nuisance(ctx: Context) -> None:
    import pandas as pd

    from ..observers import PAPER_TWELVE

    panel = PAPER_TWELVE
    controls, seeds = NUISANCE_CONTROLS, NUISANCE_SEEDS
    record, burn, rank = NUISANCE_RECORD, NUISANCE_BURN, NUISANCE_RANK
    modes = ("qp", "noise")
    if ctx.fast:
        panel = ("w_fro", "c_proj1", "g_fro")
        controls = ("baseline", "obs_scale", "rotate", "noise_step", "freq_half")
        seeds = (0, 1, 2)
        # Long enough for four windows at the frozen geometry, so that the half-to-half
        # statistic and its threshold are exercised and not merely reached.
        record, burn = 14_000, 1_000

    ctx.config(controls=list(controls), modes=list(modes), seeds=list(seeds),
               rank=rank, observers=list(panel), record=record, burn=burn,
               geometry="frozen eight-direction, window and stride unchanged")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions", "rotation")

    cells = [(control, mode, seed, rank, record, burn, panel, _shrink(ctx.fast))
             for mode in modes for control in controls for seed in seeds
             if (mode, control) not in NUISANCE_SKIP]
    collected = map_ordered(_nuisance_cell, cells, jobs=ctx.jobs, desc="valid.nuisance")
    raw = pd.DataFrame([row for cell in collected for row in cell])
    ctx.store.table("controls_raw.csv", raw)

    rotation_observers = tuple(n for n in ROTATION_OBSERVERS if n in panel) or panel[:1]
    scored = _nuisance_score(raw, rotation_observers)
    ctx.store.table("controls_scored.csv", scored)
    ctx.note("false_alarm_rate",
             {str(key): float(group["fires"].mean())
              for key, group in scored.groupby(["mode", "control"])})


# ============================================================== valid.anisotropy

ANISO_RANKS: Tuple[int, ...] = (2, 4, 6, 8)
#: The amplitude of mode ``l`` is ``q ** l``. One is the isotropic construction every other
#: experiment uses, and it is what makes the two candidate estimands agree there.
ANISO_DECAY: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
ANISO_SEEDS: Tuple[int, ...] = (0, 1, 2)
ANISO_LENGTH = 12_000
ANISO_F0 = 1 / 16.0


def _aniso_cell(args) -> Dict[str, Any]:
    """One anisotropic torus, observed through the squared Frobenius norm.

    The construction is the oscillating diagonal matrix of section 5.1 with one thing
    changed: the amplitudes decay geometrically instead of being drawn. It is written out
    here rather than added to :mod:`actdim.systems.matrix`, because it is not a rung -- it
    is a probe of what the estimator reports when the manifold dimension and the effective
    rank disagree, and no ladder row is allowed to have them disagree.
    """
    rank, decay, seed, cfg_values = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..linalg import participation_ratio, rank_report
    from ..runtime.determinism import rng as stream_rng
    from ..systems.drive import DEFAULT_BAND, centre_for_octave, frequencies

    freqs = frequencies(rank, centre_for_octave(ANISO_F0, DEFAULT_BAND), seed=seed)
    generator = stream_rng(seed, "drive_phases")
    phases = generator.uniform(0.0, 2.0 * np.pi, rank)
    amplitudes = np.asarray([decay ** level for level in range(rank)], dtype=float)
    t = np.arange(ANISO_LENGTH, dtype=float)

    oscillating = amplitudes * np.sin(2.0 * np.pi * np.outer(t, freqs) + phases)
    offsets = 1.0 + 0.5 * stream_rng(seed, "matrix_base").random(rank)
    series = ((offsets + oscillating) ** 2).sum(axis=1)

    measured = rank_report(oscillating, center=True).effective_rank
    predicted = participation_ratio(amplitudes ** 2)
    scored = windows.summarise(series, EstimatorConfig.from_dict(dict(cfg_values)),
                               seed=seed)
    return {"r": rank, "rho": decay, "seed": seed, "manifold": rank,
            "pr_pos": measured, "pr_pred": predicted,
            "MG": scored["MG"], "LB": scored["LB"], "PRdelay": scored["PRdelay"],
            "specPR256": scored["specPR256"], "roughness": scored["roughness"],
            "frac_degenerate": scored["frac_degenerate"]}


@experiment(
    id="valid.anisotropy",
    title="Whether the estimand is a dimension or an effective rank",
    paper=("sec:aniso", "tab:aniso", "fig:aniso"),
    device=CPU,
    minutes=4,
    promotes=("aniso_raw.csv", "aniso_summary.csv"),
    tier=1,
    notes="Every other construction here equalises the drive, which makes the two "
          "candidate estimands numerically equal. This separates them.",
)
def anisotropy(ctx: Context) -> None:
    import itertools

    import pandas as pd

    ranks, decays, seeds = ANISO_RANKS, ANISO_DECAY, ANISO_SEEDS
    if ctx.fast:
        ranks, decays, seeds = (2, 4), (1.0, 0.5), (0,)

    cfg = _frozen()
    ctx.config(ranks=list(ranks), decay=list(decays), seeds=list(seeds),
               length=ANISO_LENGTH, f0=ANISO_F0, configuration=cfg.tag())
    ctx.declare_seeds("drive_phases", "matrix_base")

    cells = [(rank, decay, seed, tuple(cfg.as_dict().items()))
             for rank, decay, seed in itertools.product(ranks, decays, seeds)]
    raw = pd.DataFrame(map_ordered(_aniso_cell, cells, jobs=ctx.jobs,
                                   desc="valid.anisotropy"))
    ctx.store.table("aniso_raw.csv", raw)

    usable = raw[raw["frac_degenerate"].fillna(1.0) < 0.5]
    summary = (usable.groupby(["r", "rho"], sort=True)
               .agg(pr_pos=("pr_pos", "median"), pr_pred=("pr_pred", "median"),
                    MG=("MG", "median"), MG_sd=("MG", "std"),
                    PRdelay=("PRdelay", "median"), n=("MG", "size")).reset_index())
    ctx.store.table("aniso_summary.csv", summary)

    # The verdict the experiment exists to reach, in one number per side.
    ctx.note("mean_error_against_manifold_dimension",
             mae(summary["MG"], summary["r"]))
    ctx.note("mean_error_against_effective_rank",
             mae(summary["MG"], summary["pr_pos"]))


# ============================================================== valid.geometry

#: Both controls run over these seeds. Eight, because the one-clock arm's estimate is a
#: property of one frequency rather than of four and is correspondingly more variable across
#: seeds; the verdict is taken on the median, and eight is enough for one to mean something.
GEOMETRY_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
#: The scheduled trace: three segments, four-dimensional either side of a one-dimensional
#: middle, crossfaded over ``RAMP`` samples at each switch.
GEOMETRY_SEGMENT = 12_000
GEOMETRY_RAMP = 1_200
#: The level and warp arms are stationary, so they need only enough record for a few
#: windows at the frozen length.
GEOMETRY_RECORD = 16_000
#: Short enough to localise the switch, long enough to resolve four phases in this
#: construction. Both are the frozen eight-direction configuration with the window geometry
#: overridden and no estimator field touched -- the same discipline as the three overrides
#: :mod:`actdim.frozen` states. The archived script wrote the seven fields out as literals,
#: which is how a configuration comes to differ from the one that was calibrated.
GEOMETRY_TRACE_GEOMETRY: Dict[str, int] = {"window": 4_000, "stride": 400}
GEOMETRY_LEVEL_GEOMETRY: Dict[str, int] = {"window": 8_000, "stride": 4_000}

#: **Pre-specified, before the experiment was first run, and not to be moved.** The switch
#: passes when the estimate reads four either side and one in the middle, each within
#: :data:`GEOMETRY_TOLERANCE` components, while the roughness the two arms were matched on
#: changes by less than :data:`GEOMETRY_ROUGHNESS_LIMIT`. The observer control passes when
#: the roughness spans at least :data:`WARP_ROUGHNESS_MINIMUM` of its own smallest value
#: while every median estimate stays within :data:`WARP_ERROR_LIMIT` of four and the
#: estimates span less than :data:`WARP_SPAN_LIMIT`.
GEOMETRY_TOLERANCE = 0.5
GEOMETRY_ROUGHNESS_LIMIT = 0.05
WARP_ROUGHNESS_MINIMUM = 0.30
WARP_SPAN_LIMIT = 0.5
WARP_ERROR_LIMIT = 0.5

#: Where the estimate is read from in the trace, and what is reported beside it. The three
#: companions are the nulls this experiment exists to defeat: if any of them separated the
#: two arms, the estimate would not be carrying anything of its own.
GEOMETRY_TRACE_COLUMNS: Tuple[str, ...] = ("MG", "roughness", "PRdelay", "specPR0")


def _pure_levels(envelope: np.ndarray, hands: int) -> Dict[float, np.ndarray]:
    """Where the crossfade has finished, per level, read off the envelope itself.

    The truth is undefined for the length of a ramp, and the honest way to say where a ramp
    ends is to ask the envelope rather than to recompute the convolution's edges by hand:
    ``mode="same"`` on an even-length kernel puts them half a sample from where the
    arithmetic says, and a guard band derived from the arithmetic is off by that much.
    """
    envelope = np.asarray(envelope, dtype=float)
    return {float(hands): envelope <= 1e-12, 1.0: envelope >= 1.0 - 1e-12}


def _segment_truth(envelope: np.ndarray, left: np.ndarray, right: np.ndarray,
                   hands: int) -> np.ndarray:
    """The true dimension of each window, NaN for any window a ramp reaches into.

    A window lying partly in a ramp has no single true answer. The archived version labelled
    a window by its *centre* and marked only the centres inside the ramp, so a window whose
    centre cleared the switch by one sample was scored against the post-switch level with a
    third of its data from before it -- the defect the errata register's hygiene section
    names for two other experiments. A window is scored here only when every sample in it is
    at one settled level.
    """
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    width = right - left + 1
    truth = np.full(len(left), float("nan"))
    for level, mask in _pure_levels(envelope, hands).items():
        counted = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
        truth[(counted[right + 1] - counted[left]) == width] = level
    return truth


def _geometry_trace_cell(args) -> List[Dict[str, Any]]:
    """One seed of the scheduled 4D -> 1D -> 4D trace."""
    seed, segment, ramp, cfg_values = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..systems import clocks

    length = 3 * int(segment)
    cfg = EstimatorConfig.from_dict(dict(cfg_values))
    pair = clocks.pair(seed, length)
    envelope = clocks.switch_envelope(length, segment, ramp)
    right, traces = windows.sliding(clocks.scheduled(pair, envelope), cfg, seed=seed)
    left = right - cfg.window + 1
    truth = _segment_truth(envelope, left, right, clocks.HANDS)

    return [{"seed": seed, "start": int(a), "right": int(b),
             "centre": float(b) - (cfg.window - 1) / 2.0, "truth": truth[index],
             **{name: float(traces[name][index]) for name in GEOMETRY_TRACE_COLUMNS},
             "degenerate": bool(traces["degenerate"][index] > 0.5)}
            for index, (a, b) in enumerate(zip(left, right))]


def _geometry_level_cell(args) -> List[Dict[str, Any]]:
    """One seed, both arms held at their own level: what does each read?"""
    seed, record, cfg_values = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..systems import clocks

    cfg = EstimatorConfig.from_dict(dict(cfg_values))
    pair = clocks.pair(seed, record)
    rows = []
    for arm, label, series in (("one", "one clock, four hands", pair.one),
                               ("four", "four independent clocks", pair.four)):
        z = (series - series.mean()) / series.std()
        rows.append({"seed": seed, "arm": label, **pair.report(arm),
                     **windows.summarise(z, cfg, seed=seed)})
    return rows


def _geometry_warp_cell(args) -> List[Dict[str, Any]]:
    """One seed of the four-torus, read through each monotone observer scale."""
    seed, record, cfg_values = args

    from ..estimator import windows
    from ..estimator.companions import roughness
    from ..estimator.config import EstimatorConfig
    from ..systems import clocks

    cfg = EstimatorConfig.from_dict(dict(cfg_values))
    pair = clocks.pair(seed, record)
    x = pair.four / pair.four.std()
    rows = []
    for observer in clocks.WARP_NAMES:
        y = clocks.warp(observer, x)
        z = (y - y.mean()) / y.std()
        rows.append({"seed": seed, "observer": observer, "truth": float(clocks.HANDS),
                     "observed_roughness": roughness(y),
                     **windows.summarise(z, cfg, seed=seed)})
    return rows


def _geometry_segments(trace):
    """Per (seed, segment): the median of each statistic over the windows that were scored.

    The windows a ramp reached into carry a NaN truth and are dropped here rather than
    averaged into whichever level they are nearest.
    """
    scored = trace[(~trace["degenerate"]) & trace["truth"].notna()].copy()
    scored["segment"] = np.where(scored["truth"] == 1.0, "one clock", "four clocks")
    return (scored.groupby(["seed", "segment", "truth"], as_index=False)
            .agg(**{name: (name, "median") for name in GEOMETRY_TRACE_COLUMNS},
                 n_windows=("MG", "size")))


def _geometry_verdict(segments, warps) -> Dict[str, Any]:
    """The two pre-specified criteria, applied to the medians over seeds."""
    from ..systems.clocks import HANDS

    levels = segments.groupby("segment").median(numeric_only=True)
    mg_one = float(levels.loc["one clock", "MG"])
    mg_four = float(levels.loc["four clocks", "MG"])
    rough_one = float(levels.loc["one clock", "roughness"])
    rough_four = float(levels.loc["four clocks", "roughness"])
    rough_change = abs(rough_one - rough_four) / rough_four

    by_observer = warps.groupby("observer").median(numeric_only=True)
    rough = by_observer["observed_roughness"]
    rough_span = float((rough.max() - rough.min()) / rough.min())
    mg_span = float(by_observer["MG"].max() - by_observer["MG"].min())
    mg_error = float(np.abs(by_observer["MG"] - float(HANDS)).max())

    return {
        "geometry_mg_one": mg_one,
        "geometry_mg_four": mg_four,
        "geometry_roughness_relative_change": rough_change,
        "geometry_pass": bool(abs(mg_one - 1.0) < GEOMETRY_TOLERANCE
                              and abs(mg_four - float(HANDS)) < GEOMETRY_TOLERANCE
                              and rough_change < GEOMETRY_ROUGHNESS_LIMIT),
        "warp_roughness_relative_span": rough_span,
        "warp_mg_span": mg_span,
        "warp_max_abs_error": mg_error,
        "warp_pass": bool(rough_span >= WARP_ROUGHNESS_MINIMUM
                          and mg_span < WARP_SPAN_LIMIT
                          and mg_error < WARP_ERROR_LIMIT),
    }


@experiment(
    id="valid.geometry",
    title="Two controls separating the estimator from the roughness it is shadowed by",
    paper=("app:nulls", "sec:nuisance"),
    device=CPU,
    minutes=2,
    promotes=("geometry_levels.csv", "geometry_switch_summary.csv", "observer_warps.csv",
              "verdict.json"),
    tier=1,
    notes="Every other null in section 6 is measured beside the estimate on data chosen "
          "for something else. Here the data is built so that the roughness cannot "
          "separate the two arms even in principle: the second arm's frequency is solved "
          "for until it matches the first arm's roughness to the last bit. Both verdicts "
          "were fixed before the first run and are in the code as named constants.",
)
def geometry(ctx: Context) -> None:
    import pandas as pd

    from ..systems.clocks import BAND_MODE, HANDS, WARP_NAMES

    seeds = GEOMETRY_SEEDS
    segment, ramp, record = GEOMETRY_SEGMENT, GEOMETRY_RAMP, GEOMETRY_RECORD
    if ctx.fast:
        # Two seeds so the medians over seeds are medians of something, and a segment long
        # enough that each of the three levels still holds whole windows once the ramps are
        # excluded -- otherwise the branch this experiment exists to exercise never fires.
        seeds = (0, 1)
        segment, ramp, record = 8_000, 800, 9_000

    trace_cfg = _frozen(**GEOMETRY_TRACE_GEOMETRY)
    level_cfg = _frozen(**GEOMETRY_LEVEL_GEOMETRY)
    ctx.config(seeds=list(seeds), segment=segment, ramp=ramp, record=record,
               hands=HANDS, observers=list(WARP_NAMES), band_mode=BAND_MODE,
               trace_configuration=trace_cfg.tag(),
               level_configuration=level_cfg.tag(),
               criteria={"geometry_tolerance": GEOMETRY_TOLERANCE,
                         "geometry_roughness_limit": GEOMETRY_ROUGHNESS_LIMIT,
                         "warp_roughness_minimum": WARP_ROUGHNESS_MINIMUM,
                         "warp_span_limit": WARP_SPAN_LIMIT,
                         "warp_error_limit": WARP_ERROR_LIMIT},
               geometry="frozen eight-direction, window and stride overridden")
    # The frequency layout takes no stream in this mode, which is the point of it here: the
    # four-clock reference is the same on every replicate and only the clocks' phases and
    # amplitudes move. See actdim.systems.clocks.BAND_MODE.
    ctx.declare_seeds("clock_phases", "clock_amplitudes")

    trace_values = tuple(trace_cfg.as_dict().items())
    level_values = tuple(level_cfg.as_dict().items())
    trace = pd.DataFrame([row for cell in map_ordered(
        _geometry_trace_cell, [(seed, segment, ramp, trace_values) for seed in seeds],
        jobs=ctx.jobs, desc="valid.geometry switch") for row in cell])
    levels = pd.DataFrame([row for cell in map_ordered(
        _geometry_level_cell, [(seed, record, level_values) for seed in seeds],
        jobs=ctx.jobs, desc="valid.geometry levels") for row in cell])
    warps = pd.DataFrame([row for cell in map_ordered(
        _geometry_warp_cell, [(seed, record, level_values) for seed in seeds],
        jobs=ctx.jobs, desc="valid.geometry warps") for row in cell])

    segments = _geometry_segments(trace)
    ctx.store.table("geometry_switch_trace.csv", trace)
    ctx.store.table("geometry_switch_summary.csv", segments)
    ctx.store.table("geometry_levels.csv", levels)
    ctx.store.table("observer_warps.csv", warps)

    verdict = _geometry_verdict(segments, warps)
    ctx.store.json("verdict.json", verdict)
    for key, value in verdict.items():
        ctx.note(key, value)
    ctx.note("windows_excluded_by_a_ramp",
             int(trace["truth"].isna().sum()) if len(trace) else 0)
    # The matching is the premise of the whole control, so it is reported as a measurement
    # and not asserted: the largest relative gap between the two arms' roughness over every
    # seed. It should be at the level of the last bits of a float.
    matched = levels.pivot_table(index="seed", columns="arm", values="matched_roughness")
    if matched.shape[1] == 2:
        left, right = (matched[name].to_numpy(dtype=float) for name in matched.columns)
        ctx.note("worst_roughness_mismatch",
                 float(np.max(np.abs(left - right) / np.abs(right))))

    # TwoNN is in the levels table and it is not readable on the one-clock arm: the near
    # recurrences of a circle sampled on an integer grid arrive in plus-and-minus pairs at
    # the same phase offset, so the two nearest neighbours of a point are equidistant, the
    # ratio it is built on is one, and its denominator collapses. It reaches four thousand on
    # one seed here and 0.44 on one in the archived run. Nothing in the verdict uses it; the
    # spread is recorded so that a reader who meets the number knows it was seen.
    circle = levels[levels["arm"].str.startswith("one")]
    if len(circle):
        ctx.note("twonn_range_on_the_one_clock_arm",
                 [float(circle["TwoNN"].min()), float(circle["TwoNN"].max())])


# ============================================================== valid.transitions

TRANSITION_LEVELS: Tuple[Tuple[int, int], ...] = ((6, 2), (8, 3), (4, 1))
TRANSITION_SEEDS: Tuple[int, ...] = (0, 1, 2, 3)
TRANSITION_SEGMENT = 15_000
TRANSITION_BURN = 4_000
#: The detector fires when the estimate crosses the pre-switch median by this many of that
#: segment's own window-to-window scatter, and stays across for this many windows.
TRANSITION_Z = 3.0
TRANSITION_HOLD = 2


def _transition_cell(args) -> List[Dict[str, Any]]:
    mode, high, low, seed, segment, burn, panel, extra = args

    from .. import frozen as frozen_mod
    from ..estimator import windows
    from ..linalg import rank_report
    from ..systems.digits_parameter import F_FAST, Schedules, ten_direction, trajectory

    length = burn + 3 * segment
    schedule = np.full(length, high, dtype=int)
    schedule[burn + segment:burn + 2 * segment] = low
    settings: Dict[str, Any] = dict(k=high, window=length - burn, burn=burn, eta=0.15,
                                    precondition=True, f0=F_FAST)
    settings.update(dict(mode="qp", drive_amp=0.8, noise_amp=0.0) if mode == "qp"
                    else dict(mode="noise", drive_amp=0.0, noise_amp=0.08))
    settings.update(dict(extra))
    config = ten_direction(**settings)

    # The trajectory rather than the simulation, because the ground truth wanted here is
    # per segment and has to be measured on the coordinates of that segment alone.
    series, coordinates, _, _, condition, _ = trajectory(
        config, seed, Schedules(rank=schedule))

    # The ground truth of each segment, measured on the segment itself rather than taken
    # from the schedule: what the construction achieved is a measurement.
    bounds = [(0, segment), (segment, 2 * segment), (2 * segment, 3 * segment)]
    truth = [rank_report(coordinates[a:b], center=True).effective_rank
             for a, b in bounds]

    cfg = frozen_mod.eight_direction()
    rows = []
    for name in panel:
        values = series[name]
        if float(values.std()) <= 1e-12:
            continue
        right, traces = windows.sliding(values, cfg, seed=seed)
        estimate = traces["MG"]
        usable = (~(traces["degenerate"] > 0.5)) & np.isfinite(estimate)
        left = right - cfg.window + 1

        levels = []
        for a, b in bounds:
            # Only windows lying entirely inside a segment: a window that straddles a
            # switch has no single true answer, and scoring it against the rank at its
            # right edge -- which two archived experiments did -- scores part of its data
            # against the wrong level.
            inside = usable & (left >= a) & (right < b)
            levels.append(_median(estimate[inside]))

        lags = []
        for switch, before in ((segment, 0), (2 * segment, 1)):
            window = usable & (left >= bounds[before][0]) & (right < bounds[before][1])
            if int(window.sum()) < 3:
                lags.append(float("nan"))
                continue
            centre = _median(estimate[window])
            scatter = float(np.std(estimate[window]))
            falling = before == 0
            threshold = centre - TRANSITION_Z * scatter if falling \
                else centre + TRANSITION_Z * scatter
            crossed = (estimate < threshold) if falling else (estimate > threshold)
            hit = float("nan")
            for index in np.flatnonzero(usable & crossed & (right >= switch)):
                held = slice(index, index + TRANSITION_HOLD)
                if (index + TRANSITION_HOLD <= len(estimate) and usable[held].any()
                        and bool(np.all(crossed[held][usable[held]]))):
                    hit = float(right[index] - switch)
                    break
            lags.append(hit)

        rows.append({"observer": name, "mode": mode, "hi": high, "lo": low, "seed": seed,
                     "level0": levels[0], "level1": levels[1], "level2": levels[2],
                     "truth0": truth[0], "truth1": truth[1], "truth2": truth[2],
                     "lag_down": lags[0], "lag_up": lags[1], "window": cfg.window,
                     "detected_down": bool(np.isfinite(lags[0])),
                     "detected_up": bool(np.isfinite(lags[1])),
                     "drive_cond": condition, "n_windows": int(usable.sum()),
                     "eta": config.eta, "precondition": config.precondition,
                     "drive_amp": config.drive_amp, "noise_amp": config.noise_amp})
    return rows


@experiment(
    id="valid.transitions",
    title="Whether a change in the active dimension is detected when one occurs",
    paper=("app:validity", "tab:switch"),
    device=CPU,
    minutes=52,
    promotes=("transitions_raw.csv",),
    tier=1,
    notes="Three segments, high to low and back, with nothing else altered. The "
          "threshold uses the pre-switch segment alone; a midpoint of the two observed "
          "levels would hand the detector the answer.",
)
def transitions(ctx: Context) -> None:
    import pandas as pd

    from ..observers import PAPER_TWELVE

    panel = PAPER_TWELVE
    levels, seeds = TRANSITION_LEVELS, TRANSITION_SEEDS
    segment, burn = TRANSITION_SEGMENT, TRANSITION_BURN
    modes = ("qp", "noise")
    if ctx.fast:
        panel = ("w_fro", "loss_full")
        levels, seeds = ((6, 2),), (0,)
        # A segment has to hold at least three whole windows or the detector has no
        # pre-switch scatter to set its threshold from, and the branch that finds a lag is
        # never reached.
        segment, burn = 12_000, 1_000

    ctx.config(levels=[list(pair) for pair in levels], seeds=list(seeds), modes=list(modes),
               segment=segment, burn=burn, observers=list(panel),
               threshold=f"pre-switch median -/+ {TRANSITION_Z} scatters, held "
                         f"{TRANSITION_HOLD} windows")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions")

    cells = [(mode, high, low, seed, segment, burn, panel, _shrink(ctx.fast))
             for mode in modes for high, low in levels for seed in seeds]
    collected = map_ordered(_transition_cell, cells, jobs=ctx.jobs,
                            desc="valid.transitions")
    raw = pd.DataFrame([row for cell in collected for row in cell])
    ctx.store.table("transitions_raw.csv", raw)
    ctx.note("detection_rate",
             {mode: float(group[["detected_down", "detected_up"]].to_numpy().mean())
              for mode, group in raw.groupby("mode")})


# ============================================================== valid.theiler.cap

#: The two exclusions appendix N compares: the cap the implementation imposed, and the one
#: the twenty-direction configuration's own rule asks for, ``(40 - 1) * 16``.
THEILER_ARMS: Tuple[Tuple[str, int], ...] = (("capped_150", 150), ("span_624", 624))
CAP_RANKS: Tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20)
#: High enough that an explicit integer exclusion is never clipped. The cap is a field of
#: the configuration, so this experiment states it once instead of writing to a module
#: global from inside a worker, which is what the archived version did.
UNCAPPED = 10 ** 9


def _cap_cell(args) -> List[Dict[str, Any]]:
    path, label, exclusion, panel, cfg_values = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from .calibration import k20_load

    logs, info = k20_load(Path(path), panel)
    cfg = EstimatorConfig.from_dict(dict(cfg_values)).replace(theiler=int(exclusion),
                                                              theiler_cap=UNCAPPED)
    rows = []
    for name in panel:
        series = logs[name]
        if not np.isfinite(series).all() or float(series.std()) <= 1e-12:
            continue
        scored = windows.summarise(series, cfg, seed=int(info["seed"]))
        rows.append({"arm": label, "theiler": int(exclusion), "r": int(info["r"]),
                     "seed": int(info["seed"]), "observer": name,
                     "truth": float(info["traj_pr"]),
                     "MG": scored["MG"], "LB": scored["LB"],
                     "PRdelay": scored["PRdelay"],
                     "theiler_used": scored["theiler_used"],
                     "degenerate": float(scored["frac_degenerate"]) > 0.5})
    return rows


@experiment(
    id="valid.theiler.cap",
    title="What the exclusion cap costs at the twenty-direction configuration",
    paper=("app:theiler", "tab:theiler"),
    device=CPU,
    minutes=3,
    needs=("calib.e20",),
    promotes=("theiler_quick_raw.csv",),
    tier=1,
    notes="Re-scores the stored trajectories at both exclusions, so the systems cannot "
          "drift between the two settings. The exclusion is set through the "
          "configuration; the archived version wrote a module global from inside its "
          "workers (errata 2).",
)
def theiler_cap(ctx: Context) -> None:
    import pandas as pd

    from .. import frozen as frozen_mod
    from ..observers import K20_PANEL

    import re

    from ..runtime.store import is_plumbing_check

    panel = ("w_fro", "c_norm", "g_fro")
    upstream = ctx.input_dir("calib.e20")
    if is_plumbing_check(upstream) and not ctx.fast:
        # The trajectories are the input to this appendix, and a smoke test's have the
        # right names and the wrong numbers. Refusing is the alternative to a published
        # exclusion comparison quietly measured on a four-hundred-step backbone.
        raise ValueError(
            f"{upstream} was produced by a --fast run, so its trajectories are a "
            f"plumbing check.\nRun it for real first:  python -m actdim run calib.e20")
    directory = upstream / "trajectories"
    # Matched rather than globbed: ``qp_r*_s*.npz`` also catches ``qp_rotate_r10_s00``,
    # and the two fixed-rank controls are not what this appendix re-scores.
    pattern = re.compile(r"^(?P<arm>.+)_r(?P<rank>\d+)_s(?P<seed>\d+)$")
    available = []
    for path in sorted(directory.glob("*.npz")) if directory.is_dir() else []:
        found = pattern.match(path.stem)
        if found and found.group("arm") == "qp":
            available.append((path, int(found.group("rank"))))
    if not available:
        raise FileNotFoundError(
            f"no twenty-direction recurrent trajectories under {directory}.\n"
            f"Run it first:  python -m actdim run calib.e20")

    chosen = [path for path, rank in available if rank in CAP_RANKS]
    if ctx.fast:
        chosen = (chosen or [path for path, _ in available])[:2]
        panel = panel[:2]
    if not chosen:
        raise FileNotFoundError(
            f"none of the stored trajectories is at a rank in {CAP_RANKS}; "
            f"calib.e20 appears to have been run with --fast")

    cfg = frozen_mod.twenty_direction()
    with np.load(chosen[0], allow_pickle=False) as stored:
        probe = stored[f"log__{panel[0]}"]
    if len(probe) < cfg.window:
        raise ValueError(
            f"the stored records are {len(probe)} samples and the twenty-direction "
            f"window is {cfg.window}, so no window can be scored on them")

    ctx.config(ranks=list(CAP_RANKS), observers=list(panel),
               arms={label: exclusion for label, exclusion in THEILER_ARMS},
               configuration=cfg.tag(), trajectories=len(chosen),
               panel_stored=list(K20_PANEL))
    cells = [(str(path), label, exclusion, panel, tuple(cfg.as_dict().items()))
             for path in chosen for label, exclusion in THEILER_ARMS]
    collected = map_ordered(_cap_cell, cells, jobs=ctx.jobs, desc="valid.theiler.cap")
    raw = pd.DataFrame([row for cell in collected for row in cell])
    ctx.store.table("theiler_quick_raw.csv", raw)

    summary = {}
    for label, _ in THEILER_ARMS:
        arm = raw[(raw["arm"] == label) & (~raw["degenerate"])]
        errors, correlations = [], []
        for _, group in arm.groupby("observer"):
            cell = group.groupby("r", sort=True).agg(
                MG=("MG", "median"), truth=("truth", "median")).reset_index()
            errors.append(mae(cell["MG"], cell["truth"]))
            correlations.append(spearman(cell["MG"], cell["truth"]))
        summary[label] = {"median_mae": _median(errors), "median_rho": _median(correlations)}
    ctx.note("by_arm", summary)


# ============================================================== valid.theiler.contrast

CONTRAST_ARMS: Tuple[Tuple[str, str], ...] = (("fast", "qp"), ("slow", "qp_slow"),
                                              ("transient", "gd"))
CONTRAST_RANKS: Tuple[int, ...] = (2, 4, 6, 8)
CONTRAST_SEEDS: Tuple[int, ...] = (0, 1, 2)
CONTRAST_OBSERVERS: Tuple[str, ...] = ("w_fro", "c_proj1", "g_fro", "loss_probe")
#: The sweep. ``frozen`` is the configuration's own autocorrelation rule with the cap the
#: implementation imposes; ``uncapped`` is the same rule with the cap lifted. Both are
#: settings of :class:`~actdim.estimator.config.EstimatorConfig`, which is the point: the
#: published value near 29 is the value at the cap, and until the cap was a field there was
#: no way to say so from the stored table.
CONTRAST_EXCLUSIONS: Tuple[Any, ...] = (0, 1, 2, 5, 10, 20, 50, 100, 150, "frozen",
                                        "uncapped")
#: The trace the article plots, and where it is taken from.
TRACE_RANK, TRACE_SEED, TRACE_OBSERVER, TRACE_SAMPLES = 4, 0, "w_fro", 8000
TRACE_ARMS: Tuple[Tuple[str, str], ...] = (("fast", "recurrent"),
                                           ("transient", "transient"))


def _pool(distances: np.ndarray, cfg) -> Dict[str, Any]:
    """The pooled estimate on an already-filtered block of neighbour distances.

    The one place in this package that reproduces
    :func:`actdim.estimator.mle.estimate_from` rather than calling it, because the sweep
    below has to share a single neighbour query across eleven exclusions and the estimator
    owns its own query. ``test_validity`` pins the two against each other cell by cell, so
    the duplication is checked rather than trusted.
    """
    floored = np.maximum(distances, cfg.floor_distance)
    floor_fraction = float((floored <= cfg.floor_distance * 1.000001).mean())
    sums = np.sum(np.log(floored[:, -1:] / floored[:, :-1]), axis=1)
    sum_floor_fraction = float((sums <= cfg.floor_ratio_sum).mean())
    sums = np.maximum(sums, cfg.floor_ratio_sum)
    count, total = len(sums), float(sums.sum())
    pooled = per_point = float("nan")
    if np.isfinite(total) and total > 0.0:
        pooled = (count * (cfg.k_neighbors - 1) - 1) / total
        local = (cfg.k_neighbors - 1) / sums
        local = local[np.isfinite(local)]
        per_point = float(np.mean(local)) if len(local) else float("nan")
    return {"MG": float(pooled), "LB": float(per_point), "S_med": float(np.median(sums)),
            "frac_floor": floor_fraction, "frac_sumfloor": sum_floor_fraction,
            "degenerate": bool(floor_fraction > cfg.degenerate_fraction
                               or sum_floor_fraction > cfg.degenerate_fraction)}


def _exclusion_cells(window: np.ndarray, cfg, seed: int,
                     exclusions: Sequence[Any]) -> List[Dict[str, Any]]:
    """Every exclusion in the sweep, on one window, from one neighbour query.

    The query asks for ``k + 2 W_max + 1`` candidates, which is exactly what the estimator
    asks for at the largest exclusion, so filtering the same candidate list at a smaller
    one reproduces its neighbour set exactly. It is done in row blocks because the largest
    exclusion here is the uncapped autocorrelation rule, about 1600 samples on the
    transient arm, and a single query would need a third of a gigabyte per window.
    """
    from sklearn.neighbors import KDTree

    from ..estimator.companions import (autocorrelation_time, delay_participation_ratio,
                                        roughness)
    from ..estimator.embedding import (delay_embedding, dither, resolve_tau,
                                       resolve_theiler, standardise)

    x = np.asarray(window, dtype=float)
    z = dither(standardise(x), cfg.dither, np.random.default_rng(seed))
    tau = resolve_tau(cfg, x)
    # The two named settings are the configuration's own rule, with the cap the
    # implementation imposes and without it. Both come out of the configuration object;
    # neither is a module global a worker writes to.
    # The frozen rule and the autocorrelation rule, the second with the cap lifted. The
    # rule is named here rather than taken from the configuration: the eight-direction
    # selection returned "embedding" when it was recalibrated, and the embedding span does
    # not depend on the cap, so "the configuration's rule, uncapped" measured the frozen
    # setting twice. It is the autocorrelation rule the appendix is about -- on a monotone
    # decay its time is some 1600 samples against an embedding span of 76, and the whole
    # question is what the estimate does when that exclusion is actually applied.
    named = {"frozen": resolve_theiler(cfg, z, tau),
             "uncapped": resolve_theiler(
                 cfg.replace(theiler="autocorr", theiler_cap=UNCAPPED), z, tau)}
    wanted = [(str(label), int(named[label] if label in named else label))
              for label in exclusions]
    points = delay_embedding(z, cfg.max_E, tau)
    count, k = len(points), cfg.k_neighbors
    widest = max(exclusion for _, exclusion in wanted)
    depth = min(count, k + 2 * widest + 1)

    base = {"n_points": int(count), "tau_used": int(tau), "n_query": int(depth),
            "PRdelay": delay_participation_ratio(points),
            "acorr": float(autocorrelation_time(z)), "roughness": roughness(x)}
    if depth < k + 1:
        return [{"theiler_label": label, "theiler_used": exclusion, **base,
                 "MG": float("nan"), "LB": float("nan"), "degenerate": True}
                for label, exclusion in wanted]

    tree = KDTree(points)
    kept: Dict[str, List[np.ndarray]] = {label: [] for label, _ in wanted}
    times: Dict[str, List[np.ndarray]] = {label: [] for label, _ in wanted}
    enough = {label: True for label, _ in wanted}
    valid_fraction = 0.0
    block = max(1, min(count, 4_000_000 // max(depth, 1)))
    for start in range(0, count, block):
        stop = min(start + block, count)
        distances, indices = tree.query(points[start:stop], k=depth)
        separation = np.abs(indices - np.arange(start, stop)[:, None])
        for label, exclusion in wanted:
            allowed = separation > exclusion
            if exclusion == widest:
                # How much of the query survives the widest exclusion, which is what says
                # whether one query was deep enough to serve every setting in the sweep.
                valid_fraction += float(allowed.sum()) / float(count * depth)
            if int(allowed.sum(axis=1).min()) < k:
                enough[label] = False
                continue
            order = np.argsort(~allowed, axis=1, kind="stable")
            kept[label].append(np.take_along_axis(distances, order, axis=1)[:, :k])
            times[label].append(np.take_along_axis(separation, order, axis=1)[:, :k])

    reference = None
    if enough.get("0") and kept.get("0"):
        floored = np.maximum(np.vstack(kept["0"]), cfg.floor_distance)
        reference = {"r1": float(np.median(floored[:, 0])),
                     "rk": float(np.median(floored[:, -1])),
                     "dt": float(np.median(np.vstack(times["0"]))),
                     "kept": np.vstack(times["0"])}

    records = []
    for label, exclusion in wanted:
        record = {"theiler_label": label, "theiler_used": int(exclusion), **base,
                  "frac_query_valid": valid_fraction,
                  "r1_med_W0": reference["r1"] if reference else float("nan"),
                  "rk_med_W0": reference["rk"] if reference else float("nan"),
                  "dt_med_W0": reference["dt"] if reference else float("nan"),
                  "d_ref": reference["rk"] if reference else float("nan")}
        if not enough[label] or not kept[label]:
            records.append({**record, "MG": float("nan"), "LB": float("nan"),
                            "degenerate": True})
            continue
        distances = np.vstack(kept[label])
        separations = np.vstack(times[label])
        record.update(_pool(distances, cfg))
        floored = np.maximum(distances, cfg.floor_distance)
        record.update(
            n_pairs=int(floored.size),
            r1_med=float(np.median(floored[:, 0])),
            rk_med=float(np.median(floored[:, -1])),
            spread_med=float(np.median(floored[:, -1] / floored[:, 0])),
            dt_med=float(np.median(separations)))
        if reference is not None:
            near = float(reference["rk"])
            record.update(
                frac_kept_from_W0=float((reference["kept"] > exclusion).mean()),
                n_pairs_near_ref=int((floored <= near).sum()),
                frac_near_ref=float((floored <= near).mean()),
                dist_inflation=float(record["r1_med"]
                                     / max(reference["r1"], cfg.floor_distance)))
        records.append(record)
    return records


def _contrast_cell(args) -> Dict[str, Any]:
    """One (arm, rank, seed): simulate, sweep the exclusion over every window."""
    arm, mode, rank, seed, record, burn, panel, exclusions, extra = args

    from .. import frozen as frozen_mod
    from ..estimator.windows import window_starts
    from ..systems import digits_parameter

    simulation = digits_parameter.simulate(sweep_config(mode, rank, record, burn, extra),
                                           seed=seed)
    cfg = frozen_mod.constructed_geometry(frozen_mod.eight_direction(), simulation.length)

    rows: List[Dict[str, Any]] = []
    for name in panel:
        series = simulation[name]
        if not np.isfinite(series).all() or float(series.std()) <= 1e-12:
            continue
        starts = window_starts(len(series), cfg)
        for index, start in enumerate(starts):
            for record_row in _exclusion_cells(series[start:start + cfg.window], cfg,
                                               seed, exclusions):
                rows.append({"arm": arm, "r": rank, "seed": seed, "observer": name,
                             "window": index, "start": int(start),
                             "n_windows": len(starts),
                             "traj_PR": simulation.info["traj_PR"], **record_row})

    trace = None
    if (rank == TRACE_RANK and seed == TRACE_SEED
            and arm in dict(TRACE_ARMS) and TRACE_OBSERVER in simulation.series):
        trace = np.asarray(simulation[TRACE_OBSERVER][:TRACE_SAMPLES], dtype=float)
    return {"rows": rows, "trace": trace, "arm": arm}


@experiment(
    id="valid.theiler.contrast",
    title="Is a transient's active dimension undefined, or only unidentifiable?",
    paper=("app:exclusion", "tab:exclusion", "fig:traces"),
    device=CPU,
    minutes=40,
    promotes=("sweep_windows.csv", "example_traces.csv",
              "exclusion_table.csv"),
    tier=1,
    notes="A two-by-two on the same points with the same estimator, changing only the "
          "Theiler exclusion, plus the sweep between the two ends. The cap is a field of "
          "the configuration here, and the uncapped rule is measured beside it.",
)
def theiler_contrast(ctx: Context) -> None:
    import pandas as pd

    arms, ranks, seeds = CONTRAST_ARMS, CONTRAST_RANKS, CONTRAST_SEEDS
    panel, exclusions = CONTRAST_OBSERVERS, CONTRAST_EXCLUSIONS
    record, burn = SWEEP_RECORD, SWEEP_BURN
    if ctx.fast:
        ranks, seeds = (2, 4), (0,)
        panel = ("w_fro", "loss_probe")
        exclusions = (0, 20, 150, "frozen", "uncapped")
        record, burn = FAST_RECORD, FAST_BURN

    ctx.config(arms=[arm for arm, _ in arms], ranks=list(ranks), seeds=list(seeds),
               observers=list(panel), exclusions=[str(e) for e in exclusions],
               record=record, burn=burn,
               geometry="frozen eight-direction, stride max(500, (n - window) // 6)")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions")

    cells = [(arm, mode, rank, seed, record, burn, panel, exclusions, _shrink(ctx.fast))
             for arm, mode in arms for rank in ranks for seed in seeds]
    collected = map_ordered(_contrast_cell, cells, jobs=ctx.jobs,
                            desc="valid.theiler.contrast")
    raw = pd.DataFrame([row for cell in collected for row in cell["rows"]])
    ctx.store.table("sweep_windows.csv", raw)

    traces: Dict[str, np.ndarray] = {}
    for cell in collected:
        label = dict(TRACE_ARMS).get(cell["arm"])
        if label and cell["trace"] is not None and label not in traces:
            traces[label] = cell["trace"]
    if traces:
        length = min(len(v) for v in traces.values())
        ctx.store.table("example_traces.csv", pd.DataFrame(
            {"sample": np.arange(length),
             **{name: values[:length] for name, values in traces.items()}}))

    # The table appendix P prints, collapsed in a stated order rather than pooled: the
    # median over a run's windows, then over the four observers, then over the three
    # seeds and the four ranks. Pooling instead mixes the window spread into the observer
    # median and moves the fast arm by six hundredths. The spread across cells is carried
    # beside the median, because on the transient arm at the uncapped exclusion it is the
    # finding: the estimate has no level there, only a range.
    cells = (raw.groupby(["arm", "theiler_label", "r", "seed", "observer"],
                         sort=True)["MG"].median())
    table = (cells.groupby(level=[0, 1]).agg(["median", "min", "max", "size"])
             .reset_index().rename(columns={"median": "MG", "size": "n_cells"}))
    ctx.store.table("exclusion_table.csv", table)

    summary = table.pivot(index="theiler_label", columns="arm", values="MG")
    ctx.note("MG_by_exclusion",
             {str(label): {arm: (None if not np.isfinite(value) else float(value))
                           for arm, value in row.items()}
              for label, row in summary.iterrows()})


# ============================================================== valid.ceiling

CEILING_RANKS: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 20)
CEILING_SEEDS: Tuple[int, ...] = (0, 1, 2)
CEILING_EMBEDDINGS: Tuple[int, ...] = (10, 14, 20, 28, 40, 56)
CEILING_LENGTHS: Tuple[int, ...] = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
CEILING_REFERENCE_N = 8000
CEILING_REFERENCE_E = 20
CEILING_WIDE_E = 56
CEILING_WIDE_LENGTHS: Tuple[int, ...] = (1000, 2000, 4000, 8000, 16000)
CEILING_BURN = 2000
CEILING_TAU = 4
#: The three arms of the embedding scan. ``E_max`` cannot be raised at fixed ``tau``
#: without also raising the delay span and, through the autocorrelation rule, the
#: exclusion, so each arm holds one of those fixed and the three together say which is
#: doing the work. Each has a defect and both are stated rather than hidden: the fixed
#: exclusion is smaller than its own embedding span above ``E_max = 20``, and the fixed
#: span changes ``tau``, which section 6.2 shows the estimate is sensitive to.
CEILING_ARMS: Tuple[str, ...] = ("frozen", "theiler76", "fixedspan")
#: The identifiability ratio needs a second embedding at twice ``E_max``. On the longest
#: records that is the most expensive cell in the study, so it is computed on one seed, and
#: at the two longest records on one rank.
CEILING_IDENT_SEED = 0
CEILING_IDENT_BIG: Tuple[int, ...] = (32000, 64000)
CEILING_IDENT_BIG_RANKS: Tuple[int, ...] = (20,)
#: Above this record length the cells that also compute the doubled embedding become
#: memory bound rather than processor bound: at 64,000 samples one of them holds a third of
#: a gigabyte of candidate distances, and eight at once is how an overnight run dies. Only
#: those cells get fewer workers. The ordinary cells at the same length resolve their
#: exclusion to the embedding span rather than to the cap, so they hold a fifth as much and
#: run at the full worker count -- which is most of this experiment's cost.
CEILING_HEAVY_FROM = 32000
CEILING_HEAVY_JOBS = 3


#: The delay span at the frozen configuration, in samples, which two of the three arms
#: hold fixed while the third lets it grow with the embedding dimension.
CEILING_SPAN = (CEILING_REFERENCE_E - 1) * CEILING_TAU


def _ceiling_arm(arm: str, embedding: int) -> Tuple[Any, int]:
    """The exclusion and the lag of one arm at one embedding dimension."""
    if arm == "frozen":
        return "autocorr", CEILING_TAU
    if arm == "theiler76":
        return CEILING_SPAN, CEILING_TAU
    if arm == "fixedspan":
        return "autocorr", max(1, int(round(CEILING_SPAN / (embedding - 1))))
    raise ValueError(f"unknown ceiling arm {arm!r}")


def _ceiling_path(directory: str, rank: int, seed: int) -> Path:
    return Path(directory) / "trajectories" / f"qp_r{rank:02d}_s{seed:02d}.npz"


def _ceiling_simulate(args) -> Dict[str, Any]:
    """One long twenty-direction trajectory, and its measured rank at every prefix.

    The record is lengthened, never resampled: every length in the scan is a *prefix* of
    this one trajectory, so the sampling rate, the drive frequencies and the delay lag are
    identical at every one of them and a longer record is a longer record.
    """
    rank, seed, lengths, burn, directory, panel, extra = args

    from ..linalg import TRAJECTORY_RANK_TOL, rank_report
    from ..systems.digits_parameter import trajectory

    from .calibration import k20_system

    config = k20_system("qp", rank, max(lengths), burn, extra)
    series, coordinates, _, drive, condition, _ = trajectory(config, seed)
    truth = {str(n): rank_report(coordinates[:n], center=True,
                                 tol=TRAJECTORY_RANK_TOL)[:2] for n in lengths}

    path = _ceiling_path(directory, rank, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        info=json.dumps({"r": rank, "seed": seed,
                         "truth": {k: [int(v[0]), float(v[1])] for k, v in truth.items()},
                         "margin_res": float(drive.margin),
                         "drive_cond": float(condition)}),
        **{f"log__{name}": series[name] for name in panel})
    return {"r": rank, "seed": seed, "samples": int(len(coordinates))}


def _ceiling_cell(args) -> Optional[Dict[str, Any]]:
    """One (arm, E_max, N, rank, seed, observer) cell: one window, one estimate."""
    (sweep, arm, embedding, length, rank, seed, observer, ident, exclusion, tau,
     directory, cfg_values) = args

    from ..estimator import windows
    from ..estimator.config import EstimatorConfig
    from ..estimator.diagnostics import ratio, trend_crossings

    path = _ceiling_path(directory, rank, seed)
    with np.load(path, allow_pickle=False) as stored:
        info = json.loads(str(stored["info"]))
        series = np.asarray(stored[f"log__{observer}"], dtype=float)[:length]
    hard, effective = info["truth"][str(length)]
    if not np.isfinite(series).all() or float(series.std()) <= 1e-12:
        return None

    cfg = EstimatorConfig.from_dict(dict(cfg_values)).replace(
        max_E=embedding, window=length, stride=length, theiler=exclusion, tau=tau)
    scored = windows.score(series, cfg, seed=seed)
    row = {
        "sweep": sweep, "arm": arm, "max_E": embedding, "N": length, "r": rank,
        "seed": seed, "observer": observer, "tau": tau, "truth_r": rank,
        "traj_rank": hard, "traj_pr": effective,
        "MG": scored["MG"], "LB": scored["LB"], "TwoNN": scored["TwoNN"],
        "PRdelay": scored["PRdelay"], "specPR256": scored["specPR256"],
        "roughness": scored["roughness"], "acorr": scored["acorr"],
        "degenerate": bool(scored["degenerate"]),
        "frac_floor": scored["frac_floor"], "frac_sumfloor": scored["frac_sumfloor"],
        "tau_used": scored["tau_used"], "theiler_used": scored["theiler_used"],
        "theiler_requested": ((embedding - 1) * int(tau) if exclusion == "autocorr"
                              else int(exclusion)),
        "embed_span": (embedding - 1) * int(tau),
        "n_delay_vectors": max(0, length - (embedding - 1) * int(tau)),
        "crossings": trend_crossings(series),
        "margin_res": info["margin_res"], "drive_cond": info["drive_cond"],
        "MG_2E": float("nan"), "rho_ident": float("nan"),
        "theiler_used_2E": float("nan"),
    }
    if ident:
        doubled = windows.score(series, cfg.replace(max_E=2 * embedding), seed=seed)
        row["MG_2E"] = doubled["MG"]
        row["theiler_used_2E"] = doubled["theiler_used"]
        row["rho_ident"] = ratio(scored["MG"], doubled["MG"])
    return row


def _ceiling_jobs(embeddings, lengths, wide_lengths, ranks, seeds, arms, reference_n,
                  reference_e, wide_e, directory, cfg_values):
    jobs = []
    for arm in arms:
        for embedding in embeddings:
            exclusion, tau = _ceiling_arm(arm, embedding)
            for rank in ranks:
                for seed in seeds:
                    for observer in CEILING_PANEL:
                        jobs.append(("E", arm, embedding, reference_n, rank, seed,
                                     observer,
                                     arm == "frozen" and seed == CEILING_IDENT_SEED,
                                     exclusion, tau, directory, cfg_values))
    for label, embedding, grid in (("N", reference_e, lengths),
                                   (f"N_E{wide_e}", wide_e, wide_lengths)):
        for length in grid:
            for rank in ranks:
                for seed in seeds:
                    big = length in CEILING_IDENT_BIG
                    for observer in CEILING_PANEL:
                        jobs.append((label, "frozen", embedding, length, rank, seed,
                                     observer,
                                     label == "N" and seed == CEILING_IDENT_SEED
                                     and (not big or rank in CEILING_IDENT_BIG_RANKS),
                                     "autocorr", CEILING_TAU, directory, cfg_values))
    # Shortest records first, and grouped by trajectory: the cost per cell grows faster
    # than linearly in the record, so this order leaves a usable scan on disk at every
    # checkpoint rather than only at the end.
    jobs.sort(key=lambda job: (job[3], job[4], job[5], -job[2]))
    return jobs


def _crossing_point(ranks, values, tolerance: float) -> float:
    """The first rank at which the estimate falls ``tolerance`` behind the truth.

    Interpolated in the rank, and ``inf`` when the estimate never falls that far behind on
    the grid -- a ceiling above the top of the grid is not a missing value.
    """
    x = np.asarray(ranks, dtype=float)
    gap = x - np.asarray(values, dtype=float)
    good = np.isfinite(gap)
    x, gap = x[good], gap[good]
    if len(x) < 2:
        return float("nan")
    for index in range(len(x)):
        if gap[index] > tolerance:
            if index == 0:
                return float(x[0])
            x0, x1, y0, y1 = x[index - 1], x[index], gap[index - 1], gap[index]
            return float(x0 + (tolerance - y0) * (x1 - x0) / (y1 - y0)) \
                if y1 != y0 else float(x1)
    return float("inf")


def _ceilings(block, tolerances=(0.5, 1.0, 2.0)) -> Dict[str, Any]:
    """Both operational ceilings for one (arm, E_max, N) block of the scan."""
    usable = block[~block["degenerate"].astype(bool)]
    cell = (usable.groupby("r", sort=True)
            .agg(MG=("MG", "median"), truth=("traj_pr", "median"),
                 PRdelay=("PRdelay", "median")).sort_index())
    out: Dict[str, Any] = {
        f"r_track_{t:g}": _crossing_point(cell.index.values, cell["MG"].values, t)
        for t in tolerances}
    top = cell.loc[cell.index >= 16, "MG"]
    out["MG_plateau"] = _median(top)
    out["MG_at_20"] = float(cell["MG"].loc[20]) if 20 in cell.index else float("nan")
    out["MG_at_8"] = float(cell["MG"].loc[8]) if 8 in cell.index else float("nan")
    # The same two ceilings for the linear null. The delay participation ratio knows
    # nothing about manifolds, neighbours or embedding theorems, and its only hard limit
    # is E itself: if it ceilings where the estimate does, the ceiling is a statement about
    # how many components the delay window resolves and not about geometry.
    out["PR_track_1"] = _crossing_point(cell.index.values, cell["PRdelay"].values, 1.0)
    out["PR_plateau"] = _median(cell.loc[cell.index >= 16, "PRdelay"])
    high = cell.loc[cell.index >= 8]
    out["slope_top"] = _slope(high.index.values, high["MG"].values)
    out["slope_top_PR"] = _slope(high.index.values, high["PRdelay"].values)
    out["n_r_dropped"] = int(block["r"].nunique() - cell.index.nunique())
    return out


def _ceiling_fits(points, values) -> Dict[str, Any]:
    """Which of the two candidate bounds, if either, explains the measured ceilings.

    Each bound is fitted with one free scale, so what is tested is the *shape* and not the
    constant; the unscaled versions are reported beside them, since a form that only fits
    once its coefficient has moved a long way is not the same claim. The fourth model is
    suggested by the data rather than by either hypothesis: the finite-record bound is
    already logarithmic in the record, the embedding condition is not logarithmic in
    ``E_max``, so a good fit there says the ceiling is not obeying the embedding condition.
    """
    embedding = np.array([p[0] for p in points], dtype=float)
    length = np.array([p[1] for p in points], dtype=float)
    y = np.asarray(values, dtype=float)
    good = np.isfinite(y) & np.isfinite(embedding) & np.isfinite(length)
    embedding, length, y = embedding[good], length[good], y[good]
    if len(y) < 4:
        return {}
    takens, eckmann = embedding / 2.0, 2.0 * np.log10(length)

    def error(prediction) -> float:
        return float(np.sqrt(np.mean((y - prediction) ** 2)))

    a = float((takens @ y) / (takens @ takens))
    b = float((eckmann @ y) / (eckmann @ eckmann))
    grid = np.linspace(0.2, 3.0, 141)
    best = min((error(np.minimum(u * takens, v * eckmann)), u, v)
               for u in grid for v in grid)
    design = np.column_stack([np.log10(embedding), np.log10(length),
                              np.ones_like(embedding)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"n": int(len(y)), "rmse_takens": error(takens), "rmse_er": error(eckmann),
            "rmse_min": error(np.minimum(takens, eckmann)),
            "a_takens": a, "rmse_takens_fit": error(a * takens),
            "b_er": b, "rmse_er_fit": error(b * eckmann),
            "a_min": best[1], "b_min": best[2], "rmse_min_fit": best[0],
            "loglog_dE": float(coefficients[0]), "loglog_dN": float(coefficients[1]),
            "loglog_const": float(coefficients[2]),
            "rmse_loglog": error(design @ coefficients),
            "sd_y": float(np.std(y))}


def _ceiling_slopes(summary) -> Any:
    """The sharpest form of each prediction is a slope, not a level."""
    import pandas as pd

    records = []
    for (sweep, arm), group in summary.groupby(["sweep", "arm"], sort=True):
        if sweep == "E":
            x = group["max_E"].to_numpy(dtype=float)
            predicted, units = 0.5, "per unit E_max"
        else:
            x = np.log10(group["N"].to_numpy(dtype=float))
            predicted, units = 2.0, "per decade of N"
        for column in ("r_track_1", "MG_plateau", "MG_at_20", "slope_top",
                       "slope_top_PR"):
            y = pd.to_numeric(group[column], errors="coerce").replace(
                np.inf, np.nan).to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if int(good.sum()) < 3:
                continue
            slope = _slope(x[good], y[good])
            intercept = float(y[good].mean() - slope * x[good].mean())
            records.append({"sweep": sweep, "arm": arm, "quantity": column,
                            "slope": slope, "intercept": intercept,
                            "predicted_slope": predicted, "units": units,
                            "n": int(good.sum())})
    return pd.DataFrame(records)


@experiment(
    id="valid.ceiling",
    title="The ceiling: the embedding dimension against the record length",
    paper=("app:ceiling", "tab:ceiling", "tab:ceilingfit", "fig:ceiling"),
    device=CPU,
    minutes=360,
    promotes=("ceiling_summary.csv", "ceiling_cells.csv", "ceiling_fits.csv",
              "ceiling_slopes.csv"),
    tier=1,
    notes="The longest experiment here, and the estimate is measured rather than "
          "inherited: the archived script claimed three hours, and one cell at 32,000 "
          "samples takes 130 seconds on this machine, which puts the record scan alone "
          "above four. Each record length is a prefix of one trajectory, so the delay lag "
          "keeps its meaning in periods. The raw table is rewritten after every "
          "record-length block, so a run that dies late still leaves a usable scan.",
)
def ceiling(ctx: Context) -> None:
    import pandas as pd

    from ..observers import CEILING_STORED

    embeddings, lengths = CEILING_EMBEDDINGS, CEILING_LENGTHS
    wide_lengths, ranks, seeds = CEILING_WIDE_LENGTHS, CEILING_RANKS, CEILING_SEEDS
    arms, reference_n = CEILING_ARMS, CEILING_REFERENCE_N
    if ctx.fast:
        # The ranks have to straddle the top of the grid or every ceiling measure is
        # undefined: the plateau is a median over r >= 16 and the slope a fit over r >= 8,
        # and a smoke test on low ranks alone would write three empty tables.
        embeddings, lengths = (10, 14, 20), (1000, 2000, 4000)
        wide_lengths, ranks, seeds = (1000, 2000, 4000), (2, 8, 16, 20), (0,)
        reference_n = 2000

    cfg = _frozen(tau=CEILING_TAU)
    directory = str(ctx.store.dir)
    ctx.config(embeddings=list(embeddings), lengths=list(lengths),
               wide_lengths=list(wide_lengths), wide_embedding=CEILING_WIDE_E,
               ranks=list(ranks), seeds=list(seeds), arms=list(arms),
               reference_N=reference_n, reference_E=CEILING_REFERENCE_E,
               observers=list(CEILING_PANEL), stored=list(CEILING_STORED),
               configuration=cfg.tag())
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions")

    all_lengths = tuple(sorted(set(lengths) | set(wide_lengths) | {reference_n}))
    simulated = map_ordered(
        _ceiling_simulate,
        [(rank, seed, all_lengths, CEILING_BURN, directory, tuple(CEILING_STORED),
          _shrink(ctx.fast))
         for rank in ranks for seed in seeds],
        jobs=ctx.jobs, desc="valid.ceiling simulate")
    ctx.note("trajectories", len(simulated))

    jobs = _ceiling_jobs(embeddings, lengths, wide_lengths, ranks, seeds, arms,
                         reference_n, CEILING_REFERENCE_E, CEILING_WIDE_E, directory,
                         tuple(cfg.as_dict().items()))
    blocks: Dict[int, List[Any]] = {}
    for index, job in enumerate(jobs):
        blocks.setdefault(job[3], []).append((index, job))

    # Kept with their job index and sorted before writing, because the heavy cells of a
    # record-length block are dispatched in a second pass and the raw table has to come out
    # in one deterministic order whichever pass produced a row.
    collected: List[Tuple[int, Dict[str, Any]]] = []
    for length in sorted(blocks):
        heavy = length >= CEILING_HEAVY_FROM
        passes = ((("", [pair for pair in blocks[length] if not (heavy and pair[1][7])],
                    ctx.jobs),
                   (" 2E", [pair for pair in blocks[length] if heavy and pair[1][7]],
                    max(1, min(default_jobs(ctx.jobs), CEILING_HEAVY_JOBS))))
                  if heavy else (("", blocks[length], ctx.jobs),))
        for label, batch, workers in passes:
            if not batch:
                continue
            results = map_ordered(_ceiling_cell, [pair[1] for pair in batch], jobs=workers,
                                  desc=f"valid.ceiling N={length}{label}")
            collected.extend((pair[0], row) for pair, row in zip(batch, results)
                             if row is not None)
        # Rewritten after every block, so a run that dies late still leaves a usable scan.
        ctx.store.table("ceiling_raw.csv",
                        pd.DataFrame([row for _, row in sorted(collected,
                                                               key=lambda p: p[0])]))
    rows = [row for _, row in sorted(collected, key=lambda pair: pair[0])]
    raw = pd.DataFrame(rows)

    per_cell = (raw.groupby(["sweep", "arm", "max_E", "N", "r"], sort=True)
                .agg(MG=("MG", "median"), MG_sd=("MG", "std"), LB=("LB", "median"),
                     PRdelay=("PRdelay", "median"), specPR256=("specPR256", "median"),
                     traj_pr=("traj_pr", "median"), traj_rank=("traj_rank", "median"),
                     rho_ident=("rho_ident", "median"), MG_2E=("MG_2E", "median"),
                     crossings=("crossings", "median"),
                     degen_rate=("degenerate", "mean"),
                     frac_floor=("frac_floor", "median"),
                     theiler_used=("theiler_used", "median"),
                     theiler_requested=("theiler_requested", "median"),
                     n_delay_vectors=("n_delay_vectors", "median"),
                     roughness=("roughness", "median"),
                     margin_res=("margin_res", "median"),
                     n=("MG", "size")).reset_index())
    ctx.store.table("ceiling_cells.csv", per_cell)

    records = []
    for (sweep, arm, embedding, length), block in raw.groupby(
            ["sweep", "arm", "max_E", "N"], sort=True):
        records.append({"sweep": sweep, "arm": arm, "max_E": embedding, "N": length,
                        "takens": embedding / 2.0,
                        "eckmann_ruelle": 2.0 * np.log10(length),
                        "theiler_used": _median(block["theiler_used"]),
                        "degen_rate": float(block["degenerate"].mean()),
                        "rho_ident": _median(block["rho_ident"]),
                        "crossings": _median(block["crossings"]),
                        **_ceilings(block)})
    summary = pd.DataFrame(records).sort_values(["sweep", "arm", "max_E", "N"])
    ctx.store.table("ceiling_summary.csv", summary)

    frozen_arm = summary[summary["arm"] == "frozen"].drop_duplicates(
        subset=["max_E", "N"])
    points = list(zip(frozen_arm["max_E"], frozen_arm["N"]))
    fits = {}
    for quantity in ("r_track_1", "MG_plateau", "PR_plateau"):
        values = pd.to_numeric(frozen_arm[quantity], errors="coerce").replace(
            np.inf, np.nan).to_numpy(dtype=float)
        fits[quantity] = _ceiling_fits(points, values)
    fit_frame = pd.DataFrame(fits).T.reset_index().rename(columns={"index": "quantity"})
    ctx.store.table("ceiling_fits.csv", fit_frame)
    ctx.store.table("ceiling_slopes.csv", _ceiling_slopes(summary))
    ctx.note("cells", len(points))

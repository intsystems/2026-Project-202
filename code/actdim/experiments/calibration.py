"""Appendix C: the two estimator configurations, selected once and then frozen.

Everything downstream is measured at one of these two, so the way they are chosen is the
part of the protocol that decides whether any later number may be read as a recovery.
Three rules hold, and each of them is a defect the archived predecessor had.

*The grid contains estimator parameters and nothing else.* The earlier calibration swept
the drive period and the learning rate alongside ``max_E`` and ``tau``, so selecting a
configuration selected the data-generating process as well as the measurement. One system
configuration is simulated per (seed, rank) here and every estimator configuration is
scored on those same logs, so the data cannot move. That separation is what
:class:`actdim.estimator.config.EstimatorConfig` exists to enforce.

*The split is disjoint in both seed and rank.* The eight-direction configuration is
selected on seeds 90-92 and ranks 2, 4, 6; every later experiment uses seeds 0-5 and
reports ranks 1, 3, 5, 8 separately. Withholding seeds alone would have withheld nothing,
because until errata item 1 was fixed the frequency geometry -- the thing the estimator
responds to -- was bit-identical across seeds.

*The objective is scored against the measured effective rank*, never against the nominal
rank. What a construction achieved is a measurement and the two differ; scoring against
the label would credit an estimator for a rank the trajectory never carried.

**One scorer.** The twenty-direction calibration was scored by two programs that wrote the
same three files by different aggregation code, and nothing in the files recorded which
produced the committed copy (errata item 17). They are merged here into the single pass
below, which keeps the aggregation of the parallel scorer -- the one whose columns the
committed ``heldout_qp_summary.csv`` has -- and the stage-one selection of the serial one.

The trajectories are written once and scored twice, because selection has to finish before
the frozen configuration is known and re-simulating between the two stages would mean the
selection and the test ran on different systems.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..runtime import CPU, Context, experiment
from ..runtime.parallel import map_ordered
# The package's one rank correlation. It lives in the ladder module because that was
# written first; importing it is the alternative to a fifth copy, which is what the
# archived tree had -- three of them differing in how they treated a constant input.
from .systems import _spearman as spearman


def mae(estimate, truth) -> float:
    """Mean absolute error over the pairs where both sides are finite.

    The companion of :func:`spearman` above and the other half of every objective in this
    module and in :mod:`actdim.experiments.validity`. An all-NaN comparison returns NaN
    rather than the mean of an empty selection, which is a RuntimeWarning and was how
    three of them reached a committed table.
    """
    a = np.asarray(estimate, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[good] - b[good]))) if good.any() else float("nan")

#: The eight-direction split of table 7: three seeds and three ranks, disjoint from every
#: later experiment in both.
E8_SEEDS: Tuple[int, ...] = (90, 91, 92)
E8_RANKS: Tuple[int, ...] = (2, 4, 6)

#: The twenty-direction split. Its seeds are *not* withheld -- the article says so in
#: table 7 -- which is one of the reasons section 5.2 calls that arm exploratory.
K20_SEEDS: Tuple[int, ...] = (0, 1, 2)
K20_RANKS: Tuple[int, ...] = tuple(range(1, 21))
K20_SELECTION: Tuple[int, ...] = (2, 6, 10, 14, 18)
K20_HELDOUT: Tuple[int, ...] = tuple(r for r in K20_RANKS if r not in K20_SELECTION)

#: The five anchor ranks the twenty-direction robustness arms are run at. The complete
#: grid is not repeated for them because they answer a different question -- whether
#: changing the excitation class at a fixed measured rank changes the estimate -- and five
#: points span the range without obscuring the main test.
K20_ANCHORS: Tuple[int, ...] = (1, 5, 10, 15, 20)
K20_ARMS: Tuple[str, ...] = ("qp_slow", "noise", "batch_proj", "gd")

#: How long a calibration record is. The eight-direction system runs 30,000 steps with the
#: first 4,000 burnt; the twenty-direction one is shorter because there are 82 of them.
E8_RECORD, E8_BURN = 26_000, 4_000
K20_RECORD, K20_BURN = 10_000, 2_000
FAST_RECORD, FAST_BURN = 9_000, 1_000

#: What ``--fast`` replaces the backbone with, wherever the parameter-subspace system is
#: simulated. A smoke test has to exercise every branch in seconds, and training a
#: 96-by-96 network on a thousand examples is most of the cost of a cell. The outputs then
#: carry the right columns and the wrong numbers, which is why the runner refuses to
#: promote them.
FAST_BACKBONE: Dict[str, Any] = dict(train_examples=384, probe_examples=192,
                                     hidden=(48, 48), backbone_steps=400,
                                     solve_steps=1500)


def shrink(fast: bool) -> Tuple[Tuple[str, Any], ...]:
    """The backbone overrides for this run, in a form a worker argument can carry."""
    return tuple(sorted(FAST_BACKBONE.items())) if fast else ()


#: The seven fields the archived configuration files carried, so that a re-selection still
#: diffs against them column by column. The stored JSON holds every field of the
#: configuration object; these are the ones the ranking tables print.
CONFIG_COLUMNS: Tuple[str, ...] = ("max_E", "tau", "k_neighbors", "theiler", "window",
                                   "stride", "dither")


def _config_columns(cfg) -> Dict[str, Any]:
    values = cfg.as_dict()
    return {name: values[name] for name in CONFIG_COLUMNS}


# ----------------------------------------------------------------- the two grids

def eight_direction_grid(fast: bool = False) -> List[Any]:
    """The 48 configurations of table 7's first column.

    Five axes, four of them with two levels, which is why appendix C concedes that every
    point of this grid sits at a boundary in four of five. The stride is fixed before
    anything is measured: it is a property of how a record is divided, not of how a window
    is scored, and sweeping it would have made the grid a search over both.
    """
    import itertools

    from ..estimator.config import EstimatorConfig

    axes = ((10, 20), (1, 2, 4), (5, 20), ("embedding", "autocorr"), (4000, 8000))
    if fast:
        # Two points, one per level of every axis that has two, so both branches of the
        # Theiler rule and both window lengths are still exercised.
        axes = ((10, 20), (4,), (20,), ("embedding", "autocorr"), (4000, 8000))
    return [EstimatorConfig(max_E=max_e, tau=tau, k_neighbors=neighbours, theiler=theiler,
                            window=window, stride=2000)
            for max_e, tau, neighbours, theiler, window
            in itertools.product(*axes)]


def twenty_direction_grid(fast: bool = False) -> List[Any]:
    """The four configurations of table 7's second column: embedding capacity by lag.

    A deliberately small grid. Nothing about the data-generating process is tuned here,
    and the window length is fixed before the answer is seen.
    """
    import itertools

    from ..estimator.config import EstimatorConfig

    max_es = (20, 40) if not fast else (20, 40)
    taus = (4, 16) if not fast else (16,)
    return [EstimatorConfig(max_E=max_e, tau=tau, k_neighbors=20, theiler="autocorr",
                            window=8000, stride=4000)
            for max_e, tau in itertools.product(max_es, taus)]


# ----------------------------------------------------------------- the systems

def _e8_system(rank: int, record: int, burn: int, extra=()):
    from ..systems.digits_parameter import F_FAST, ten_direction

    return ten_direction(k=rank, mode="qp", drive_amp=0.8, noise_amp=0.0,
                         f0=F_FAST, precondition=True, eta=0.15,
                         window=record, burn=burn, **dict(extra))


def k20_system(arm: str, rank: int, record: int, burn: int, extra=()):
    """One twenty-direction trajectory's configuration, by arm.

    The five arms and the two fixed-rank controls of appendix F, in one place: the
    archived tree spelled the same seven settings out in three scripts and two of them
    disagreed about the backbone.
    """
    from ..systems.digits_parameter import F_FAST, F_SLOW, twenty_direction

    settings: Dict[str, Any] = dict(k=rank, mode="qp", drive_amp=0.8, noise_amp=0.0,
                                    f0=F_FAST, precondition=True, eta=0.15,
                                    window=record, burn=burn)
    if arm == "qp_slow":
        settings["f0"] = F_SLOW
    elif arm == "noise":
        settings.update(mode="noise", drive_amp=0.0, noise_amp=0.08)
    elif arm == "batch_proj":
        settings.update(mode="batch_proj", drive_amp=0.0, noise_amp=3.0, batch=64)
    elif arm == "gd":
        # A transient must be slow enough to fill the window and must not be
        # preconditioned away: with the curvature spread this system has, a rate of 0.006
        # gives time constants of a few thousand steps.
        settings.update(mode="gd", drive_amp=0.0, noise_amp=0.0, eta=0.006,
                        precondition=False, burn=0, window=record + burn,
                        displacement=1.0)
    elif arm == "qp_rotate":
        settings["rotate"] = True
    elif arm not in ("qp", "qp_scale2"):
        raise ValueError(f"unknown twenty-direction arm {arm!r}")
    settings.update(dict(extra))
    return twenty_direction(**settings)


def _k20_schedules(arm: str, length: int):
    """The one control that acts on the observers rather than on the dynamics."""
    from ..systems.digits_parameter import EMPTY, Schedules

    if arm == "qp_scale2":
        return Schedules(observer_gain=np.full(length, 2.0))
    return EMPTY


# ----------------------------------------------------------------- calib.e8

def _e8_cell(args) -> List[Dict[str, Any]]:
    """Simulate one (seed, rank) once, then score every configuration on those same logs.

    A module-level function, not a closure, so it can cross a process boundary.
    """
    seed, rank, record, burn, observer_names, grid_fast, extra = args

    from .. import observers as observer_registry
    from ..estimator import windows
    from ..systems import digits_parameter

    simulation = digits_parameter.simulate(_e8_system(rank, record, burn, extra),
                                           seed=seed)
    truth = simulation.truth.measured["trajectory_effective_rank"]
    update = simulation.truth.measured["update_effective_rank"]
    condition = simulation.truth.measured["drive_condition"]

    rows: List[Dict[str, Any]] = []
    for cfg_id, cfg in enumerate(eight_direction_grid(grid_fast)):
        for name in observer_names:
            series = simulation[name]
            if float(series.std()) <= 1e-12:
                continue
            stats = windows.summarise(series, cfg, seed=seed)
            rows.append({"cfg_id": cfg_id, "seed": seed, "r": rank, "observer": name,
                         "family": observer_registry.get(name).family,
                         "traj_PR": truth, "upd_PR": update, "drive_cond": condition,
                         **_config_columns(cfg), **stats})
    return rows


def _score_configurations(raw, grid, observer_names: Sequence[str], spectral_bins):
    """Per (configuration, observer): does the estimate track the measured rank?

    The rank correlation is taken over all nine (seed, rank) points rather than over the
    three rank medians. Over three points it can only take four values, and a penalty on
    it then swamps the error term the objective is mostly about.
    """
    import pandas as pd

    records = []
    for (cfg_id, observer), group in raw.groupby(["cfg_id", "observer"], sort=True):
        cell = group.groupby("r", sort=True).agg(
            MG=("MG", "median"), truth=("traj_PR", "median"),
            degenerate=("frac_degenerate", "mean")).reset_index()
        record = {
            "cfg_id": int(cfg_id), "observer": observer,
            "rho": spearman(group["MG"], group["traj_PR"]),
            "mae_raw": mae(cell["MG"], cell["truth"]),
            "rho_rough": spearman(group["roughness"], group["traj_PR"]),
            "rho_prd": spearman(group["PRdelay"], group["traj_PR"]),
            "mae_prd": mae(group.groupby("r", sort=True)["PRdelay"].median().values,
                           cell["truth"]),
            "degenerate": float(cell["degenerate"].mean()),
            "sd_across_seeds": float(group.groupby("r", sort=True)["MG"].std().mean()),
            **_config_columns(grid[int(cfg_id)]),
        }
        for bins in spectral_bins:
            column = f"specPR{bins}"
            record[f"rho_spec{bins}"] = spearman(group[column], group["traj_PR"])
            record[f"mae_spec{bins}"] = mae(
                group.groupby("r", sort=True)[column].median().values, cell["truth"])
        records.append(record)
    return pd.DataFrame(records)


def _rank_configurations(scores, grid, observers_used: int):
    """The objective of appendix C: error, plus a penalty on ordering and on seed spread.

    A configuration that is degenerate for most observers must not win on the survivor, so
    a configuration is only ranked if all of the selection observers survived the
    degeneracy filter at it.
    """
    import pandas as pd

    usable = scores[scores["degenerate"] < 0.05]
    ranked = (usable.groupby("cfg_id", sort=True)
              .agg(mae=("mae_raw", "median"), rho=("rho", "median"),
                   sd=("sd_across_seeds", "median"), n=("observer", "count"))
              .reset_index())
    ranked = ranked[ranked["n"] == observers_used].copy()
    if ranked.empty:
        raise ValueError(
            "no configuration survived the degeneracy filter on every selection observer; "
            "the calibration cannot select one and would otherwise pick a survivor")
    ranked["score"] = ranked["mae"] + 2.0 * (1.0 - ranked["rho"]) + 0.5 * ranked["sd"]
    ranked = ranked.sort_values(["score", "cfg_id"]).reset_index(drop=True)
    grid_frame = pd.DataFrame([{"cfg_id": i, **_config_columns(cfg)}
                               for i, cfg in enumerate(grid)])
    return ranked.merge(grid_frame, on="cfg_id")


def _isotonic_maps(raw, cfg_id: int, knots: int = 41):
    """The monotone map per observer, fitted at the frozen configuration.

    Stored as knots so that a later run rebuilds the map rather than refitting it: a
    calibration refitted on whatever data is at hand turns a held-out error into a
    training error.
    """
    from ..estimator.calibration import Calibration

    maps: Dict[str, Dict[str, List[float]]] = {}
    points: Dict[str, Dict[str, List[float]]] = {}
    best = raw[raw["cfg_id"] == cfg_id]
    for observer, group in best.groupby("observer", sort=True):
        usable = group[np.isfinite(group["MG"])]
        if len(usable) < 3:
            continue
        fitted = Calibration("isotonic").fit(usable["MG"].values, usable["traj_PR"].values)
        grid = np.linspace(float(usable["MG"].min()), float(usable["MG"].max()), knots)
        maps[observer] = {"x": grid.tolist(),
                          "y": np.asarray(fitted.predict(grid)).tolist()}
        # The pairs the map was fitted on, kept beside it. Without them a downstream
        # experiment that wants an affine or an uncalibrated comparison has to refit on
        # its own test data, which is the error this file exists to prevent.
        points[observer] = {"estimate": usable["MG"].tolist(),
                            "truth": usable["traj_PR"].tolist()}
    return maps, points


@experiment(
    id="calib.e8",
    title="The eight-direction estimator configuration, selected on withheld data",
    paper=("app:config", "tab:frozen"),
    device=CPU,
    minutes=48,
    promotes=("frozen_config.json", "config_ranking.csv", "calibration_scores.csv"),
    tier=0,
    notes="Requirement 2 forbids reselecting on any later outcome, so re-run this to "
          "check the selection, never to improve a result. Promoting a new "
          "frozen_config.json moves every number in sections 5 to 7.",
)
def eight_direction(ctx: Context) -> None:
    import pandas as pd

    from ..estimator.config import DEFAULT
    from ..observers import K20_CALIBRATION

    seeds, ranks = E8_SEEDS, E8_RANKS
    observer_names = K20_CALIBRATION           # one observer per family, as table 7 says
    record, burn = E8_RECORD, E8_BURN
    if ctx.fast:
        # Two seeds and two ranks, not one of each: the isotonic map needs three finite
        # pairs, and a fast run that wrote an empty one would leave a frozen_config.json
        # in runs/ that shadows the committed file for every later experiment.
        seeds, ranks = seeds[:2], ranks[:2]
        observer_names = observer_names[:2]
        record, burn = FAST_RECORD, FAST_BURN
    grid = eight_direction_grid(ctx.fast)

    ctx.config(seeds=list(seeds), ranks=list(ranks), observers=list(observer_names),
               grid=len(grid), record=record, burn=burn,
               objective="mae + 2 (1 - rho) + 0.5 sd, over the selection observers")
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions", "adapter")

    cells = [(seed, rank, record, burn, tuple(observer_names), ctx.fast, shrink(ctx.fast))
             for seed in seeds for rank in ranks]
    collected = map_ordered(_e8_cell, cells, jobs=ctx.jobs, desc="calib.e8")
    raw = pd.DataFrame([row for cell in collected for row in cell])

    scores = _score_configurations(raw, grid, observer_names, DEFAULT.spectral_bins)
    ctx.store.table("calibration_scores.csv", scores)

    ranking = _rank_configurations(scores, grid, len(observer_names))
    ctx.store.table("config_ranking.csv", ranking)
    # The evidence the selection was made on, kept in runs/ rather than promoted: it is an
    # intermediate, and the two files above are what appendix C prints.
    ctx.store.table("calibration_raw.csv", raw)

    chosen = int(ranking.iloc[0]["cfg_id"])
    maps, points = _isotonic_maps(raw, chosen)
    ctx.store.json("frozen_config.json", {
        "config": grid[chosen].as_dict(),
        "cfg_id": chosen,
        "cal_seeds": list(seeds),
        "cal_r": list(ranks),
        "score": float(ranking.iloc[0]["score"]),
        "isotonic": maps,
        "calibration_points": points,
    })
    ctx.note("frozen", grid[chosen].tag())
    ctx.note("selection_error", [float(ranking["mae"].min()), float(ranking["mae"].max())])


# ----------------------------------------------------------------- calib.e20

def _k20_path(directory: str, arm: str, rank: int, seed: int) -> Path:
    return Path(directory) / "trajectories" / f"{arm}_r{rank:02d}_s{seed:02d}.npz"


def _k20_simulate(args) -> Dict[str, Any]:
    """One twenty-direction trajectory, written where the two scoring passes can find it.

    Written rather than held, because selection has to finish before the frozen
    configuration is known, and re-simulating between the two passes would mean the
    selection and the test ran on two different systems.
    """
    arm, rank, seed, record, burn, directory, panel, extra = args

    from ..systems import digits_parameter

    config = k20_system(arm, rank, record, burn, extra)
    schedules = _k20_schedules(arm, config.length)
    simulation = digits_parameter.simulate(config, seed=seed, schedules=schedules)

    path = _k20_path(directory, arm, rank, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    measured = simulation.truth.measured
    info = {"arm": arm, "mode": config.mode, "r": rank, "seed": seed,
            "available": config.available,
            "traj_rank": measured["trajectory_rank"],
            "traj_pr": measured["trajectory_effective_rank"],
            "update_pr": measured["update_effective_rank"],
            "functional_rank": measured["functional_rank"],
            "functional_pr": measured["functional_effective_rank"],
            "resonance_margin": measured["resonance_margin"],
            "drive_cond": measured["drive_condition"]}
    np.savez_compressed(path, info=json.dumps(info),
                        **{f"log__{name}": simulation[name] for name in panel})
    return info


def k20_load(path: Path, panel: Sequence[str]):
    """One stored twenty-direction trajectory, as ``(logs, what was measured on it)``."""
    with np.load(path, allow_pickle=False) as stored:
        info = json.loads(str(stored["info"]))
        logs = {name: np.asarray(stored[f"log__{name}"], dtype=float) for name in panel}
    return logs, info


def _k20_score(args) -> List[Dict[str, Any]]:
    """Score one stored trajectory, at one or more configurations, on one panel."""
    path, panel, configs = args

    from ..estimator.config import EstimatorConfig
    from ..estimator import windows

    logs, info = k20_load(Path(path), panel)
    rows: List[Dict[str, Any]] = []
    for cfg_id, values in configs:
        cfg = EstimatorConfig.from_dict(values)
        for name in panel:
            series = logs[name]
            if len(series) < cfg.window or float(series.std()) <= 1e-12:
                continue
            stats = windows.summarise(series, cfg, seed=int(info["seed"]))
            rows.append({"file": Path(path).name, "cfg_id": cfg_id, "tag": info["arm"],
                         "mode": info["mode"], "r": info["r"], "seed": info["seed"],
                         "observer": name, **{k: info[k] for k in
                                              ("available", "functional_rank",
                                               "functional_pr", "traj_rank", "traj_pr",
                                               "update_pr")},
                         **stats})
    return rows


def _k20_jobs(fast: bool) -> List[Tuple[str, int, int]]:
    """Every twenty-direction trajectory, as (arm, rank, seed), in a stable order."""
    if fast:
        return ([("qp", r, 0) for r in (2, 3, 6, 10)]
                + [("noise", 2, 0), ("gd", 2, 0), ("qp_scale2", 2, 0),
                   ("qp_rotate", 2, 0)])
    jobs = [("qp", r, s) for r in K20_RANKS for s in K20_SEEDS]
    jobs += [(arm, r, 0) for arm in K20_ARMS for r in K20_ANCHORS]
    jobs += [("qp_scale2", 10, 0), ("qp_rotate", 10, 0)]
    return jobs


def _k20_select(calibration, grid, selection_ranks):
    """Stage one: rank (configuration, observer) pairs and freeze the winning configuration.

    The criterion is the raw numerical agreement of the median estimate with the measured
    rank. An isotonic fit is deliberately not used: with five calibration ranks it
    interpolates any monotone curve exactly, every error comes out zero, and the four
    configurations tie. The archived file recorded that tie for one release after the code
    that produced it had been fixed.
    """
    import pandas as pd

    records = []
    for (cfg_id, observer), group in calibration.groupby(["cfg_id", "observer"],
                                                         sort=True):
        estimate = group.groupby("r", sort=True)["MG"].median()
        truth = group.groupby("r", sort=True)["traj_pr"].median()
        if len(estimate) < 3 or estimate.isna().any():
            continue
        records.append({
            "cfg_id": int(cfg_id), "observer": observer,
            "cal_mae": mae(estimate.values, truth.values),
            "cal_rho": spearman(estimate.values, truth.values),
            "cal_degenerate": float(group["frac_degenerate"].mean()),
            **_config_columns(grid[int(cfg_id)]),
        })
    ranking = pd.DataFrame(records)
    if ranking.empty:
        raise ValueError("no configuration was scored on at least three selection ranks")
    ranking["score"] = (ranking["cal_mae"] + 0.25 * (1.0 - ranking["cal_rho"].fillna(0.0))
                        + 2.0 * ranking["cal_degenerate"])
    ranking = ranking.sort_values(["score", "cfg_id"]).reset_index(drop=True)
    best = int(ranking.groupby("cfg_id")["score"].median().idxmin())
    return ranking, best


def _k20_summary(scored, selection_ranks, heldout_ranks):
    """The held-out recovery table, with the affine calibration fitted on the selection ranks.

    Both the raw and the calibrated numbers are reported. A monotone map cannot change the
    rank correlation, so the correlation is the same on either; the error is not, and
    quoting one without the other hides which of the two is being claimed.
    """
    import pandas as pd

    from ..estimator.calibration import Calibration

    recurrent = scored[scored["tag"] == "qp"]
    selection = recurrent[recurrent["r"].isin(selection_ranks)]
    heldout = recurrent[recurrent["r"].isin(heldout_ranks)].copy()
    if heldout.empty:
        raise ValueError("no held-out ranks were scored")

    calibrated = np.full(len(heldout), np.nan)
    for observer, group in selection.groupby("observer", sort=True):
        usable = group[np.isfinite(group["MG"]) & np.isfinite(group["traj_pr"])]
        if len(usable) < 3:
            continue
        fitted = Calibration("affine").fit(usable["MG"].values, usable["traj_pr"].values)
        rows = (heldout["observer"] == observer).to_numpy()
        calibrated[rows] = fitted.predict(heldout.loc[rows, "MG"].to_numpy())
    heldout["MG_cal"] = calibrated

    records = []
    for observer, group in heldout.groupby("observer", sort=True):
        error = np.abs(group["MG_cal"] - group["traj_pr"])
        records.append({
            "observer": observer,
            "rho_raw": spearman(group["traj_pr"], group["MG"]),
            "mae_raw": mae(group["MG"], group["traj_pr"]),
            "rho_cal": spearman(group["traj_pr"], group["MG_cal"]),
            "mae_cal": mae(group["MG_cal"], group["traj_pr"]),
            "max_error_cal": float(np.nanmax(error)) if np.isfinite(error).any()
                             else float("nan"),
            "degenerate": float(group["frac_degenerate"].mean()),
        })
    return pd.DataFrame(records).sort_values("mae_cal").reset_index(drop=True)


def _k20_controls(scored, rank: int, panel: Sequence[str]):
    """The two fixed-rank invariance controls, each paired with its own baseline run.

    Paired with the single (rank, seed) they were run at rather than with the three-seed
    median, because comparing a one-seed control against a multi-seed median confounds
    invariance with seed-to-seed variability.
    """
    import pandas as pd

    base = scored[(scored["tag"] == "qp") & (scored["r"] == rank) & (scored["seed"] == 0)]
    records = []
    for control in ("qp_scale2", "qp_rotate"):
        other = scored[(scored["tag"] == control) & (scored["r"] == rank)
                       & (scored["seed"] == 0)]
        for observer in panel:
            left = base[base["observer"] == observer]
            right = other[other["observer"] == observer]
            if len(left) and len(right):
                records.append({"control": control, "observer": observer,
                                "delta_MG": float(right["MG"].iloc[0]
                                                  - left["MG"].iloc[0])})
    return pd.DataFrame(records)


@experiment(
    id="calib.e20",
    title="The twenty-direction estimator configuration, and what it recovers",
    paper=("app:config", "tab:frozen", "tab:k20"),
    device=CPU,
    minutes=70,
    promotes=("frozen_k20.json", "scores_frozen.csv", "frozen_per_r.csv",
              "heldout_qp_summary.csv", "invariance_controls.csv"),
    tier=0,
    notes="One scorer. The archived tree had two writing these three files by different "
          "aggregation code with no record of which produced the committed copy "
          "(errata 17). The stored trajectories are read again by valid.theiler.cap.",
)
def twenty_direction(ctx: Context) -> None:
    import pandas as pd

    from ..observers import K20_CALIBRATION, K20_PANEL

    panel = K20_PANEL
    calibration_observers = K20_CALIBRATION
    selection_ranks, heldout_ranks = K20_SELECTION, K20_HELDOUT
    record, burn = K20_RECORD, K20_BURN
    if ctx.fast:
        panel = K20_PANEL[:3]
        calibration_observers = tuple(n for n in K20_CALIBRATION if n in panel)
        selection_ranks, heldout_ranks = (2, 6, 10), (3,)
        record, burn = FAST_RECORD, FAST_BURN
    grid = twenty_direction_grid(ctx.fast)
    directory = str(ctx.store.dir)

    ctx.config(seeds=list(K20_SEEDS), selection_ranks=list(selection_ranks),
               heldout_ranks=list(heldout_ranks), observers=list(panel),
               selection_observers=list(calibration_observers),
               grid=len(grid), record=record, burn=burn)
    ctx.declare_seeds("drive_phases", "drive_groups", "observer_directions", "adapter",
                      "rotation")

    jobs = _k20_jobs(ctx.fast)
    simulated = map_ordered(
        _k20_simulate,
        [(arm, rank, seed, record, burn, directory, tuple(panel), shrink(ctx.fast))
         for arm, rank, seed in jobs],
        jobs=ctx.jobs, desc="calib.e20 simulate")
    paths = [str(_k20_path(directory, arm, rank, seed)) for arm, rank, seed in jobs]
    ctx.note("trajectories", len(paths))

    # Stage one: selection, on the calibration ranks of the recurrent arm only. The
    # held-out ranks and every robustness arm are untouched until the configuration is
    # frozen.
    selection_paths = [path for path, (arm, rank, _) in zip(paths, jobs)
                       if arm == "qp" and rank in selection_ranks]
    grid_values = [(i, cfg.as_dict()) for i, cfg in enumerate(grid)]
    selection_rows = map_ordered(
        _k20_score, [(path, tuple(calibration_observers), grid_values)
                     for path in selection_paths],
        jobs=ctx.jobs, desc="calib.e20 select")
    calibration = pd.DataFrame([row for rows in selection_rows for row in rows])
    ranking, chosen = _k20_select(calibration, grid, selection_ranks)
    ctx.store.table("config_observer_ranking.csv", ranking)
    ctx.store.table("calibration_configs.csv", calibration)

    # Stage two: the one frozen configuration on every trajectory and every observer.
    frozen = grid[chosen]
    scored_rows = map_ordered(
        _k20_score, [(path, tuple(panel), [(chosen, frozen.as_dict())]) for path in paths],
        jobs=ctx.jobs, desc="calib.e20 score")
    scored = pd.DataFrame([row for rows in scored_rows for row in rows])
    ctx.store.table("scores_frozen.csv", scored)

    per_rank = (scored.groupby(["tag", "mode", "observer", "r"], sort=True)
                .agg(MG=("MG", "median"), LB=("LB", "median"), TwoNN=("TwoNN", "median"),
                     PRdelay=("PRdelay", "median"), traj_pr=("traj_pr", "median"),
                     update_pr=("update_pr", "median"),
                     degenerate=("frac_degenerate", "mean")).reset_index())
    ctx.store.table("frozen_per_r.csv", per_rank)

    ctx.store.table("heldout_qp_summary.csv",
                    _k20_summary(scored, selection_ranks, heldout_ranks))
    control_rank = 2 if ctx.fast else 10
    ctx.store.table("invariance_controls.csv",
                    _k20_controls(scored, control_rank, panel))

    ctx.store.json("frozen_k20.json", {
        "cfg_id": chosen,
        "config": frozen.as_dict(),
        "calibration_r": list(selection_ranks),
        "calibration_seeds": list(K20_SEEDS),
        "selection": "minimum median raw absolute error across non-degenerate "
                     "calibration observers",
        "observers": list(panel),
    })
    ctx.note("frozen", frozen.tag())
    ctx.note("simulated", len(simulated))
    ctx.note("record", record)

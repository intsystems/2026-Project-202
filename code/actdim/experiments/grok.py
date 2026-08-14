"""Section 7: the application to delayed generalisation.

Nine experiments, in four groups, and the order below is the order the argument is made in.

*The diagnostics* (``grok.diagnostics.*``) place every training log of both settings on the
two admissibility axes. Neither setting is admissible, and they fail differently: the
regularised mini-batch runs occupy a band of the identifiability ratio no deterministic
regime reaches, the full-batch runs carry a transient's signature. Nothing downstream may
read a level as a dimension, and everything downstream reads a change instead.

*The direct measurement* (``grok.rank.dip``) does not go through a log at all. It reads the
stored trajectory, whose effective rank is defined whether or not the orbit recurs, and
locates the collapse at generalisation run by run.

*The matched window* (``grok.matched.*``) returns to the log with the one question a
rejected level still permits, at a window set in units of the transition rather than of
the record, on the direct measurement's own grid. Two things about it are protocol and not
taste. Requirement 2 forbids choosing a configuration on the outcome, so every cell of the
grid is reported and the headline is named by a rule that cannot see the answer; and the
fall is put against surrogates that keep the log's shape and destroy only its fine
structure, because a detrended scalar has had most of itself removed and proves nothing.

*The limits* (``grok.extended.outcomes``, ``grok.prwindow``, ``grok.eos``, ``grok.repr``)
are the four results that bound the claim: what a six-times-longer budget does to the
controls, what the full-batch statistic does when the window is lengthened, whether an
undriven run ever recurs, and what the representation looks like when it is written down.

Every input is resolved through ``ctx.input`` and every run set is declared rather than
discovered. The archived tree globbed its inputs, and a directory that had gained three
runs since the table was written produced a different table from the same command.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..runtime import CPU, Context, experiment
from ..runtime.parallel import map_ordered

# =============================================================================
# 7.1 -- the two admissibility diagnostics on a training log
# =============================================================================

#: Every column of the per-window diagnostic table, in the order the archived atlas wrote
#: them. The two experiments below emit different subsets of it and one driver fills it.
WINDOW_COLUMNS: Tuple[str, ...] = (
    "run", "column", "right_step", "log_stride", "MG", "MG_2E", "ident_ratio", "LB",
    "TwoNN", "PRdelay", "roughness", "acorr", "oscillations", "degenerate")

#: What ``fig_map`` reads: one row per run and observer, medians over the run's windows.
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "run", "column", "MG", "MG_2E", "ident", "PRdelay", "roughness", "acorr", "osc",
    "degen", "n")


def _log_diagnostics(args) -> List[Dict[str, Any]]:
    """One (run, observer): the estimate and its diagnostics over the log's windows.

    A module-level function, not a closure, so it can cross a process boundary. The
    identifiability ratio is computed on the *same* window at twice the embedding
    dimension, so it compares two embeddings and not two stretches of record.
    """
    run, column, path = args

    from ..analysis import logs
    from ..estimator import windows
    from ..estimator.diagnostics import ratio, trend_crossings
    from ..estimator.mle import estimate

    frame = logs.load_log(path)
    if column not in frame.columns:
        return []
    x = frame[column].to_numpy(dtype=float)
    step = frame["step"].to_numpy()
    stride = logs.log_stride(frame)

    cfg = logs.article_geometry(len(x))
    doubled = cfg.replace(max_E=2 * cfg.max_E)

    rows: List[Dict[str, Any]] = []
    for start in windows.window_starts(len(x), cfg):
        window = x[start:start + cfg.window]
        if not np.isfinite(window).all() or window.std() <= 1e-12:
            continue
        scored = windows.score(window, cfg)
        at_2e = estimate(window, doubled).MG
        rows.append({
            "run": run, "column": column,
            "right_step": int(step[start + cfg.window - 1]), "log_stride": stride,
            "MG": scored["MG"], "MG_2E": at_2e,
            "ident_ratio": ratio(scored["MG"], at_2e),
            "LB": scored["LB"], "TwoNN": scored["TwoNN"], "PRdelay": scored["PRdelay"],
            "roughness": scored["roughness"], "acorr": scored["acorr"],
            "oscillations": trend_crossings(window),
            "degenerate": bool(scored["degenerate"]),
        })
    return rows


def _diagnostic_frames(ctx: Context, upstream: str, runs: Sequence[str],
                       columns: Sequence[str], desc: str):
    """Run the diagnostic driver over a declared set of runs and observers."""
    import pandas as pd

    from ..analysis import logs

    paths = {run: str(logs.find_log(ctx, upstream, run)) for run in runs}
    jobs = [(run, column, paths[run]) for run in runs for column in columns]
    results = map_ordered(_log_diagnostics, jobs, jobs=ctx.jobs, desc=desc)
    rows = [row for result in results for row in result]
    frame = pd.DataFrame(rows, columns=list(WINDOW_COLUMNS))

    if len(frame):
        summary = (frame.groupby(["run", "column"], sort=True)
                   .agg(MG=("MG", "median"), MG_2E=("MG_2E", "median"),
                        ident=("ident_ratio", "median"), PRdelay=("PRdelay", "median"),
                        roughness=("roughness", "median"), acorr=("acorr", "median"),
                        osc=("oscillations", "median"), degen=("degenerate", "mean"),
                        n=("MG", "size")).reset_index())
    else:
        summary = pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    return frame, summary[list(SUMMARY_COLUMNS)]


@experiment(
    id="grok.diagnostics.logs",
    title="Where the transformer logs sit on the two admissibility axes",
    paper=("sec:grok-diagnostics", "app:pairs", "fig:map"),
    device=CPU,
    minutes=5,
    needs=("train.transformer.extended",),
    promotes=("real_logs_windows.csv", "real_logs_summary.csv"),
    tier=3,
    notes="The seven 120,000-step reruns, five observers each, at the frozen "
          "configuration with only the window geometry shortened to fit the record.",
)
def diagnostics_logs(ctx: Context) -> None:
    from ..analysis import logs

    runs: Sequence[str] = logs.TRANSFORMER_EXTENDED
    columns: Sequence[str] = logs.TRANSFORMER_LOG_COLUMNS
    if ctx.fast:
        runs, columns = runs[:1], columns[:2]

    ctx.config(runs=list(runs), observers=list(columns),
               geometry="training log: window a third of the record, stride 1000")
    frame, summary = _diagnostic_frames(ctx, "train.transformer.extended", runs, columns,
                                        "real logs")
    ctx.store.table("real_logs_windows.csv", frame)
    ctx.store.table("real_logs_summary.csv", summary)
    ctx.note("n_windows", int(len(frame)))


@experiment(
    id="grok.diagnostics.perceptron",
    title="The same two diagnostics on the full-batch perceptron logs",
    paper=("sec:grok-diagnostics", "sec:pairs", "app:pairs", "fig:map", "fig:pairs"),
    device=CPU,
    minutes=1,
    needs=("train.perceptron.arith", "train.perceptron.poly"),
    promotes=("dimension_probe.csv", "dimension_probe_summary.csv",
              "dimension_probe_poly.csv", "dimension_probe_summary_poly.csv"),
    tier=3,
    notes="Four arithmetic runs and six polynomial ones, two observers each. The run set "
          "and the observers are declared here; the archived command globbed both.",
)
def diagnostics_perceptron(ctx: Context) -> None:
    from ..analysis import logs

    # Declared, not globbed, and two observers rather than three. The archived script took
    # every `*_train.csv` beside it -- seven runs in the arithmetic directory -- and
    # defaulted to three columns, so re-running it turned fig_map's "perceptron, full
    # batch (10)" into 13 and added the observer that defines generalisation. The
    # committed table is right for the article and its own command was not; docs/errata.md
    # item 8.
    columns: Sequence[str] = logs.PERCEPTRON_PROBE_COLUMNS
    arms = [
        ("train.perceptron.arith", logs.PERCEPTRON_ARITH,
         ("dimension_probe.csv", "dimension_probe_summary.csv"), "arith"),
        ("train.perceptron.poly", logs.PERCEPTRON_POLY,
         ("dimension_probe_poly.csv", "dimension_probe_summary_poly.csv"), "poly"),
    ]
    ctx.config(observers=list(columns),
               arith_runs=list(logs.PERCEPTRON_ARITH), poly_runs=list(logs.PERCEPTRON_POLY))

    # The probe table carries no LB, TwoNN or logging stride: those columns are in the
    # transformer atlas and not in this one, and the figures read the shared subset.
    keep = [c for c in WINDOW_COLUMNS if c not in ("log_stride", "LB", "TwoNN")]

    for upstream, runs, names, desc in arms:
        if ctx.fast:
            runs = runs[:2]
        frame, summary = _diagnostic_frames(ctx, upstream, runs, columns, desc)
        ctx.store.table(names[0], frame[keep])
        ctx.store.table(names[1], summary)


# =============================================================================
# 7.2 -- the trajectory itself
# =============================================================================


def _rank_pass(args) -> Dict[str, Any]:
    """Every window statistic of one run's sketch, at one window geometry."""
    run, log_path, sketch_path, geometry_name = args

    from ..analysis import logs
    from ..sketch import analysis as sketch

    geometry = {"coarse": sketch.COARSE, "fine": sketch.FINE}[geometry_name]
    log = logs.load_log(log_path)
    # allow_pickle stays off: a sketch is arrays and metadata, and unpickling an array
    # file is a code path that has no business existing in an analysis.
    with np.load(sketch_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    frame = sketch.sliding(arrays, geometry, run=run)
    summary = sketch.summarise(run, log, arrays, frame, geometry)
    return {"windows": frame, "summary": summary}


def _rename_offset(frame):
    """The collapse tables call the dip's position ``offset``; the published ones say ``at``.

    The library name is the better one and the file keeps the published one, because the
    committed table, the figures and the table audit all read ``at`` and a silent rename
    would be a column that moved without a diff to show it.
    """
    return frame.rename(columns={"offset": "at"})


@experiment(
    id="grok.rank.dip",
    title="The effective-rank collapse of the stored trajectory, run by run",
    paper=("sec:direct", "sec:nowd", "app:dip", "fig:dip"),
    device=CPU,
    minutes=9,
    needs=("train.transformer.sketched",),
    promotes=("rank_windows.csv", "rank_summary.csv", "rank_milestones.json",
              "rank_dip.csv", "rank_dip_controls.csv", "rank_dip_controls_aligned.csv",
              "mod_wd1_train.csv"),
    tier=3,
    notes="Reads the six trajectory sketches. Both window geometries are run; the fine "
          "one is what the article reports and the coarse one is stamped with its name, "
          "because in the archived tree the second pass overwrote the first's output.",
)
def rank_dip(ctx: Context) -> None:
    import pandas as pd

    from ..analysis import logs
    from ..sketch import analysis as sketch

    runs: Sequence[str] = logs.TRANSFORMER_SKETCHED
    if ctx.fast:
        # One generalising run and its control: the smallest set on which the collapse
        # table, the control table and the aligned control table all have a row.
        runs = ("mod_wd0", "mod_wd1")

    upstream = "train.transformer.sketched"
    inputs = {run: (str(logs.find_log(ctx, upstream, run)),
                    str(logs.require_sketch(ctx, upstream, run))) for run in runs}

    ctx.config(runs=list(runs), geometries=["fine", "coarse"],
               window=sketch.ARTICLE.window, stride=sketch.ARTICLE.stride,
               smoothing_dropped=list(sketch.ARTICLE.dropped_smoothing()))

    for geometry in (sketch.ARTICLE, sketch.COARSE):
        jobs = [(run, inputs[run][0], inputs[run][1], geometry.name) for run in runs]
        results = map_ordered(_rank_pass, jobs, jobs=ctx.jobs, desc=f"rank {geometry.name}")

        frames = [result["windows"] for result in results]
        summaries = [result["summary"] for result in results]
        windows = pd.concat(frames, ignore_index=True)
        by_run = {record["run"]: (record["t_mem"], record["t_gen"])
                  for record in summaries}
        # Carried on every window row as well as in the summary: the archived table had
        # them, the aligned control table is computed from them, and a window table that
        # cannot say which side of the transition a row is on is not readable alone.
        windows["t_mem"] = windows["run"].map(lambda r: by_run[r][0])
        windows["t_gen"] = windows["run"].map(lambda r: by_run[r][1])

        collapse = _rename_offset(sketch.collapse(windows, by_run))
        controls = _rename_offset(sketch.collapse_controls(windows, by_run))
        aligned = sketch.collapse_controls_aligned(windows, by_run)

        # The article reports the fine pass, which keeps the published names; the coarse
        # pass is stamped with its geometry. Neither can overwrite the other, which is the
        # failure the archived collapse script shipped: it took its results directory from
        # the command line and pinned its figure directory beside the code, so the second
        # pass overwrote the first's figure and the coarse one has been a copy of the fine
        # one ever since.
        article = geometry is sketch.ARTICLE
        name = (lambda published: published) if article else geometry.stamp
        ctx.store.table(name("rank_windows.csv"), windows)
        ctx.store.table(name("rank_summary.csv"), pd.DataFrame(summaries))
        ctx.store.json(name("rank_milestones.json"), summaries)
        ctx.store.table(name("rank_dip.csv"), collapse)
        ctx.store.table(name("rank_dip_controls.csv"), controls)
        ctx.store.table(name("rank_dip_controls_aligned.csv"), aligned)

        if article:
            ctx.note("milestones", {run: list(value) for run, value in by_run.items()})
            ctx.note("n_windows", int(len(windows)))

    # fig_traces draws this log and section 7.2 is computed from it. Two `mod_wd1` logs
    # exist and they are not the same series -- they agree for 198 rows and then diverge
    # under float64 rounding, ending 110 steps apart on generalisation (docs/errata.md
    # item 23) -- so the one the windows were cut from travels with them rather than
    # being looked up again later from whichever directory happens to hold one.
    if "mod_wd1" in inputs:
        import shutil

        copy = ctx.store.path("mod_wd1_train.csv")
        shutil.copy2(inputs["mod_wd1"][0], copy)
        ctx.store.adopt(copy)


# =============================================================================
# 7.3 -- the log estimate at a window matched to the transition
# =============================================================================

#: Every axis the estimate depends on, crossed: thirty-six cells. The frozen configuration
#: cannot be used at all here -- its delay span, (20-1)*4 = 76 samples, exceeds the
#: sixty-sample window -- so a configuration has to be chosen, and requirement 2 forbids
#: choosing one on the outcome. Every cell is reported.
MATCHED_GRID: Tuple[Dict[str, Any], ...] = tuple(
    {"max_E": max_e, "tau": tau, "k_neighbors": k, "theiler": theiler}
    for max_e in (4, 6, 10) for tau in (1, 2, 4) for k in (5, 20)
    for theiler in ("autocorr", "embedding"))

def _matched_headline() -> Dict[str, Any]:
    """The headline cell, named by a rule applied without reference to any output.

    The delay span ``(max_E - 1) * tau`` must be at most a quarter of the window, and
    subject to that ``max_E`` is as large as it goes, at the frozen configuration's own
    neighbour count and Theiler rule. It is not the best cell, and the article reports how
    many of the others agree with it.

    The two settings the rule takes from the frozen configuration are read from it rather
    than written out here. Recalibrating moved the frozen Theiler rule from ``autocorr`` to
    ``embedding`` -- an arbitrary move, the two returning bit-identical values on the
    calibration logs -- and a copy of the old name here would have left the headline cell
    claiming to follow a rule it no longer followed.
    """
    from .. import frozen as frozen_mod
    from ..analysis import logs

    base = frozen_mod.eight_direction()
    quarter = logs.MATCHED_WINDOW / 4.0
    eligible = [c for c in MATCHED_GRID
                if (c["max_E"] - 1) * c["tau"] <= quarter
                and c["k_neighbors"] == base.k_neighbors
                and c["theiler"] == base.theiler]
    if not eligible:
        raise ValueError("no cell of the matched grid satisfies the headline rule")
    return max(eligible, key=lambda c: (c["max_E"], -c["tau"]))


MATCHED_HEADLINE: Dict[str, Any] = _matched_headline()

MATCHED_DETREND: Tuple[bool, ...] = (False, True)

#: The observers the grid is run on. The last two define the event being aligned on, and
#: they are here so that their circularity is visible rather than hidden.
MATCHED_COLUMNS: Tuple[str, ...] = ("weight_norm", "train_loss", "val_loss",
                                    "train_acc", "val_acc")

#: The observers the grid tables are written for, and the two the headline trace keeps.
GRID_COLUMNS: Tuple[str, ...] = ("weight_norm", "train_loss", "val_loss")
HEADLINE_COLUMNS: Tuple[str, ...] = ("weight_norm", "train_loss")

MATCHED_RAW_COLUMNS: Tuple[str, ...] = (
    "run", "column", "mid_step", "max_E", "tau", "k", "theiler", "detrend", "MG", "MG_2E",
    "ident_ratio", "PRdelay", "roughness", "oscillations", "degenerate")


def _matched_cell(args) -> List[Dict[str, Any]]:
    """One (run, observer, configuration, detrending) over the direct measurement's grid."""
    run, column, path, centres, cell, detrend, base = args

    from ..analysis import logs
    from ..estimator import windows
    from ..estimator.diagnostics import ratio, trend_crossings
    from ..estimator.mle import estimate

    frame = logs.load_log(path)
    if column not in frame.columns:
        return []
    x = frame[column].to_numpy(dtype=float)
    stride = logs.log_stride(frame)

    cfg = base.replace(window=logs.MATCHED_WINDOW, stride=1, **cell)
    doubled = cfg.replace(max_E=2 * cfg.max_E)

    rows: List[Dict[str, Any]] = []
    for centre, values in logs.matched_windows(x, stride, centres, detrend=detrend):
        scored = windows.score(values, cfg)
        at_2e = estimate(values, doubled).MG
        rows.append({
            "run": run, "column": column, "mid_step": centre,
            "max_E": cfg.max_E, "tau": cfg.tau, "k": cfg.k_neighbors,
            "theiler": cfg.theiler, "detrend": bool(detrend),
            "MG": scored["MG"], "MG_2E": at_2e, "ident_ratio": ratio(scored["MG"], at_2e),
            "PRdelay": scored["PRdelay"], "roughness": scored["roughness"],
            "oscillations": trend_crossings(values),
            "degenerate": bool(scored["degenerate"]),
        })
    return rows


def _grid_row(group, run: str, t_gen: float, t_mem: Optional[int]) -> Dict[str, Any]:
    """Depth at the transition, its quantile over the run, and where the floor is."""
    from ..analysis import logs

    out = {f"dep_{run}": float("nan"), f"pct_{run}": float("nan"),
           f"loc_{run}": float("nan")}
    if not len(group) or not np.isfinite(t_gen):
        return out
    ordered = group.sort_values("mid_step")
    t = ordered["mid_step"].to_numpy(dtype=float)
    y = ordered["MG"].to_numpy(dtype=float)

    observed = logs.depth(t, y, t_gen)
    out[f"dep_{run}"] = observed
    out[f"loc_{run}"] = logs.floor_offset(t, y, t_gen)

    # Every admissible centre after memorisation, on the run's own grid: a fall that is
    # deep but no deeper than the rest of the run is not a signature, and this is what
    # says so.
    floor = 0.0 if t_mem is None else float(t_mem) + 1000.0
    profile = logs.depth_profile(t, y, t[t >= floor])
    defined = np.isfinite(profile)
    if np.isfinite(observed) and defined.sum() > 5:
        out[f"pct_{run}"] = float((profile[defined] <= observed).mean())
    return out


def _grid_table(raw, column: str, runs: Sequence[str],
                milestones_by_run: Dict[str, Tuple[Any, Any]]):
    """One observer's grid: every cell, every run, and whether the cell separates them."""
    import pandas as pd

    from ..analysis import logs

    keys = ["max_E", "tau", "k", "theiler", "detrend"]
    rows: List[Dict[str, Any]] = []
    subset = raw[raw["column"] == column]
    for values, group in subset.groupby(keys, sort=True):
        record: Dict[str, Any] = dict(zip(["max_E", "tau", "k", "theiler", "det"], values))
        generalising, control = [], []
        for run in runs:
            t_mem, _ = milestones_by_run.get(run, (None, None))
            cell = _grid_row(group[group["run"] == run], run,
                             logs.transition_of(run, milestones_by_run), t_mem)
            record.update(cell)
            target = control if run in logs.CONTROL_OF else generalising
            target.append(cell[f"dep_{run}"])
        record["gen_min"] = _nanmin(generalising)
        record["ctrl_max"] = _nanmax(control)
        record["separates"] = bool(np.isfinite(record["gen_min"])
                                   and np.isfinite(record["ctrl_max"])
                                   and record["gen_min"] > record["ctrl_max"])
        rows.append(record)

    columns = ["max_E", "tau", "k", "theiler", "det"]
    for run in runs:
        columns += [f"dep_{run}", f"pct_{run}", f"loc_{run}"]
    columns += ["gen_min", "ctrl_max", "separates"]
    return pd.DataFrame(rows, columns=columns)


def _nanmin(values: Iterable[float]) -> float:
    """``nanmin`` that returns NaN on an all-NaN input instead of warning."""
    finite = [v for v in values if np.isfinite(v)]
    return float(min(finite)) if finite else float("nan")


def _nanmax(values: Iterable[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite)) if finite else float("nan")


def _matched_inputs(ctx: Context, runs: Sequence[str]):
    """The six logs, the direct measurement's window grid, and its milestones."""
    import pandas as pd

    from ..analysis import logs

    upstream = "train.transformer.sketched"
    paths = {run: str(logs.find_log(ctx, upstream, run)) for run in runs}
    windows = pd.read_csv(ctx.input("grok.rank.dip", "rank_windows.csv"))
    milestones = logs.milestone_map(ctx.input("grok.rank.dip", "rank_milestones.json"))
    centres = {run: logs.matched_centres(windows, run) for run in runs}
    return paths, windows, milestones, centres


@experiment(
    id="grok.matched.window",
    title="The log estimate at a window matched to the transition, over the whole grid",
    paper=("sec:matched", "app:window", "tab:matched", "fig:dip"),
    device=CPU,
    minutes=4,
    needs=("train.transformer.sketched", "grok.rank.dip"),
    promotes=("headline_trace.csv", "grid_weight_norm.csv", "grid_train_loss.csv",
              "grid_val_loss.csv"),
    tier=3,
    notes="Thirty-six configurations by two detrendings by six runs by five observers, "
          "on the midpoints the direct measurement used. The headline cell is named by a "
          "rule blind to the outcome and is not the best cell.",
)
def matched_window(ctx: Context) -> None:
    import pandas as pd

    from .. import frozen
    from ..analysis import logs

    runs: Sequence[str] = logs.TRANSFORMER_SKETCHED
    columns: Sequence[str] = MATCHED_COLUMNS
    grid: Sequence[Dict[str, Any]] = MATCHED_GRID
    if ctx.fast:
        runs = ("mod_wd0", "mod_wd1")
        # The three observers the grid tables are written for, so every declared file
        # still has its columns, and two cells: the headline, and one that fails the
        # length gate so the NaN path is exercised too.
        columns = GRID_COLUMNS
        grid = (MATCHED_HEADLINE, {"max_E": 10, "tau": 4, "k_neighbors": 5,
                                   "theiler": "embedding"})

    paths, direct_windows, milestones, centres = _matched_inputs(ctx, runs)
    base = frozen.eight_direction()
    ctx.config(window_samples=logs.MATCHED_WINDOW, cells=len(grid),
               detrend=list(MATCHED_DETREND), runs=list(runs), observers=list(columns),
               headline=dict(MATCHED_HEADLINE))

    jobs = [(run, column, paths[run], centres[run], cell, detrend, base)
            for run in runs for column in columns
            for cell in grid for detrend in MATCHED_DETREND]
    results = map_ordered(_matched_cell, jobs, jobs=ctx.jobs, desc="matched window")
    raw = pd.DataFrame([row for result in results for row in result],
                       columns=list(MATCHED_RAW_COLUMNS))

    # The direct measurement's own statistics at the same midpoints, joined on rather than
    # carried through every worker: the two are two statistics of one run at one instant,
    # and fig_dip's panel (d) is the join.
    direct = _direct_columns(direct_windows, runs, milestones)
    raw = raw.merge(direct, on=["run", "mid_step"], how="left")

    # Written at full precision. The archived script rounded the bulk table to five
    # decimals to keep its gzip small and then took the headline trace from the rounded
    # copy, which moved every published value by up to 5e-6; this file is untracked, so
    # there is nothing to trade.
    path = ctx.store.path("matched_windows.csv.gz")
    raw.to_csv(path, index=False, compression="gzip")
    ctx.store.adopt(path)

    headline = raw[(raw["max_E"] == MATCHED_HEADLINE["max_E"])
                   & (raw["tau"] == MATCHED_HEADLINE["tau"])
                   & (raw["k"] == MATCHED_HEADLINE["k_neighbors"])
                   & (raw["theiler"] == MATCHED_HEADLINE["theiler"])
                   & (~raw["detrend"])
                   & (raw["column"].isin(HEADLINE_COLUMNS))]
    ctx.store.table("headline_trace.csv",
                    headline[["run", "column", "mid_step", "MG", "PRdelay", "roughness",
                              "PR_det", "t_gen"]])

    separating = {}
    for column in GRID_COLUMNS:
        table = _grid_table(raw, column, logs.TRANSFORMER_SKETCHED, milestones)
        ctx.store.table(f"grid_{column}.csv", table)
        defined = table[["gen_min", "ctrl_max"]].notna().all(axis=1)
        separating[column] = {"cells": int(len(table)),
                              "defined": int(defined.sum()),
                              "separates": int(table["separates"].sum())}
    ctx.note("grid", separating)
    ctx.note("n_rows", int(len(raw)))


def _direct_columns(windows, runs: Sequence[str],
                    milestones_by_run: Dict[str, Tuple[Any, Any]]):
    """The direct measurement's participation ratios and norm at each window midpoint."""
    windows = windows[windows["run"].isin(list(runs))].copy()
    windows["mid_step"] = 0.5 * (windows["right_step"] + windows["left_step"])
    direct = windows[["run", "mid_step", "fn_PR_pos_det", "PR_pos_det", "pnorm"]].rename(
        columns={"fn_PR_pos_det": "PR_det", "PR_pos_det": "PR_par_det"})
    direct["t_mem"] = direct["run"].map(lambda r: milestones_by_run.get(r, (None, None))[0])
    direct["t_gen"] = direct["run"].map(lambda r: milestones_by_run.get(r, (None, None))[1])
    return direct


# -- the surrogate control -------------------------------------------------------

#: Smoothing lengths in samples: 1,010, 2,010 and 4,010 optimiser steps on the modular
#: runs. It is the experiment's one free parameter, so it is swept -- a fall that survives
#: only one choice of it has not survived.
SURROGATE_SMOOTHING: Tuple[int, ...] = (101, 201, 401)

#: Thirty-nine surrogates plus the observed series resolve a p-value to 1/40 = 0.025.
SURROGATE_COUNT = 39

#: Five base seeds. A p-value resolved to 1/40 from a single draw is not a number to
#: quote, so the whole experiment is re-seeded and the spread across seedings is reported.
SURROGATE_SEEDS = 5

SURROGATE_COLUMNS: Tuple[str, ...] = ("run", "column", "smooth", "seed", "kind", "i",
                                      "depth", "n_windows", "grid_match")


def _surrogate_cell(args) -> List[Dict[str, Any]]:
    """One (run, observer, smoothing, seeding): the observed depth and its null."""
    (run, column, path, centres, t_gen, smooth, replicate, base_seed, cell, base,
     n_surr) = args

    from scipy.signal import savgol_filter

    from ..analysis import logs
    from ..estimator.mle import estimate
    from ..estimator.surrogates import iaaft
    from ..runtime.determinism import stream_seed

    frame = logs.load_log(path)
    if column not in frame.columns:
        return []
    x = frame[column].to_numpy(dtype=float)
    stride = logs.log_stride(frame)
    cfg = base.replace(window=logs.MATCHED_WINDOW, stride=1, **cell)

    def trace(series):
        grid, values = [], []
        for centre, window in logs.matched_windows(series, stride, centres):
            grid.append(centre)
            values.append(estimate(window, cfg).MG)
        return np.asarray(grid, dtype=float), np.asarray(values, dtype=float)

    # The shape is kept and only the fine structure is destroyed: a cubic Savitzky-Golay
    # smooth carries whatever the observer does at the transition, and the surrogate is
    # that shape plus a phase-randomised copy of the residual.
    length = min(smooth, len(x) - (1 - len(x) % 2))
    smoothed = savgol_filter(x, length, 3)
    residual = x - smoothed

    # One stream per cell, derived from the run's base seed by a stable rule. The archived
    # control seeded from `abs(hash((run, col, smooth)))`, which Python salts per
    # interpreter, so its committed table was produced under a seed nobody can recover and
    # a re-run reproduced only eight of its eighteen parameter-norm p-values.
    rng = np.random.default_rng(stream_seed(
        int(base_seed), f"surrogate/{run}/{column}/{smooth}/{replicate}"))

    observed_grid, observed_values = trace(x)
    observed = logs.depth(observed_grid, observed_values, t_gen)
    rows = [{"run": run, "column": column, "smooth": smooth, "seed": int(replicate),
             "kind": "observed", "i": -1, "depth": observed,
             "n_windows": len(observed_grid), "grid_match": True}]
    for index in range(int(n_surr)):
        # `match=False` is the published control. The estimator matches endpoints by
        # default, which trims up to fifteen per cent of the series; here the surrogate has
        # to keep the observed record's length or it stops sitting on the observed window
        # grid, and that grid is what makes this a paired comparison.
        #
        # Each surrogate is then scored on the grid it actually has. The archived control
        # paired a surrogate's values with the observed trace's grid, so a surrogate that
        # produced one constant window either crashed the call or -- when a different
        # window had been dropped and the lengths still matched -- silently compared two
        # traces sampled at different instants. `grid_match` records the difference per
        # row instead of hiding it.
        grid, values = trace(smoothed + iaaft(residual, iters=100, rng=rng, match=False))
        rows.append({
            "run": run, "column": column, "smooth": smooth, "seed": int(replicate),
            "kind": "surrogate", "i": index,
            "depth": logs.depth(grid, values, t_gen),
            "n_windows": len(grid),
            "grid_match": bool(len(grid) == len(observed_grid)
                               and np.array_equal(grid, observed_grid)),
        })
    return rows


@experiment(
    id="grok.matched.surrogate",
    title="Shape-preserving surrogates of the matched-window fall",
    paper=("sec:matched", "app:window", "tab:matched"),
    device=CPU,
    minutes=8,
    needs=("train.transformer.sketched", "grok.rank.dip"),
    promotes=("surrogates.csv", "surrogate_summary.csv", "surrogate_seed_spread.csv"),
    tier=3,
    notes="Thirty-nine surrogates per cell at three smoothing lengths and five seedings. "
          "The surrogates are unmatched at the endpoints, which is what keeps them on the "
          "observed window grid.",
)
def matched_surrogate(ctx: Context) -> None:
    import pandas as pd

    from .. import frozen
    from ..analysis import logs

    runs: Sequence[str] = logs.TRANSFORMER_SKETCHED
    columns: Sequence[str] = HEADLINE_COLUMNS
    smoothing: Sequence[int] = SURROGATE_SMOOTHING
    replicates = list(range(SURROGATE_SEEDS))
    n_surr = SURROGATE_COUNT
    if ctx.fast:
        runs = ("mod_wd0", "mod_wd1")
        columns, smoothing, replicates, n_surr = columns[:1], smoothing[:1], [0, 1], 4

    paths, _, milestones, centres = _matched_inputs(ctx, runs)
    # A control has no generalisation step of its own, so it is measured in the window its
    # matched run defines. Resolved once, here, and carried into every job.
    transition = {run: logs.transition_of(run, milestones) for run in runs}

    base = frozen.eight_direction()
    ctx.config(smoothing=list(smoothing), n_surrogates=n_surr,
               seedings=len(replicates), base_seed=ctx.seed,
               cell=dict(MATCHED_HEADLINE), endpoint_matching=False)
    ctx.note("surrogate_seed_rule",
             "stream_seed(base_seed, 'surrogate/<run>/<observer>/<smoothing>/<seeding>')")

    jobs = [(run, column, paths[run], centres[run], transition[run], smooth, replicate,
             ctx.seed, MATCHED_HEADLINE, base, n_surr)
            for run in runs for column in columns
            for smooth in smoothing for replicate in replicates]
    results = map_ordered(_surrogate_cell, jobs, jobs=ctx.jobs, desc="surrogates")
    frame = pd.DataFrame([row for result in results for row in result],
                         columns=list(SURROGATE_COLUMNS))
    ctx.store.table("surrogates.csv", frame)

    summary_rows: List[Dict[str, Any]] = []
    for (run, column, smooth, seed), group in frame.groupby(
            ["run", "column", "smooth", "seed"], sort=True):
        observed = float(group[group["kind"] == "observed"]["depth"].iloc[0])
        draws = group[group["kind"] == "surrogate"]["depth"].to_numpy(dtype=float)
        draws = draws[np.isfinite(draws)]
        summary_rows.append({
            "run": run, "column": column, "smooth": int(smooth), "seed": int(seed),
            "observed": observed,
            "surr_median": float(np.median(draws)) if len(draws) else float("nan"),
            "surr_max": float(np.max(draws)) if len(draws) else float("nan"),
            "p": ((1 + int((draws >= observed).sum())) / (1 + len(draws))
                  if len(draws) and np.isfinite(observed) else float("nan")),
            "n": int(len(draws)),
            "generalises": run in logs.GENERALISING,
        })
    summary = pd.DataFrame(summary_rows, columns=[
        "run", "column", "smooth", "seed", "observed", "surr_median", "surr_max", "p",
        "n", "generalises"]).sort_values(["column", "run", "smooth", "seed"])
    ctx.store.table("surrogate_summary.csv", summary)

    if len(summary):
        spread = (summary.groupby(["run", "column", "smooth"], sort=True)
                  .agg(generalises=("generalises", "first"),
                       observed=("observed", "first"), p_min=("p", "min"),
                       p_median=("p", "median"), p_max=("p", "max"), p_sd=("p", "std"),
                       n_seeds=("p", "size")).reset_index())
    else:
        spread = pd.DataFrame(columns=["run", "column", "smooth", "generalises",
                                       "observed", "p_min", "p_median", "p_max", "p_sd",
                                       "n_seeds"])
    ctx.store.table("surrogate_seed_spread.csv", spread)

    off_grid = int((~frame["grid_match"]).sum()) if len(frame) else 0
    ctx.note("surrogates_off_grid", off_grid)


# =============================================================================
# Appendix G -- what the 120,000-step reruns overturn
# =============================================================================

CHANCE_MODULUS = 113
"""The modulus of the extended reruns; chance accuracy is its reciprocal."""

EARLY_BUDGET = 20_000
"""The budget every conclusion in the archived project was drawn at."""


@experiment(
    id="grok.extended.outcomes",
    title="What the 120,000-step reruns settle about the controls",
    paper=("app:falls", "app:window", "fig:window"),
    device=CPU,
    minutes=1,
    needs=("train.transformer.extended",),
    promotes=("exp8_outcomes.csv", "exp8_at_20k.csv"),
    tier=3,
    notes="Two tables: the outcome at the full budget, and what each run looked like at "
          "step 20,000. Two configurations previously counted as negatives generalise.",
)
def extended_outcomes(ctx: Context) -> None:
    import pandas as pd
    from scipy.stats import spearmanr

    from ..analysis import logs

    runs: Sequence[str] = logs.TRANSFORMER_EXTENDED
    if ctx.fast:
        runs = runs[:2]
    ctx.config(runs=list(runs), chance=1.0 / CHANCE_MODULUS, early_budget=EARLY_BUDGET)

    chance = 1.0 / CHANCE_MODULUS
    outcomes: List[Dict[str, Any]] = []
    early: List[Dict[str, Any]] = []
    for run in runs:
        frame = logs.load_log(logs.find_log(ctx, "train.transformer.extended", run))
        step = frame["step"].to_numpy()
        val = frame["val_acc"].to_numpy(dtype=float)
        t_mem = logs.first_sustained(step, frame["train_acc"].to_numpy(dtype=float))
        t_gen = logs.first_sustained(step, val)

        tail = float(val[step >= step.max() - 10_000].mean())
        outcomes.append({
            "run": run, "t_mem": t_mem, "t_gen": t_gen,
            "gap": (t_gen - t_mem) if t_gen is not None and t_mem is not None else None,
            "x_chance": tail / chance, "max_val": float(val.max()),
            "groks": t_gen is not None,
        })

        inside = step <= EARLY_BUDGET
        s, v = step[inside], val[inside]
        level = float(v[s >= EARLY_BUDGET - 2000].mean())
        half = s >= EARLY_BUDGET // 2
        # A run pinned at one value has no trend to measure, and a rank correlation on a
        # constant is 0/0; NaN is the honest answer and the guard is what keeps it out of
        # the warning stream.
        trend = (float(spearmanr(s[half], v[half]).statistic)
                 if np.std(v[half]) > 0 else float("nan"))
        early.append({"run": run, "val_at_20k": level, "x_chance_at_20k": level / chance,
                      "rho_10k_20k": trend, "t_gen": t_gen})

    ctx.store.table("exp8_outcomes.csv", pd.DataFrame(
        outcomes, columns=["run", "t_mem", "t_gen", "gap", "x_chance", "max_val", "groks"]))
    ctx.store.table("exp8_at_20k.csv", pd.DataFrame(
        early, columns=["run", "val_at_20k", "x_chance_at_20k", "rho_10k_20k", "t_gen"]))
    ctx.note("generalise", [row["run"] for row in outcomes if row["groks"]])


# =============================================================================
# Appendix J -- the participation ratio against window length
# =============================================================================

#: Sixty samples per window at every length, spread further apart. The sample count sets
#: both the bound on the statistic and its noise floor, so holding it fixed is what makes
#: the ladder a test of the span alone.
PRWINDOW_STRIDES: Tuple[int, ...] = (1, 2, 4, 10, 20, 40)
PRWINDOW_SAMPLES = 60

#: The control ladder: every logged row, more of them per window. Reported beside the
#: first to say how much of any rise is sample count rather than span.
PRWINDOW_ROWS: Tuple[int, ...] = (60, 120, 240, 600, 1200, 2400)

PRWINDOW_MIN_WINDOWS = 3
PRWINDOW_RUNS: Tuple[str, ...] = ("a_add", "x_no_grok")


def _prwindow_length(args) -> List[Dict[str, Any]]:
    """Every placement of one window length, on one run, in one ladder."""
    run, sketch_path, ladder, stride, n_samples, log_every, t_gen = args

    from ..sketch.analysis import detrend, pr

    with np.load(sketch_path, allow_pickle=False) as loaded:
        step = np.asarray(loaded["step"])
        arrays = (("", np.asarray(loaded["z"])), ("fn_", np.asarray(loaded["zf"])))
        move = np.asarray(loaded["param_step"], dtype=float)

    span = stride * (n_samples - 1) + 1          # rows the window occupies
    total = len(step)
    if total < span + PRWINDOW_MIN_WINDOWS:
        return []

    # Enough placements to see the spread without recomputing a 2,400-row decomposition
    # hundreds of times: a long window costs quadratically more and consecutive placements
    # of a window covering a fifth of the run overlap almost completely.
    target = 40 if span <= 240 else 12
    advance = max(1, (total - span) // target)

    rows: List[Dict[str, Any]] = []
    for a in range(0, total - span + 1, advance):
        b = a + span
        # The window spans stride*(n-1)+1 rows, so it covers stride*(n-1) logging
        # intervals of optimiser steps. The archived label was stride*n*log_every, one
        # interval too many: the row the article prints as 600 steps covers 590. The
        # article quotes those labels; docs/errata.md item 7.
        record: Dict[str, Any] = {
            "n_samples": n_samples, "row_stride": stride,
            "window_steps": int(stride * (n_samples - 1) * log_every),
            "window_rows": int(span), "log_every": int(log_every),
            "left_step": int(step[a]), "right_step": int(step[b - 1]),
            "centre_step": int(0.5 * (step[a] + step[b - 1])),
        }
        for prefix, array in arrays:
            per_sketch = {"pos": [], "pos_det": [], "step": []}
            for s in range(array.shape[1]):
                window = array[a:a + stride * n_samples:stride, s, :]
                per_sketch["pos"].append(pr(window))
                per_sketch["pos_det"].append(pr(detrend(window)))
                per_sketch["step"].append(pr(np.diff(window, axis=0)))
            for key, values in per_sketch.items():
                finite = [v for v in values if np.isfinite(v)]
                record[f"{prefix}PR_{key}"] = float(np.mean(finite)) if finite else np.nan
        window_move = move[a:b][np.isfinite(move[a:b])]
        record["move"] = float(window_move.sum()) if len(window_move) else float("nan")
        record["run"] = run
        record["ladder"] = ladder
        record["t_gen"] = t_gen
        rows.append(record)
    return rows


PRWINDOW_STATS: Tuple[str, ...] = ("PR_pos_det", "PR_step", "fn_PR_pos_det", "fn_PR_step")


def _prwindow_summary(frame, t_mem, t_gen):
    """Median, range and the window straddling generalisation, per window length."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for (n, stride, steps), group in frame.groupby(
            ["n_samples", "row_stride", "window_steps"], sort=True):
        record: Dict[str, Any] = {"n_samples": int(n), "row_stride": int(stride),
                                  "window_steps": int(steps), "n_windows": int(len(group))}
        for stat in PRWINDOW_STATS:
            record[f"{stat}_med"] = float(group[stat].median())
            record[f"{stat}_min"] = float(group[stat].min())
            record[f"{stat}_max"] = float(group[stat].max())
        record["at_gen_centre"] = np.nan
        for stat in PRWINDOW_STATS:
            record[f"{stat}_at_gen"] = np.nan
        if t_gen is not None:
            nearest = (group["centre_step"] - t_gen).abs().idxmin()
            record["at_gen_centre"] = int(group["centre_step"][nearest])
            for stat in PRWINDOW_STATS:
                record[f"{stat}_at_gen"] = float(group[stat][nearest])
        record["t_mem"], record["t_gen"] = t_mem, t_gen
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["n_samples", "window_steps"])


@experiment(
    id="grok.prwindow",
    title="The full-batch participation ratio against window length",
    paper=("app:fb", "tab:prwindow", "fig:prwindow"),
    device=CPU,
    minutes=15,
    needs=("train.perceptron.sketched.long",),
    promotes=("pr_vs_window.csv",),
    tier=3,
    notes="Two 150,000-step sketched runs, windows spanning 600 to 120,000 steps. The "
          "archived sketches were never kept, so this needs the GPU campaign first, and "
          "its cost is the one estimate here that could not be measured: the long ladder "
          "decomposes 2,400-row windows and the estimate assumes the sketch width the "
          "trainer writes.",
)
def prwindow(ctx: Context) -> None:
    import pandas as pd

    from ..analysis import logs
    from ..sketch.analysis import milestones

    runs: Sequence[str] = PRWINDOW_RUNS
    strides: Sequence[int] = PRWINDOW_STRIDES
    rows_ladder: Sequence[int] = PRWINDOW_ROWS
    if ctx.fast:
        runs, strides, rows_ladder = runs[:1], strides[:2], rows_ladder[:2]

    upstream = "train.perceptron.sketched.long"
    jobs: List[Any] = []
    milestone_by_run: Dict[str, Tuple[Any, Any]] = {}
    for run in runs:
        log = logs.load_log(logs.find_log(ctx, upstream, run))
        sketch_path = str(logs.require_sketch(ctx, upstream, run))
        t_mem, t_gen = milestones(log)
        milestone_by_run[run] = (t_mem, t_gen)
        log_every = logs.log_stride(log)
        for ladder, pairs in (("fixed_n", [(s, PRWINDOW_SAMPLES) for s in strides]),
                              ("fixed_dt", [(1, n) for n in rows_ladder])):
            for stride, n_samples in pairs:
                jobs.append((run, sketch_path, ladder, stride, n_samples, log_every,
                             t_gen))

    ctx.config(runs=list(runs), samples_per_window=PRWINDOW_SAMPLES,
               row_strides=list(strides), row_counts=list(rows_ladder),
               window_label="stride * (n_samples - 1) * log_every")

    results = map_ordered(_prwindow_length, jobs, jobs=ctx.jobs, desc="pr vs window")
    rows = [row for result in results for row in result]
    if not rows:
        raise RuntimeError(
            "no window of any length fits these sketches; the runs are too short. "
            f"Re-train them with\n  python -m actdim run {upstream}")

    lead = ["run", "ladder", "n_samples", "row_stride", "window_steps", "window_rows",
            "log_every", "left_step", "right_step", "centre_step"]
    windows = pd.DataFrame(rows)
    windows = windows[lead + [c for c in windows.columns if c not in lead]]
    ctx.store.table("pr_vs_window_windows.csv", windows)

    summaries = []
    for run in runs:
        for ladder in ("fixed_n", "fixed_dt"):
            group = windows[(windows["run"] == run) & (windows["ladder"] == ladder)]
            if not len(group):
                continue
            t_mem, t_gen = milestone_by_run[run]
            table = _prwindow_summary(group, t_mem, t_gen)
            table.insert(0, "run", run)
            table.insert(1, "ladder", ladder)
            summaries.append(table)
    ctx.store.table("pr_vs_window.csv", pd.concat(summaries, ignore_index=True))


# =============================================================================
# Appendix Q -- descent at the edge of stability
# =============================================================================

#: Logging strides the records are re-read at. Edge-of-stability oscillation is the
#: two-cycle of the unstable mode, so a stride of ten does not blur it but aliases it away,
#: and the published protocol would have been blind to it whatever the run did. These three
#: are the ones the committed table and appendix Q report.
EOS_SUBSAMPLES: Tuple[int, ...] = (1, 10, 50)

#: The frozen lag first, because it is the protocol as published and changing it per
#: dataset is the per-system tuning requirement 2 forbids; tau = 1 beside it, because
#: against a two-step cycle an even lag is close to the worst possible choice and the gap
#: between them is the honest size of that mismatch.
EOS_TAUS: Tuple[int, ...] = (4, 1)

EOS_COLUMNS: Tuple[str, ...] = ("train_loss", "weight_norm")
EOS_WINDOW_STEPS = 8000
EOS_STRIDE_STEPS = 2000

#: The article's admissible rectangle. It describes the calibration of section 6 and is not
#: a validated decision rule, which is why it is applied here rather than trusted.
EOS_ADMISSIBLE_IDENT = 1.10
EOS_ADMISSIBLE_CROSSINGS = 8

EOS_DIAGNOSTIC_COLUMNS: Tuple[str, ...] = (
    "run", "lr", "seed", "column", "segment", "subsample", "tau", "t_grok",
    "eta_lam_over_2", "start_sample", "n_samples", "MG", "MG_2E", "ident_ratio",
    "PRdelay", "roughness", "acorr", "crossings", "rises", "degenerate")

EOS_RECURRENCE_COLUMNS: Tuple[str, ...] = (
    "run", "lr", "seed", "window_start", "nn_over_travel", "nn_over_scale", "points",
    "rises", "eta_lam_over_2", "at_eos", "diverged")


def _rises(x: np.ndarray) -> float:
    """Fraction of consecutive samples on which the series increases.

    The cheapest non-monotonicity statistic there is, and the one that needs no embedding:
    at the edge of stability it should be near a half and on a monotone descent exactly
    zero.
    """
    return float((np.diff(x) > 0).mean()) if len(x) > 1 else float("nan")


def _eos_cell(args) -> List[Dict[str, Any]]:
    """One (run, observer, segment, logging stride, lag) of the edge-of-stability sweep."""
    (key, path, column, segment, start, subsample, tau, t_grok, lr, seed, eta,
     base) = args

    from ..analysis import logs
    from ..estimator import windows
    from ..estimator.diagnostics import ratio, trend_crossings
    from ..estimator.mle import estimate

    frame = logs.load_log(path)
    if column not in frame.columns:
        return []
    series = frame[column].to_numpy(dtype=float)[start:][::subsample]
    if len(series) < 8:
        return []

    # The window and the stride are in optimiser steps, so changing the logging stride
    # changes the number of samples in a window and not the span of training it covers.
    # Holding the sample count fixed instead would silently lengthen the window.
    window = max(4, EOS_WINDOW_STEPS // subsample)
    stride = max(1, EOS_STRIDE_STEPS // subsample)
    cfg = base.replace(tau=tau, window=window, stride=stride)
    doubled = cfg.replace(max_E=2 * cfg.max_E)

    rows: List[Dict[str, Any]] = []
    for begin in windows.window_starts(len(series), cfg):
        segment_values = series[begin:begin + window]
        if (len(segment_values) < window or not np.isfinite(segment_values).all()
                or segment_values.std() <= 1e-12):
            continue
        scored = windows.score(segment_values, cfg)
        at_2e = estimate(segment_values, doubled).MG
        rows.append({
            "run": key, "lr": lr, "seed": seed, "column": column, "segment": segment,
            "subsample": subsample, "tau": tau, "t_grok": t_grok,
            "eta_lam_over_2": eta, "start_sample": int(begin), "n_samples": int(window),
            "MG": scored["MG"], "MG_2E": at_2e, "ident_ratio": ratio(scored["MG"], at_2e),
            "PRdelay": scored["PRdelay"], "roughness": scored["roughness"],
            "acorr": scored["acorr"], "crossings": trend_crossings(segment_values),
            "rises": _rises(segment_values), "degenerate": bool(scored["degenerate"]),
        })
    return rows


def _recurrence(x: np.ndarray, max_e: int = 20, tau: int = 1, exclusion: int = 150,
                probes: int = 2000) -> Tuple[float, float, int]:
    """Does the orbit come back, or is it the same curve continuing past the exclusion?

    For each reconstructed point, the distance to its nearest neighbour surviving the
    Theiler exclusion, over the distance the orbit itself travels during that exclusion.
    Dividing by the radius of the cloud is the obvious normalisation and it is the wrong
    one: a trajectory that merely moves slowly also has close surviving neighbours without
    ever returning. Both are reported; only the first discriminates.
    """
    from scipy.spatial import cKDTree

    from ..estimator.embedding import delay_embedding, standardise

    points = delay_embedding(standardise(x), max_e, tau)
    n = len(points)
    tree = cKDTree(points)
    # The excluded band holds at most 2W+1 points, so 2W+5 candidates always leave one.
    k = min(n - 1, 2 * exclusion + 5)
    distances, indices = tree.query(points, k=k + 1)
    times = np.arange(n)
    step = max(1, (n - exclusion) // probes)

    nearest, travel = [], []
    for i in range(0, n - exclusion, step):
        valid = np.abs(indices[i] - times[i]) > exclusion
        valid[0] = False                      # the point itself
        if valid.any():
            nearest.append(distances[i][valid][0])
            travel.append(float(np.linalg.norm(points[i + exclusion] - points[i])))
    nearest, travel = np.asarray(nearest), np.asarray(travel)
    if nearest.size == 0:
        return float("nan"), float("nan"), 0
    moved = travel > 0
    scale = float(np.sqrt(np.sum(points.var(axis=0))))
    return (float(np.median(nearest[moved] / travel[moved])) if moved.any() else np.nan,
            float(np.median(nearest) / scale) if scale > 0 else float("nan"),
            int(nearest.size))


def _eos_controls(n: int, exclusion: int, max_e: int, tau: int,
                  period: float = 400.0) -> List[Dict[str, Any]]:
    """The two reference series the return statistic is only interpretable against."""
    t = np.arange(n)
    phi = 0.5 * (1 + 5 ** 0.5)                # an incommensurate second frequency
    torus = np.sin(2 * np.pi * t / period) + 0.8 * np.sin(2 * np.pi * t / (period * phi))
    decay = np.exp(-t / (n / 2.667))
    rows = []
    for name, series in (("control: 2-torus", torus), ("control: monotone decay", decay)):
        ratio_travel, ratio_scale, points = _recurrence(series, max_e, tau, exclusion)
        rows.append({"run": name, "lr": np.nan, "seed": np.nan, "window_start": 0,
                     "nn_over_travel": ratio_travel, "nn_over_scale": ratio_scale,
                     "points": points, "rises": _rises(series),
                     "eta_lam_over_2": np.nan, "at_eos": False, "diverged": False})
    return rows


def _eos_recurrence_run(args) -> List[Dict[str, Any]]:
    """The return statistic over the post-transition windows of one run."""
    key, path, column, start, window, stride, lr, seed, eta, at_eos, diverged = args

    from ..analysis import logs

    x = logs.load_log(path)[column].to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for begin in range(start, len(x) - window + 1, stride):
        piece = x[begin:begin + window]
        if not np.isfinite(piece).all() or piece.std() <= 0:
            continue
        ratio_travel, ratio_scale, points = _recurrence(piece)
        rows.append({"run": key, "lr": lr, "seed": seed, "window_start": int(begin),
                     "nn_over_travel": ratio_travel, "nn_over_scale": ratio_scale,
                     "points": points, "rises": _rises(piece), "eta_lam_over_2": eta,
                     "at_eos": at_eos, "diverged": diverged})
    return rows


@experiment(
    id="grok.eos",
    title="The two diagnostics, and recurrence, on the edge-of-stability logs",
    paper=("app:eos", "tab:eos", "fig:eos"),
    device=CPU,
    minutes=9,
    needs=("train.perceptron.eos",),
    promotes=("eos_diagnostics.csv", "eos_diagnostics_summary.csv",
              "eos_recurrence.csv"),
    tier=3,
    notes="Reads the campaign table and skips every run that diverged: a run that blew "
          "up after a few hundred steps is not a trajectory and has no post-transition "
          "segment to slice. Nine minutes measured on eight cores over the eleven "
          "surviving runs, against the hour the archived script took at its own defaults, "
          "which sweep five logging strides and three lags rather than the three and two "
          "the committed table and appendix Q report.",
)
def eos(ctx: Context) -> None:
    import pandas as pd

    from .. import frozen
    from ..analysis import logs
    from ..training.eos import analysable

    campaign = pd.read_csv(ctx.input("train.perceptron.eos", "eos_runs.csv"))
    records = campaign.to_dict("records")

    # The guard, applied once and from the training module that owns it. The archived
    # analysis had two paths over these logs, one of which required `diverged_at is None`
    # and one of which did not, and the second sliced post-transition segments out of a
    # run that lasted 567 steps in total.
    usable = [r for r in records if analysable(_clean(r))]
    skipped = [str(r["key"]) for r in records if not analysable(_clean(r))]

    subsamples: Sequence[int] = EOS_SUBSAMPLES
    taus: Sequence[int] = EOS_TAUS
    columns: Sequence[str] = EOS_COLUMNS
    if ctx.fast:
        usable, subsamples, taus, columns = usable[:1], subsamples[:1], taus[:1], columns[:1]

    ctx.config(runs=[str(r["key"]) for r in usable], skipped_diverged=skipped,
               subsamples=list(subsamples), taus=list(taus), observers=list(columns),
               window_steps=EOS_WINDOW_STEPS, stride_steps=EOS_STRIDE_STEPS)

    base = frozen.eight_direction()
    jobs: List[Any] = []
    # Sorted by the file name the archived campaign wrote, so a regenerated table lines up
    # with the committed one row for row.
    for record in sorted(usable, key=lambda r: str(r["key"])):
        key = str(record["key"])
        path = str(ctx.input("train.perceptron.eos", f"{key}_train.csv"))
        t_grok = _maybe_number(record.get("t_grok"))
        eta = _maybe_number(record.get("eta_lam_over_2_median_tail"))
        n_rows = int(record.get("n_rows") or 0)
        segments = [("all", 0)]
        # Sharpness only reaches 2/eta after the transition, so the whole record mixes a
        # monotone approach with whatever follows it.
        if t_grok is not None and int(t_grok) + 2000 < n_rows:
            segments.append(("post", int(t_grok)))
        for column in columns:
            for segment, start in segments:
                for subsample in subsamples:
                    for tau in taus:
                        jobs.append((key, path, column, segment, start, subsample, tau,
                                     t_grok, _maybe_number(record.get("lr")),
                                     _maybe_number(record.get("seed")), eta, base))

    results = map_ordered(_eos_cell, jobs, jobs=ctx.jobs, desc="eos diagnostics")
    frame = pd.DataFrame([row for result in results for row in result],
                         columns=list(EOS_DIAGNOSTIC_COLUMNS))
    ctx.store.table("eos_diagnostics.csv", frame)

    if len(frame):
        summary = (frame.groupby(["run", "lr", "column", "segment", "subsample", "tau"],
                                 sort=True)
                   .agg(MG=("MG", "median"), ident=("ident_ratio", "median"),
                        PRdelay=("PRdelay", "median"), crossings=("crossings", "median"),
                        rises=("rises", "median"), degen=("degenerate", "mean"),
                        eta_lam=("eta_lam_over_2", "first"), n=("MG", "size"))
                   .reset_index())
        summary["admissible"] = ((summary["ident"] <= EOS_ADMISSIBLE_IDENT)
                                 & (summary["crossings"] > EOS_ADMISSIBLE_CROSSINGS))
    else:
        summary = pd.DataFrame(columns=[
            "run", "lr", "column", "segment", "subsample", "tau", "MG", "ident",
            "PRdelay", "crossings", "rises", "degen", "eta_lam", "n", "admissible"])
    ctx.store.table("eos_diagnostics_summary.csv", summary)
    ctx.note("admissible_cells", int(summary["admissible"].sum()) if len(summary) else 0)

    # -- recurrence, which neither diagnostic measures --------------------------
    recurrence_jobs: List[Any] = []
    for record in sorted(usable, key=lambda r: str(r["key"])):
        key = str(record["key"])
        eta = _maybe_number(record.get("eta_lam_over_2_median_tail"))
        t_grok = _maybe_number(record.get("t_grok"))
        recurrence_jobs.append((
            key, str(ctx.input("train.perceptron.eos", f"{key}_train.csv")),
            EOS_COLUMNS[0], 0 if t_grok is None else int(t_grok), EOS_WINDOW_STEPS,
            2 * EOS_STRIDE_STEPS, _maybe_number(record.get("lr")),
            _maybe_number(record.get("seed")), eta,
            bool(eta is not None and eta > 0.9), False))
    controls = _eos_controls(EOS_WINDOW_STEPS, 150, 20, 1)
    found = map_ordered(_eos_recurrence_run, recurrence_jobs, jobs=ctx.jobs,
                        desc="eos recurrence")
    ctx.store.table("eos_recurrence.csv", pd.DataFrame(
        controls + [row for result in found for row in result],
        columns=list(EOS_RECURRENCE_COLUMNS)))


def _clean(record: Dict[str, Any]) -> Dict[str, Any]:
    """A campaign row with pandas' NaN turned back into None.

    ``diverged_at`` is empty for every run that did not diverge, and pandas reads an empty
    numeric cell as NaN, which is not None and would make ``analysable`` admit every run.
    """
    out = dict(record)
    for key in ("diverged_at", "t_grok", "t_memorise"):
        value = out.get(key)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            out[key] = None
    return out


def _maybe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else number


# =============================================================================
# Appendix M -- the representation in closed form
# =============================================================================

#: Widths the closed form is scored at. The article says the analytic solution needs about
#: ninety neurons, so the sweep has to straddle that rather than start above it.
REPR_WIDTHS: Tuple[int, ...] = (50, 100, 200, 500)

#: Node budget for the complete decomposition search. Returning "no decomposition exists"
#: is a proof only if the search was exhaustive, so a budget that ran out is reported as
#: undecided and never as a negative.
REPR_BUDGET = 40_000


@experiment(
    id="grok.repr",
    title="The representation dimension in closed form, and what lies outside it",
    paper=("app:repr", "tab:ipr", "sec:pairs"),
    device=CPU,
    minutes=4,
    promotes=("repr_reference.csv", "repr_analytic.csv", "repr_decomposition.csv"),
    tier=3,
    notes="No training and no logs: the closed form of appendix M, scored on the whole "
          "p x p table, and a proof that the three perturbed polynomials are outside the "
          "class the architecture can represent.",
)
def representation(ctx: Context) -> None:
    import pandas as pd

    from ..analysis import representation as repr_
    from ..tasks import polynomials

    p = 97
    widths: Sequence[int] = REPR_WIDTHS
    moduli: Sequence[int] = (97, 23)
    budget = REPR_BUDGET
    if ctx.fast:
        widths, moduli, budget = widths[:1], moduli[:1], 2_000

    ctx.config(p=p, widths=list(widths), moduli=list(moduli), search_budget=budget)
    # The closed form gives each neuron an independent phase, and the draw is a stream of
    # the run's own seed rather than the literal zero the archived script used. Two of
    # appendix M's three numbers do not depend on it -- the mode count is (p+1)/2 and the
    # order parameter is exactly 1.000 for a solution periodic in the raw operand -- but
    # the effective rank does: it reads 148.8 at seed zero, which is the published value,
    # and 145.2 to 150.0 over the first few seeds. That is the appendix's own point made
    # measurable, since it declines to treat agreement with 148.8 as evidence of anything
    # when random initialisation already reads 139.
    ctx.declare_seeds("closed_form_phases")
    seed = ctx.seed_for("closed_form_phases")

    # -- the reference values appendix M reads the measured ones against --------
    tasks = list(repr_.ARITHMETIC_CASES) + list(repr_.POLYNOMIAL_DECOMPOSITIONS)
    floor = repr_.initialisation_reference(p, width=500, seed=seed)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        record = repr_.reference(task, p, width=500, seed=seed)
        if record is None:
            continue
        # The three numbers are not commensurable and the article says so: only the mode
        # count is a dimension the closed form fixes, the order parameter is only
        # interpretable against its own task's reference, and the effective rank reads
        # within seven per cent of the closed form before any training.
        record["order_parameter_blocks"] = ";".join(
            f"{v:.6f}" for v in record.pop("order_parameter_blocks"))
        record["init_order_parameter"] = floor["order_parameter"]
        record["init_effective_rank"] = floor["effective_rank"]
        rows.append(record)
    ctx.store.table("repr_reference.csv", pd.DataFrame(rows))
    ctx.note("initialisation_reference", floor)

    # -- the closed form scored on the whole table -----------------------------
    scored: List[Dict[str, Any]] = []
    for task in tasks:
        for width in widths:
            weights = repr_.closed_form(task, p, width, seed=seed)
            scored.append({"task": task, "p": p, "width": width,
                           **repr_.score_weights(*weights, p=p, task=task)})
    ctx.store.table("repr_analytic.csv", pd.DataFrame(scored))

    # -- whether a decomposition exists at all ----------------------------------
    verdicts: List[Dict[str, Any]] = []
    for modulus in moduli:
        for name, decomposition in repr_.decompose_all(modulus, budget=budget).items():
            verdicts.append({"expression": polynomials.EXPRESSIONS[name],
                             "learnable": polynomials.is_learnable(name),
                             **decomposition.summary()})
    table = pd.DataFrame(verdicts)
    ctx.store.table("repr_decomposition.csv", table)

    undecided = sorted(table.loc[table["verdict"] == "undecided", "name"]) if len(table) else []
    if undecided:
        # A search that ran out of nodes decided nothing, and reporting it as a negative
        # is how a budget becomes a proof.
        ctx.note("undecided", undecided)
    ctx.note("outside_the_class",
             sorted(table.loc[table["verdict"] == "none", "name"]) if len(table) else [])


#: The eleven rows of appendix M's measured table, as (run key, the closed form it is read
#: against). A run whose task has no closed form is read against nothing: appendix M
#: declines to substitute another task's reference, so those rows print a dash.
REPR_MEASURED: Tuple[Tuple[str, Optional[str]], ...] = (
    ("a_add", "add"), ("a_sub", "sub"), ("a_sq_sum", "sq_sum"), ("a_mul", None),
    ("g_p1", "p1"), ("g_p2", "p2"), ("g_p3", "p3"),
    ("g_p1x", None), ("g_p2x", None), ("g_p3x", None), ("x_no_grok", None),
)

#: Which campaign promoted each run's weight snapshots.
REPR_CAMPAIGN = {"g": "train.perceptron.poly"}

#: The four points appendix M follows the order parameter through, as the milestone each
#: is nearest. The snapshots are log-spaced, so a point is the first snapshot at or after
#: the step it names and the step actually used is recorded beside it.
REPR_POINTS: Tuple[Tuple[str, str], ...] = (
    ("just after memorisation", "t_memorise"),
    ("just before generalisation", "before_t_grok"),
    ("validation accuracy first reaches 100%", "t_perfect"),
    ("end of the budget", "last"),
)


@experiment(
    id="grok.repr.measured",
    title="The order parameter of the trained first layer, against its own closed form",
    paper=("app:repr", "tab:ipr", "tab:ipr-trajectory"),
    device=CPU,
    minutes=3,
    needs=("train.perceptron.arith", "train.perceptron.poly", "grok.repr"),
    promotes=("repr_measured.csv", "repr_trajectory.csv"),
    tier=3,
    notes="Reads the weight snapshots the perceptron campaigns write and scores them with "
          "the same Fourier inverse participation ratio grok.repr applies to the closed "
          "form. No training: appendix M's measured column had no source under data/ "
          "because nothing scored the trained weights.",
)
def representation_measured(ctx: Context) -> None:
    import numpy as np
    import pandas as pd

    from ..analysis import representation as repr_
    from ..training import runs_perceptron as registry

    rows: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, Any]] = []
    wanted = REPR_MEASURED[:2] if ctx.fast else REPR_MEASURED
    ctx.config(runs=[key for key, _ in wanted],
               statistic="Fourier inverse participation ratio of the first operand block")

    for key, form in wanted:
        config = registry.RUNS[key]
        campaign = REPR_CAMPAIGN.get(key.split("_")[0], "train.perceptron.arith")
        snapshots = np.load(ctx.input(campaign, f"{key}_snapshots.npz"))
        steps = sorted(int(n.split("_")[1]) for n in snapshots.files
                       if n.startswith("W1_"))
        if not steps:
            continue

        measured = repr_.fourier_ipr(snapshots[f"W1_{steps[-1]}"][:, :config.p])
        reference = (repr_.reference(form, config.p, width=config.width)["order_parameter"]
                     if form else None)
        rows.append({"run": key, "task": config.task, "p": config.p,
                     "closed_form": form or "", "step": steps[-1],
                     "order_parameter": measured,
                     "own_reference": reference if reference is not None else np.nan})
        print(f"    {key:<10} {measured:.3f}"
              + (f" against {reference:.3f}" if reference is not None else ""), flush=True)

        # The trajectory table follows one run, and follows it through the same snapshots.
        if key != "a_add":
            continue
        log = pd.read_csv(ctx.input(campaign, f"{key}_train.csv"))
        perfect = log.loc[log.val_acc >= 1.0, "step"]
        marks = {
            "t_memorise": registry.PAPER_MILESTONES[key]["memorise"],
            "before_t_grok": registry.PAPER_MILESTONES[key]["generalise"],
            "t_perfect": float(perfect.iloc[0]) if len(perfect) else np.nan,
            "last": float(steps[-1]),
        }
        for label, mark in REPR_POINTS:
            target = marks[mark]
            if target is None or not np.isfinite(target):
                continue
            # "just before" takes the last snapshot strictly earlier; the others take the
            # first at or after the step they name.
            if mark == "before_t_grok":
                at = max([s for s in steps if s < target], default=steps[0])
            else:
                at = min([s for s in steps if s >= target], default=steps[-1])
            trajectory.append({
                "run": key, "point": label, "milestone": mark,
                "milestone_step": target, "snapshot_step": at,
                "order_parameter": repr_.fourier_ipr(
                    snapshots[f"W1_{at}"][:, :config.p])})

    ctx.store.table("repr_measured.csv", pd.DataFrame(rows))
    ctx.store.table("repr_trajectory.csv", pd.DataFrame(trajectory))
    ctx.note("runs_scored", len(rows))

"""Where each figure's data comes from.

Every path the figures read is written here once. The archived generator resolved each
file against one ``CODE`` constant pointing into a tree of per-cluster ``results/``
directories, so a file's producer could only be recovered by reading the directory name
and guessing which script had last written it. Here a logical name maps to the
experiment that promotes the file and the name it promotes it under, and resolution goes
through :func:`actdim.runtime.store.data_root`, so no figure knows where anything lives.

Two kinds of source. Most are one file per experiment and are in :data:`SOURCES`. A few
are per-run logs -- the edge-of-stability training and sharpness logs of ``fig_eos``, the
polynomial training logs of ``fig_pairs`` -- which are one file per run name and are in
:data:`RUN_SOURCES`, keyed by run against the experiment that trains it.

**The archive fallback.** None of the article's data has been regenerated into ``data/``
yet. Passing ``allow_archive=True`` lets a name that is absent from ``data/`` resolve to
the corresponding file under ``../archived_code/`` instead, and the resolved
:class:`Source` then carries ``archived=True`` so that every caller can say so. It is a
migration aid and not a mode to leave on: a figure silently built from stale data is the
failure ``check.tables`` exists to catch, and the archived tree published one. Without
the flag an absent file raises, naming the experiment that produces it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from ..runtime.store import data_root, repo_root


class Source(NamedTuple):
    """One resolved input: what it is, who makes it, where it was found."""

    name: str
    experiment: str
    path: Path
    archived: bool


# -- the flat table ------------------------------------------------------------
#
# logical name -> (experiment id, the file that experiment promotes)

SOURCES: Dict[str, Tuple[str, str]] = {
    "sweep_raw":            ("sys.digits.parameter", "sweep_raw.csv"),
    "observer_scores":      ("sys.digits.parameter", "observer_scores.csv"),
    "real_logs_summary":    ("grok.diagnostics.logs", "real_logs_summary.csv"),
    "real_logs_windows":    ("grok.diagnostics.logs", "real_logs_windows.csv"),
    "tau_sensitivity":      ("valid.tau", "tau_sensitivity.csv"),
    "aniso_summary":        ("valid.anisotropy", "aniso_summary.csv"),
    "headline_trace":       ("grok.matched.window", "headline_trace.csv"),
    "ceiling_summary":      ("valid.ceiling", "ceiling_summary.csv"),
    "example_traces":       ("valid.theiler.contrast", "example_traces.csv"),
    "rank_windows":         ("grok.rank.dip", "rank_windows.csv"),
    "rank_milestones":      ("grok.rank.dip", "rank_milestones.json"),
    "mod_wd1_train":        ("grok.rank.dip", "mod_wd1_train.csv"),
    "exp8_outcomes":        ("grok.extended.outcomes", "exp8_outcomes.csv"),
    "probe_arith":          ("grok.diagnostics.perceptron", "dimension_probe.csv"),
    "probe_arith_summary":  ("grok.diagnostics.perceptron", "dimension_probe_summary.csv"),
    "probe_poly":           ("grok.diagnostics.perceptron", "dimension_probe_poly.csv"),
    "probe_poly_summary":   ("grok.diagnostics.perceptron", "dimension_probe_summary_poly.csv"),
    "pr_vs_window":         ("grok.prwindow", "pr_vs_window.csv"),
    "eos_runs":             ("train.perceptron.eos", "eos_runs.csv"),
    "eos_diagnostics":      ("grok.eos", "eos_diagnostics.csv"),
    "controls_scored":      ("valid.nuisance", "controls_scored.csv"),
}

# The same names under ``../archived_code/``, taken from the constants of
# ``icomp_v2/make_figures.py``. The arithmetic and polynomial probes are one file each
# there and collide by basename, which is why the port renames the polynomial pair: both
# are promoted by ``grok.diagnostics.perceptron`` into one directory.

ARCHIVE: Dict[str, str] = {
    "sweep_raw":            "active_dimension/results/e2_rank_sweep/sweep_raw.csv",
    "observer_scores":      "active_dimension/results/e2_rank_sweep/observer_scores.csv",
    "real_logs_summary":    "active_dimension/results/e5_real_logs/real_logs_summary.csv",
    "real_logs_windows":    "active_dimension/results/e5_real_logs/real_logs_windows.csv",
    "tau_sensitivity":      "active_dimension/results/e6_tau/tau_sensitivity.csv",
    "aniso_summary":        "active_dimension/results/e8_anisotropy/aniso_summary.csv",
    "headline_trace":       "active_dimension/results/e9_matched_window/headline_trace.csv",
    "ceiling_summary":      "active_dimension/results/e10_ceiling/ceiling_summary.csv",
    "example_traces":       "active_dimension/results/e11_theiler_contrast/example_traces.csv",
    "rank_windows":         "active_rank/results_fine/rank_windows.csv",
    "rank_milestones":      "active_rank/results_fine/rank_milestones.json",
    "mod_wd1_train":        "active_rank/results_fine/mod_wd1_train.csv",
    "exp8_outcomes":        "dimension_recovery/results/exp8_outcomes.csv",
    "probe_arith":          "gromov_arithmetic/results/arith/dimension_probe.csv",
    "probe_arith_summary":  "gromov_arithmetic/results/arith/dimension_probe_summary.csv",
    "probe_poly":           "gromov_polynomials/results/dimension_probe.csv",
    "probe_poly_summary":   "gromov_polynomials/results/dimension_probe_summary.csv",
    "pr_vs_window":         "gromov_arithmetic/results/rank_fb_long/pr_vs_window.csv",
    "eos_runs":             "gromov_arithmetic/results/eos/eos_runs.csv",
    "eos_diagnostics":      "gromov_arithmetic/results/eos/eos_diagnostics.csv",
    # No archived equivalent: the nuisance sweep was rerun for this port, and the
    # archived tree records only the summary the article printed.
}

# -- the per-run table ---------------------------------------------------------
#
# kind -> (experiment id, filename pattern, the archived directory it came from)

RUN_SOURCES: Dict[str, Tuple[str, str, str]] = {
    "eos_train":  ("train.perceptron.eos", "{run}_train.csv",
                   "gromov_arithmetic/results/eos"),
    "eos_sharp":  ("train.perceptron.eos", "{run}_sharp.csv",
                   "gromov_arithmetic/results/eos"),
    "poly_train": ("train.perceptron.poly", "{run}_train.csv",
                   "gromov_polynomials/results"),
}


def archive_root() -> Path:
    """The archived tree, beside ``code/``.

    ``icomp_v2/make_figures.py`` still points at ``../code``, which the port renamed:
    running it today reads this package's directory and finds no results at all.
    """
    return repo_root().parent / "archived_code"


def resolve(name: str, allow_archive: bool = False) -> Source:
    """Find one named dataset, in ``data/`` first and the archive only on request.

    A source with no archived equivalent is resolvable from ``data/`` alone; asking for
    the archive then fails with the name rather than with a KeyError on the lookup table.
    """
    if name not in SOURCES:
        raise KeyError(f"no such figure source: {name!r}")
    experiment, filename = SOURCES[name]
    return _resolve(name, experiment, filename, ARCHIVE.get(name), allow_archive)


def resolve_run(kind: str, run: str, allow_archive: bool = False) -> Source:
    """Find one per-run log, keyed by the run name the trainer used."""
    if kind not in RUN_SOURCES:
        raise KeyError(f"no such per-run source: {kind!r}")
    experiment, pattern, archive_dir = RUN_SOURCES[kind]
    filename = pattern.format(run=run)
    # A polynomial training log exists under either the label the article prints,
    # ``g_p2_p97_train.csv``, which is what the archived tree wrote, or the shorter
    # registry key this package's trainer writes. ``log_candidates`` is the one place that
    # knows both; the archived tree only ever had the first, so the fallback keeps using
    # the label. Without this a promoted re-run is invisible here and the figure quietly
    # goes on drawing the archived log.
    from ..analysis.logs import log_candidates

    names = log_candidates(run) if pattern == "{run}_train.csv" else (filename,)
    for candidate in names:
        current = data_root() / experiment / candidate
        if current.exists():
            return Source(f"{kind}:{run}", experiment, current, False)
    return _resolve(f"{kind}:{run}", experiment, filename,
                    f"{archive_dir}/{filename}", allow_archive)


def _resolve(name: str, experiment: str, filename: str, archive_rel: Optional[str],
             allow_archive: bool) -> Source:
    current = data_root() / experiment / filename
    if current.exists():
        return Source(name, experiment, current, False)

    if archive_rel is not None:
        old = archive_root() / archive_rel
        if allow_archive and old.exists():
            return Source(name, experiment, old, True)

    hint = ("" if allow_archive or archive_rel is None else
            "\nTo draw from the archived tree instead, pass allow_archive=True.")
    if archive_rel is None:
        hint = "\nThis source has no archived equivalent; it has to be run."
    raise FileNotFoundError(
        f"the figures need {filename!r} from {experiment!r}, which has not been "
        f"promoted.\nRun it first:  python -m actdim run {experiment}"
        f"\nthen:            python -m actdim promote {experiment}{hint}"
    )


class Reader:
    """The inputs of one build, and a record of where each came from.

    A figure asks for a logical name and never sees a path. The reader keeps every
    source it handed out, so that a figure drawn from stale data can be reported as
    such rather than looking exactly like a figure drawn from a fresh run.
    """

    def __init__(self, allow_archive: bool = False):
        self.allow_archive = allow_archive
        self.used: List[Source] = []

    # -- resolution ------------------------------------------------------------

    def source(self, name: str) -> Source:
        return self._keep(resolve(name, self.allow_archive))

    def run_source(self, kind: str, run: str) -> Source:
        return self._keep(resolve_run(kind, run, self.allow_archive))

    def _keep(self, source: Source) -> Source:
        if source not in self.used:
            self.used.append(source)
        return source

    # -- loading ---------------------------------------------------------------

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.source(name).path)

    def object(self, name: str) -> Any:
        return json.loads(self.source(name).path.read_text(encoding="utf-8"))

    def run_table(self, kind: str, run: str) -> pd.DataFrame:
        return pd.read_csv(self.run_source(kind, run).path)

    # -- what it read ----------------------------------------------------------

    @property
    def archived(self) -> List[str]:
        """The names that came from the archived tree, in the order they were read."""
        return [s.name for s in self.used if s.archived]

    def record(self) -> Dict[str, Any]:
        """What this reader read, as plain data fit for a provenance note."""
        return {
            "sources": {s.name: s.path.as_posix() for s in self.used},
            "archived": list(self.archived),
        }

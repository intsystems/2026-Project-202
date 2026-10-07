"""The article's figures, and the check on its tables.

These two are last in the order: they read what every other experiment produced and turn
it into what the document includes. Neither computes a result of its own, which is the
point -- a figure that computes something is a figure whose number cannot be checked.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..runtime import CPU, Context, experiment
from ..runtime.store import repo_root

#: Where the document expects to find its figures. The build step in `icomp_v2/` includes
#: them from here by name, so the port writes to the same place the archived generator did.
ARTICLE_FIGURES = repo_root().parent / "icomp_v2" / "figures"


#: What the figures read, taken from the one place the paths are written. Declared as
#: dependencies rather than left implicit: without them a full run draws the figures
#: wherever registration order happens to put them, which is before the section 6
#: experiments, and three of the twelve come out of stale data reporting themselves clean.
FIGURE_INPUTS = (
    "sys.digits.parameter",
    "valid.tau", "valid.anisotropy", "valid.ceiling", "valid.theiler.contrast",
    "valid.nuisance", "valid.curves", "valid.geometry",
    "train.perceptron.eos", "train.perceptron.poly",
    "grok.diagnostics.logs", "grok.diagnostics.perceptron", "grok.eos",
    "grok.extended.outcomes", "grok.matched.window", "grok.matched.surrogate",
    "grok.prwindow", "grok.rank.dip",
)


@experiment(
    id="paper.figures",
    title="The nineteen figures, into ../icomp_v2/figures/",
    paper=("fig:method", "fig:regimes", "fig:dip", "fig:observers", "fig:tau",
           "fig:aniso", "fig:map", "fig:pairs", "fig:prwindow", "fig:window", "fig:eos",
           "fig:ceiling", "fig:traces", "fig:signal", "fig:switch", "fig:shapes",
           "fig:exclusion", "fig:surrogate", "fig:timing"),
    device=CPU,
    minutes=2,
    needs=FIGURE_INPUTS,
    promotes=(),
    tier=5,
    notes="Set install=false to draw into runs/ only; allow_archive=true to build from "
          "../archived_code where an experiment has not been re-run.",
)
def figures(ctx: Context) -> None:
    from ..figures.panels import build, summary

    names = ctx.option("only", "")
    wanted = tuple(n.strip() for n in str(names).split(",") if n.strip()) if names else ()
    allow_archive = bool(ctx.option("allow_archive", False))

    record = build(ctx.store.dir, names=wanted, allow_archive=allow_archive)
    for entry in record["figures"].values():
        for path in entry["files"]:
            ctx.store.adopt(Path(path))

    ctx.config(figures=sorted(record["figures"]), allow_archive=allow_archive)
    ctx.note("sources", {name: entry.get("sources", [])
                         for name, entry in record["figures"].items()})
    print(summary(record))

    if record["archived_figures"]:
        # Not an error: it is the expected state until every experiment has been re-run.
        # It must never be silent, though -- a figure built from the archived tree shows
        # the article's old numbers under the new code's name.
        ctx.note("built_from_archive", record["archived_figures"])

    if ctx.option("install", True) and not ctx.fast:
        ARTICLE_FIGURES.mkdir(parents=True, exist_ok=True)
        installed = []
        for entry in record["figures"].values():
            for path in entry["files"]:
                target = ARTICLE_FIGURES / Path(path).name
                shutil.copy2(path, target)
                installed.append(target.name)
        ctx.note("installed", sorted(installed))
        print(f"\ninstalled {len(installed)} file(s) into "
              f"{ARTICLE_FIGURES.relative_to(repo_root().parent)}/")


@experiment(
    id="check.tables",
    title="Recompute every mechanical table cell and diff it against the printed value",
    paper=("tab:runs", "tab:ladder", "tab:alts", "tab:k20", "tab:controls",
           "tab:grok-diagnostics", "tab:dip", "tab:frozen"),
    device=CPU,
    minutes=1,
    # The auditor reads across the whole tree, so it goes last. Anything not yet
    # regenerated it reads from `data/`, and it says which of its inputs are still the
    # archived ones -- a table checked against archived data has been checked against the
    # numbers the article was written from, not against a regeneration.
    needs=FIGURE_INPUTS + ("calib.e8", "calib.e20", "sys.matrix", "sys.linear",
                           "sys.logistic", "sys.decoder", "sys.subspace",
                           "sys.digits.function", "train.transformer.sketched",
                           "train.perceptron.arith", "grok.matched.surrogate"),
    promotes=("table_audit.csv",),
    tier=5,
    notes="The release check. This repository has twice committed a result file its own "
          "script could no longer reproduce; run this before submitting anything.",
)
def tables(ctx: Context) -> None:
    import pandas as pd

    from ..tables import MISMATCH, ROUNDING, audit, format_report

    report = audit()
    rows = report.rows()
    ctx.store.table("table_audit.csv", pd.DataFrame(rows))
    text = format_report(report, verbose=bool(ctx.option("verbose", False)))
    ctx.store.text("table_audit.txt", text)
    print(text)

    mismatches = sum(1 for row in rows if row["status"] == MISMATCH)
    rounding = sum(1 for row in rows if row["status"] == ROUNDING)
    ctx.note("mismatches", int(mismatches))
    ctx.note("rounding", int(rounding))
    ctx.note("archived_inputs", report.archived_inputs)
    ctx.config(
        article=report.article,
        tables_checked=sorted(r.label for r in report.results if r.state == "checked"),
        tables_skipped=sorted(r.label for r in report.results if r.state == "skipped"),
        cells_compared=sum(1 for row in rows if row["kind"] != "table"),
    )

    # A mismatch is the whole point of this experiment, so it must not fail the run: the
    # report is the result, and a caller that stopped here would leave it unwritten. The
    # count is in the provenance and `python -m actdim.tables` exits non-zero for the
    # release check that wants a status rather than a file.
    if not report.ok:
        print(f"\n{len(report.mismatches)} cell(s) or claim(s) disagree with the article. "
              f"See runs/{ctx.experiment}/table_audit.csv.")

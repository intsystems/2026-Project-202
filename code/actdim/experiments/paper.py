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


@experiment(
    id="paper.figures",
    title="The twelve figures, into ../icomp_v2/figures/",
    paper=("fig:regimes", "fig:dip", "fig:observers", "fig:tau", "fig:aniso", "fig:map",
           "fig:pairs", "fig:prwindow", "fig:window", "fig:eos", "fig:ceiling",
           "fig:traces"),
    device=CPU,
    minutes=2,
    promotes=(),
    tier=6,
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
    promotes=("table_audit.csv",),
    tier=6,
    notes="The release check. This repository has twice committed a result file its own "
          "script could no longer reproduce; run this before submitting anything.",
)
def tables(ctx: Context) -> None:
    import pandas as pd

    from ..tables import audit, format_report

    report = audit()
    rows = report.get("rows") if isinstance(report, dict) else None
    if rows is None:
        rows = report if isinstance(report, list) else []
    frame = pd.DataFrame(rows)
    ctx.store.table("table_audit.csv", frame)
    text = format_report(report)
    ctx.store.text("table_audit.txt", text)
    print(text)

    mismatches = int(sum(1 for row in rows if row.get("status") == "mismatch"))
    ctx.note("mismatches", mismatches)
    ctx.config(tables_checked=sorted({row.get("table") for row in rows}))

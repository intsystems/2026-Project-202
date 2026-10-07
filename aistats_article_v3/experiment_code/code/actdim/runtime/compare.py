"""Diffing a regenerated table against the archived one.

The port fixes defects, and fixing them moves values. What matters is knowing which
values, so that a number in the article is either unchanged or deliberately updated, and
never quietly different. This module answers one question per file: what moved, by how
much, and where.

Rows are aligned on the columns that identify a cell -- seed, rank, observer, run, arm --
rather than on position, because the archived tables were written in process-pool
completion order and their row order carries no meaning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Columns that identify a row rather than measure one. Any of these present in both
# frames is used as a key; the rest of the alignment falls back to row order.
KEY_COLUMNS = (
    "run", "key", "tag", "arm", "mode", "system", "observer", "column", "statistic",
    "seed", "r", "rank", "k", "cfg_id", "config", "step", "window", "start", "mid",
    "tau", "max_E", "E", "N", "lr", "stride", "detrend", "cell",
)

TOLERANCE = 1e-9


def _keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    shared = [c for c in KEY_COLUMNS if c in left.columns and c in right.columns]
    if not shared:
        return []
    # Keep only a set that is actually unique, otherwise alignment is ambiguous.
    for size in range(len(shared), 0, -1):
        candidate = shared[:size]
        if not left.duplicated(candidate).any() and not right.duplicated(candidate).any():
            return candidate
    return []


def compare_tables(new: Path, old: Path, tolerance: float = TOLERANCE) -> Dict[str, Any]:
    """Compare two CSVs and describe every difference that matters."""
    left = pd.read_csv(new)
    right = pd.read_csv(old)

    report: Dict[str, Any] = {
        "file": new.name,
        "rows": {"new": int(len(left)), "archived": int(len(right))},
        "columns_added": sorted(set(left.columns) - set(right.columns)),
        "columns_removed": sorted(set(right.columns) - set(left.columns)),
        "keys": [],
        "changed": [],
        "unchanged_columns": [],
        "rows_added": 0,
        "rows_removed": 0,
    }

    shared = [c for c in left.columns if c in right.columns]
    if not shared:
        report["note"] = "no columns in common"
        return report

    keys = _keys(left[shared], right[shared])
    report["keys"] = keys
    if keys:
        merged = left[shared].merge(right[shared], on=keys, how="outer",
                                    suffixes=("__new", "__old"), indicator=True)
        report["rows_added"] = int((merged["_merge"] == "left_only").sum())
        report["rows_removed"] = int((merged["_merge"] == "right_only").sum())
        both = merged[merged["_merge"] == "both"]
    else:
        size = min(len(left), len(right))
        report["rows_added"] = max(0, len(left) - size)
        report["rows_removed"] = max(0, len(right) - size)
        both = pd.concat(
            [left[shared].head(size).add_suffix("__new").reset_index(drop=True),
             right[shared].head(size).add_suffix("__old").reset_index(drop=True)], axis=1)

    for column in shared:
        if column in keys:
            continue
        a, b = f"{column}__new", f"{column}__old"
        if a not in both.columns or b not in both.columns:
            continue
        new_values, old_values = both[a], both[b]
        if not (pd.api.types.is_numeric_dtype(new_values)
                and pd.api.types.is_numeric_dtype(old_values)):
            differing = int((new_values.astype(str) != old_values.astype(str)).sum())
            if differing:
                report["changed"].append({"column": column, "kind": "text",
                                          "cells": differing})
            else:
                report["unchanged_columns"].append(column)
            continue

        a_v = new_values.to_numpy(dtype=float)
        b_v = old_values.to_numpy(dtype=float)
        finite = np.isfinite(a_v) & np.isfinite(b_v)
        nan_flips = int((np.isnan(a_v) != np.isnan(b_v)).sum())
        if not finite.any():
            if nan_flips:
                report["changed"].append({"column": column, "kind": "nan",
                                          "cells": nan_flips})
            continue
        delta = np.abs(a_v[finite] - b_v[finite])
        scale = np.maximum(np.abs(b_v[finite]), 1e-12)
        cells = int((delta > tolerance).sum())
        if cells or nan_flips:
            worst = int(np.argmax(delta))
            report["changed"].append({
                "column": column,
                "kind": "numeric",
                "cells": cells,
                "of": int(finite.sum()),
                "max_abs": float(delta.max()),
                "max_rel": float((delta / scale).max()),
                "median_abs": float(np.median(delta[delta > tolerance])) if cells else 0.0,
                "nan_flips": nan_flips,
                "worst_new": float(a_v[finite][worst]),
                "worst_archived": float(b_v[finite][worst]),
            })
        else:
            report["unchanged_columns"].append(column)

    return report


def format_report(report: Dict[str, Any]) -> str:
    """One file's diff, as a few readable lines."""
    lines = [f"{report['file']}"]
    rows = report["rows"]
    if rows["new"] != rows["archived"]:
        lines.append(f"  rows      {rows['archived']} -> {rows['new']}"
                     f"  (+{report['rows_added']} / -{report['rows_removed']})")
    if report.get("note"):
        lines.append(f"  {report['note']}")
    if report["columns_added"]:
        lines.append(f"  added     {', '.join(report['columns_added'])}")
    if report["columns_removed"]:
        lines.append(f"  removed   {', '.join(report['columns_removed'])}")
    if not report["changed"]:
        lines.append(f"  unchanged ({len(report['unchanged_columns'])} columns)")
        return "\n".join(lines)
    for change in sorted(report["changed"],
                         key=lambda c: -(c.get("max_rel") or 0)):
        if change["kind"] == "numeric":
            lines.append(
                f"  {change['column']:<24} {change['cells']}/{change['of']} cells   "
                f"max |d| {change['max_abs']:.3g}   max rel {change['max_rel']:.3g}   "
                f"worst {change['worst_archived']:.6g} -> {change['worst_new']:.6g}")
        else:
            lines.append(f"  {change['column']:<24} {change['cells']} cells "
                         f"({change['kind']})")
    return "\n".join(lines)


def compare_experiment(experiment: str, names: Optional[Sequence[str]] = None,
                       tolerance: float = TOLERANCE) -> List[Dict[str, Any]]:
    """Diff every regenerated table of one experiment against its archived counterpart."""
    from .archive import baseline_names, baseline_path
    from .store import runs_root

    run_dir = runs_root() / experiment
    if not run_dir.exists():
        raise FileNotFoundError(
            f"{experiment} has not been run.\n  python -m actdim run {experiment}")

    wanted = list(names) if names else baseline_names(experiment)
    reports = []
    for name in wanted:
        if not name.endswith(".csv"):
            continue
        new = run_dir / name
        old = baseline_path(experiment, name)
        if not new.exists():
            reports.append({"file": name, "note": "not produced by the new run",
                            "rows": {"new": 0, "archived": 0}, "changed": [],
                            "columns_added": [], "columns_removed": [],
                            "unchanged_columns": [], "rows_added": 0, "rows_removed": 0})
            continue
        if old is None:
            reports.append({"file": name, "note": "no archived counterpart",
                            "rows": {"new": int(len(pd.read_csv(new))), "archived": 0},
                            "changed": [], "columns_added": [], "columns_removed": [],
                            "unchanged_columns": [], "rows_added": 0, "rows_removed": 0})
            continue
        reports.append(compare_tables(new, old, tolerance))
    return reports

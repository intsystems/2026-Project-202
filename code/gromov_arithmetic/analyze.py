"""Turn the fetched logs into the tables ``report.md`` quotes.

Everything here is descriptive; nothing fits a model or estimates a dimension.  The
dimension work lives in ``../active_dimension/`` and consumes ``<key>_train.csv``
directly.  What this file adds is the bookkeeping that makes a log interpretable:
when the run memorised, when it generalised, and what the representation looked like
at the end compared with the closed-form solution.

    python analyze.py --results ./results
    python analyze.py --results ./results --markdown

The reference column is the point.  ``ipr_ref`` is the mean Fourier IPR of the
analytic weights at the same width, computed on the spot, so "the representation
became periodic" is a comparison against a constructed solution rather than against
the assertion that a rising number means periodicity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analytic
import tasks
from gromov import Config, _fourier_ipr, _participation


def _decomposition(task):
    """``(inner, outer)`` for ``analytic.build``, or None if the task has no solution.

    Three sources, in order: the arithmetic cases of ``analytic.py``, the polynomial
    decompositions of ``../gromov_polynomials/analytic_poly.py``, and otherwise nothing.

    The "otherwise nothing" is the point.  Falling back to the modular-addition
    reference for any task not in ``analytic.CASES`` -- which is what this function
    replaced -- silently reported the *wrong* ground truth for every polynomial run:
    ``g_p3_p97`` was printed against ``ipr_ref = 1.000, erank_ref = 148.8`` and looked
    like a total failure to form a periodic representation, when its own reference is
    ``0.062`` and ``51.2`` and the run matches it almost exactly.  A missing reference
    must read as missing, not as a different task's answer.
    """
    if task in analytic.CASES:
        return analytic.CASES[task]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gromov_polynomials"))
        from analytic_poly import DECOMPOSITIONS
    except ImportError:
        return None
    if task in DECOMPOSITIONS:
        g1, g2, h = DECOMPOSITIONS[task]
        return [g1, g2], h
    return None


def _task_of(rec):
    """The short task name, from the record or reconstructed from the run key.

    Summaries written before ``run_poly.py`` recorded ``task`` carry only ``key``, and
    a missing task silently costs the run its analytic reference. The polynomial keys
    are ``<arm>_<task>_p<modulus>``, so the name is recoverable.
    """
    if rec.get("task"):
        return rec["task"]
    parts = str(rec.get("key", "")).split("_")
    return parts[1] if len(parts) == 3 and parts[2].startswith("p") else ""


def analytic_reference(p, width, task="add", seed=0):
    """Fourier IPR and effective rank of the closed-form solution at this width.

    All-NaN when the task has no closed-form solution -- ``mul``, ``mix_quad``,
    ``no_grok`` and every perturbed polynomial.  For the perturbed ones that is not a
    gap in the implementation: ``analytic_check.md`` proves no decomposition exists.
    """
    nan = dict(ipr_u1=np.nan, ipr_w=np.nan, erank_w1=np.nan, weight_norm=np.nan)
    spec = _decomposition(task)
    if spec is None:
        return nan
    inner, outer = spec
    cfg = Config(p=p, width=width, task=task)
    w1, w2 = analytic.build(cfg, inner, outer, seed=seed)
    return dict(
        ipr_u1=_fourier_ipr(w1[:, :p]),
        ipr_w=_fourier_ipr(w2.T),
        erank_w1=_participation(np.linalg.svd(w1, compute_uv=False)),
        weight_norm=float(np.linalg.norm(np.concatenate([w1.ravel(), w2.ravel()]))
                          / np.sqrt(w1.size + w2.size)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results")
    ap.add_argument("--subdir", default="arith")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.results) / args.subdir
    if not root.is_dir():
        root = Path(args.results)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"no summary.json under {root}")
    records = json.loads(summary_path.read_text())

    rows = []
    for rec in records:
        obs_path = root / f"{rec['key']}_obs.csv"
        final = {}
        if obs_path.exists():
            obs = pd.read_csv(obs_path)
            last = obs.iloc[-1]
            final = dict(ipr_u1=last.get("ipr_u1", np.nan),
                         ipr_w=last.get("ipr_w", np.nan),
                         erank_w1=last.get("erank_w1", np.nan))
        ref = analytic_reference(rec["p"], rec["width"], _task_of(rec))
        rows.append(dict(
            key=rec["key"], task=rec.get("task", rec.get("polynomial", "")),
            alpha=rec["fraction"],
            t_mem=rec["t_memorise"], t_grok=rec["t_grok"],
            val_acc=rec["final_val_acc"], val_loss=rec["final_val_loss"],
            wnorm=rec["final_weight_norm"], wnorm_ref=ref["weight_norm"],
            ipr_u1=final.get("ipr_u1", np.nan), ipr_ref=ref["ipr_u1"],
            erank=final.get("erank_w1", np.nan), erank_ref=ref["erank_w1"],
        ))
    df = pd.DataFrame(rows)

    if args.markdown:
        print("| run | task | alpha | memorised | grokked | val acc | val loss | "
              "\\|W\\| | \\|W\\| analytic | IPR | IPR analytic |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in df.itertuples():
            fmt = lambda v: "never" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"
            print(f"| `{r.key}` | {r.task} | {r.alpha:g} | {fmt(r.t_mem)} | {fmt(r.t_grok)} | "
                  f"{r.val_acc:.1%} | {r.val_loss:.2e} | {r.wnorm:.2f} | {r.wnorm_ref:.2f} | "
                  f"{r.ipr_u1:.3f} | {r.ipr_ref:.3f} |")
    else:
        pd.set_option("display.width", 200)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    out = root / "analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

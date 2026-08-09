"""Train registered polynomial runs and write the same three files ``../gromov_arithmetic``
writes, so both papers' logs land in one shape for the dimension analysis.

    python run_poly.py pairs97 --outdir ./results
    python run_poly.py faithful97 --outdir ./results --force
    python run_poly.py g_p1_p97 --set max_steps=20000

``<key>_train.csv`` carries exactly step, train_loss, val_loss, train_acc, val_acc,
weight_norm.  Rows are flushed as they are produced so a reclaimed VM leaves a
truncated but usable trajectory rather than nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import polynomials as P
import runs_poly as registry
from _core import TRAIN_COLUMNS, grok_summary, train


def _parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_one(cfg, outdir: Path, force=False):
    train_csv = outdir / f"{cfg.key}_train.csv"
    if train_csv.exists() and not force:
        print(f"[{cfg.key}] exists, skipping (use --force to overwrite)", flush=True)
        return None

    print(f"\n=== {cfg.key} ===\n{cfg.summary()}\n{cfg.description}", flush=True)
    t0 = time.time()

    handle = train_csv.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS))
    writer.writeheader()

    def on_row(row):
        writer.writerow(row)
        if row["step"] % 1000 == 0:
            handle.flush()

    try:
        train_rows, obs_rows, snapshots = train(
            cfg, P.evaluator(cfg.task, cfg.p), on_row=on_row, verbose=True)
    finally:
        handle.close()

    if obs_rows:
        with (outdir / f"{cfg.key}_obs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(obs_rows[0]))
            w.writeheader()
            w.writerows(obs_rows)
    if snapshots:
        np.savez_compressed(outdir / f"{cfg.key}_snapshots.npz", **snapshots)

    summary = grok_summary(train_rows)
    image = P.distinct_outputs(cfg.task, cfg.p)
    # `task` as well as `polynomial`: ../gromov_arithmetic/analyze.py keys the analytic
    # reference off the short name, and without it every polynomial run is reported
    # against no reference at all.
    summary.update(key=cfg.key, task=cfg.task, polynomial=P.EXPRESSIONS[cfg.task], p=cfg.p,
                   learnable=P.is_learnable(cfg.task),
                   paper_test_acc=P.PAPER_TEST_ACC[(cfg.p, cfg.task)],
                   width=cfg.width, fraction=cfg.fraction, optimizer=cfg.optimizer,
                   lr=cfg.lr, weight_decay=cfg.weight_decay,
                   n_distinct=image["n_distinct"], majority_share=image["majority_share"],
                   seconds=round(time.time() - t0, 1))
    print(f"[{cfg.key}] grokked at {summary['t_grok']}, final val acc "
          f"{summary['final_val_acc']:.2%} (paper {summary['paper_test_acc']:.2%}), "
          f"{summary['seconds']:.0f}s", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+",
                    help="run keys or groups (faithful97, nowd97, pairs97, ...)")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    overrides = _parse_overrides(args.overrides)
    if args.device:
        overrides["device"] = args.device

    keys = registry.expand(args.keys)
    print(f"{len(keys)} run(s): {', '.join(keys)}", flush=True)

    for key in keys:
        cfg = registry.get(key)
        if overrides:
            cfg = cfg.with_overrides(overrides)
        s = run_one(cfg, outdir, force=args.force)
        if s is not None:
            path = outdir / "summary.json"
            existing = json.loads(path.read_text()) if path.exists() else []
            existing = [r for r in existing if r["key"] != s["key"]] + [s]
            path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    sys.exit(main())

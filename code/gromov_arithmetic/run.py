"""Train registered runs and write the logs the dimension analysis consumes.

    python run.py a_add x_no_grok --outdir ./results
    python run.py core --outdir ./results --force
    python run.py a_add --set lr=1e4 --set max_steps=20000

Three files per run:

``<key>_train.csv``      step, train_loss, val_loss, train_acc, val_acc, weight_norm
                         -- byte-compatible with ``dimension_recovery/results/extended``
``<key>_obs.csv``        the spectral probes at the coarser ``obs_every`` stride
``<key>_snapshots.npz``  log-spaced float32 weight dumps
``summary.json``         one record per run: when it memorised, when (or whether) it grokked

Rows are flushed as they are produced.  A Colab VM can be reclaimed mid-run, and a
partial CSV that stops at step 60 000 is still a usable trajectory; a buffered one
that is lost entirely is not.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import runs as registry
import tasks
from gromov import TRAIN_COLUMNS, grok_summary, train


def _parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_one(cfg, outdir: Path, force=False, verbose=True):
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
        train_rows, obs_rows, snapshots = train(cfg, tasks.get(cfg.task),
                                                on_row=on_row, verbose=verbose)
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
    summary.update(key=cfg.key, task=cfg.task, p=cfg.p, width=cfg.width,
                   fraction=cfg.fraction, optimizer=cfg.optimizer, lr=cfg.lr,
                   weight_decay=cfg.weight_decay, seconds=round(time.time() - t0, 1))
    print(f"[{cfg.key}] memorised at {summary['t_memorise']}, "
          f"grokked at {summary['t_grok']}, final val acc "
          f"{summary['final_val_acc']:.2%}, {summary['seconds']:.0f}s", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+", help="run keys or group names (grok, nogrok, core, ...)")
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

    summaries = []
    for key in keys:
        cfg = registry.get(key)
        if overrides:
            cfg = cfg.with_overrides(overrides)
        s = run_one(cfg, outdir, force=args.force)
        if s is not None:
            summaries.append(s)
            # Rewritten after every run: a reclaimed VM then still leaves a valid file.
            path = outdir / "summary.json"
            existing = json.loads(path.read_text()) if path.exists() else []
            existing = [r for r in existing if r["key"] != s["key"]] + [s]
            path.write_text(json.dumps(existing, indent=2))

    if summaries:
        print(f"\n{'key':<16}{'memorise':>10}{'grok':>10}{'val acc':>10}{'val loss':>12}")
        for s in summaries:
            g = "never" if s["t_grok"] is None else str(s["t_grok"])
            m = "never" if s["t_memorise"] is None else str(s["t_memorise"])
            print(f"{s['key']:<16}{m:>10}{g:>10}{s['final_val_acc']:>9.2%}"
                  f"{s['final_val_loss']:>12.3e}")


if __name__ == "__main__":
    sys.exit(main())

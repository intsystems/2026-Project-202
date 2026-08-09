"""Train the matched pairs with the trajectory sketch attached.

Writes ``<key>_rank.npz`` and ``<key>_train.csv`` side by side, under the names
``../active_rank/analyze_rank.py`` globs for, so the measurement runs unchanged:

    python run_rank.py pairs --outdir ./results/rank --steps 30000
    python ../active_rank/analyze_rank.py --indir ./results/rank --window 60 --stride 5

Run keys resolve against both registries -- ``runs.py`` for the arithmetic tasks and
``../gromov_polynomials/runs_poly.py`` for the polynomials -- because a matched pair is
one of each kind in the arithmetic case and two polynomials in the others.

The step budget is deliberately not the 100 000 of the main campaign. ``active_rank``
measures over sliding windows of 60 logged rows at stride 5; at ``log_every = 10`` that
is a 600-step window, the resolution its "fine" analysis used. 30 000 steps at that
stride gives 3 000 rows, matching its run length, and every one of these runs groks
before step 13 000.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_POLY = Path(__file__).resolve().parent.parent / "gromov_polynomials"
if str(_POLY) not in sys.path:
    sys.path.insert(0, str(_POLY))

import runs as arith_registry              # noqa: E402
import runs_poly as poly_registry          # noqa: E402
import tasks                               # noqa: E402
import polynomials as P                    # noqa: E402
from gromov import TRAIN_COLUMNS, grok_summary, train   # noqa: E402
from rank import GromovRankProbe           # noqa: E402

PAIRS = {
    # a generalising run and a control that differs only in the label function
    "pairs": ("a_add", "x_no_grok",
              "g_p1_p97", "g_p1x_p97",
              "g_p2_p97", "g_p2x_p97",
              "g_p3_p97", "g_p3x_p97"),
    "arith": ("a_add", "x_no_grok"),
    "poly": ("g_p1_p97", "g_p1x_p97", "g_p2_p97", "g_p2x_p97", "g_p3_p97", "g_p3x_p97"),
}


def label_fn(cfg):
    """The label function for a config.  Arithmetic tasks are p-independent (the
    modulus is applied by ``build_dataset``); polynomial evaluators close over p."""
    try:
        return tasks.get(cfg.task)
    except KeyError:
        return P.evaluator(cfg.task, cfg.p)


def resolve(key):
    """(config, label function) for a key from either registry."""
    try:
        cfg = arith_registry.get(key)
    except KeyError:
        cfg = poly_registry.get(key)
    return cfg, label_fn(cfg)


def run_one(key, outdir: Path, steps, log_every, force, dim, n_probe, overrides=None):
    cfg, fn = resolve(key)
    cfg = cfg.with_overrides(dict(max_steps=steps, log_every=log_every,
                                  obs_every=log_every * 10, n_snapshots=0))
    if overrides:
        cfg = cfg.with_overrides(overrides)
        # Rebuild the label function against the final config: a polynomial evaluator
        # closes over p, so a `--set p=23` would otherwise keep emitting p=97 labels.
        fn = label_fn(cfg)
    csv_path = outdir / f"{key}_train.csv"
    npz_path = outdir / f"{key}_rank.npz"
    if npz_path.exists() and not force:
        print(f"[{key}] exists, skipping", flush=True)
        return None

    print(f"\n=== {key} ===\n{cfg.summary()}", flush=True)
    t0 = time.time()
    probe = GromovRankProbe(dim=dim, n_probe=n_probe, progress_every=500)

    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(TRAIN_COLUMNS))
    writer.writeheader()

    def on_row(row):
        writer.writerow(row)
        if row["step"] % 2000 == 0:
            handle.flush()

    try:
        train_rows, _, _ = train(cfg, fn, on_row=on_row, verbose=True, observer=probe)
    finally:
        handle.close()

    probe.save(npz_path)
    s = grok_summary(train_rows)
    s.update(key=key, task=cfg.task, p=cfg.p, n_params=probe.n_params,
             rows=len(train_rows), seconds=round(time.time() - t0, 1))
    print(f"[{key}] t_mem={s['t_memorise']} t_gen={s['t_grok']} "
          f"val={s['final_val_acc']:.2%} {s['seconds']:.0f}s "
          f"-> {npz_path.name}", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+", help="run keys or a group (pairs, arith, poly)")
    ap.add_argument("--outdir", default="./results/rank")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override, e.g. --set batch_size=64 --set p=23")
    args = ap.parse_args()

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    keys = []
    for name in args.keys:
        for k in PAIRS.get(name, (name,)):
            if k not in keys:
                keys.append(k)
    print(f"{len(keys)} run(s): {', '.join(keys)}", flush=True)

    summaries = []
    for key in keys:
        s = run_one(key, outdir, args.steps, args.log_every, args.force,
                    args.dim, args.n_probe, overrides)
        if s is not None:
            summaries.append(s)
            (outdir / "rank_runs.json").write_text(json.dumps(summaries, indent=2))

    if summaries:
        print(f"\n{'key':<14}{'t_mem':>9}{'t_gen':>9}{'val acc':>10}")
        for s in summaries:
            g = "never" if s["t_grok"] is None else str(s["t_grok"])
            print(f"{s['key']:<14}{str(s['t_memorise']):>9}{g:>9}{s['final_val_acc']:>9.1%}")


if __name__ == "__main__":
    sys.exit(main())

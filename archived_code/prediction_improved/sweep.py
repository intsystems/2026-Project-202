"""Factorial split x init sweep: what actually determines the grokking delay?

``method.md`` §7 found the canonical 12 000-step delay to be seed-specific -- two
replicate seeds generalized after ~1 740 steps. But ``seed`` moves two things at once:
``grok.tasks`` seeds one RNG stream, draws the train/val split from it, and the model
initialisation continues that same stream. So a seed change alters *which examples are
in the training set* and *what the initial weights are* together, and the replicate
tells us nothing about which one mattered.

``RunConfig.init_seed`` restarts the stream between those two steps. This script crosses
``seed`` (the split) with ``init_seed`` (the initialisation, and the mini-batch order
that follows it) and records the memorization/generalization gap for every cell, so the
variance in the delay can be attributed to one axis or the other.

Weight decay is fixed at 1.0 throughout, so nothing here can be explained by
regularization -- the confound that falsified the function-space signal.

Why it matters beyond the article: if the split dominates, the delay is a property of
*which data you happened to sample*, and it should be predictable from the split alone,
before a single gradient step. The training index sets are saved for exactly that.

    python sweep.py --outdir /content/out/sweep                 # 5 x 5
    python sweep.py --splits 0 1 2 --inits 0 1 2 --outdir out   # smaller
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

EPS = 0.05        # the article's Definition 1 threshold
RUNS = 5          # consecutive logged rows required to call an event sustained


def sustained(steps, values, threshold, runs=RUNS):
    ok = np.asarray(values) >= threshold
    for i in range(len(ok) - runs + 1):
        if ok[i:i + runs].all():
            return int(steps[i])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="values for RunConfig.seed (the train/val split)")
    parser.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="values for RunConfig.init_seed (weights + batch order)")
    parser.add_argument("--base", default="mod_wd1", help="run key to vary")
    parser.add_argument("--outdir", default="sweep")
    parser.add_argument("--keep-logs", action="store_true",
                        help="also write each cell's full training log")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already present in summary.csv. Colab VMs are "
                             "reclaimed without warning, so a long sweep should be "
                             "restartable rather than restarted.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    import runs                                             # deferred: needs torch
    from grok import train

    outdir = Path(args.outdir)
    (outdir / "splits").mkdir(parents=True, exist_ok=True)

    overrides = dict(pair.split("=", 1) for pair in args.overrides)
    base = runs.get(args.base)
    if overrides:
        base = base.with_overrides(overrides)

    summary_path = outdir / "summary.csv"
    rows = []
    done = set()
    if args.resume and summary_path.exists():
        previous = pd.read_csv(summary_path)
        rows = previous.to_dict("records")
        done = {(int(r["split_seed"]), int(r["init_seed"])) for r in rows}
        print(f"resuming: {len(done)} cells already in {summary_path}")

    total = len(args.splits) * len(args.inits)
    started = time.perf_counter()

    print(f"sweep: {len(args.splits)} splits x {len(args.inits)} inits = {total} runs")
    print(f"base : {base.summary()}\n", flush=True)

    for si, split_seed in enumerate(args.splits):
        for ii, init_seed in enumerate(args.inits):
            n = si * len(args.inits) + ii + 1
            if (split_seed, init_seed) in done:
                print(f"[{n}/{total}] split={split_seed} init={init_seed} -- skipped "
                      f"(already done)", flush=True)
                continue
            config = base.with_overrides({"seed": str(split_seed),
                                          "init_seed": str(init_seed)})
            print(f"[{n}/{total}] split={split_seed} init={init_seed}", flush=True)

            df, _ = train(config, outdir=None, progress=False)
            steps = df["step"].to_numpy()
            t_mem = sustained(steps, df["train_acc"].to_numpy(), 1 - EPS)
            t_gen = sustained(steps, df["val_acc"].to_numpy(), 1 - EPS)

            rows.append({
                "split_seed": split_seed,
                "init_seed": init_seed,
                "t_mem": t_mem,
                "t_gen": t_gen,
                "gap": None if (t_mem is None or t_gen is None) else t_gen - t_mem,
                "final_val_acc": float(df["val_acc"].iloc[-20:].median()),
                "final_weight_norm": float(df["weight_norm"].iloc[-1]),
            })
            # Written every cell: an 85-minute job should never lose everything to a
            # timeout at run 24.
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            if args.keep_logs:
                df.to_csv(outdir / f"s{split_seed}_i{init_seed}_train.csv", index=False)

            print(f"          t_mem={t_mem} t_gen={t_gen} "
                  f"gap={rows[-1]['gap']} val_acc={rows[-1]['final_val_acc']:.3f} "
                  f"({time.perf_counter() - started:.0f}s elapsed)", flush=True)

        _save_split(outdir, base, split_seed)

    print(f"\nwrote {summary_path} ({len(rows)} rows) in "
          f"{(time.perf_counter() - started) / 60:.1f} min")
    _report(pd.DataFrame(rows))
    return 0


def _save_split(outdir, base, split_seed):
    """Persist which pairs this split trained on, for the pre-training predictor.

    Rebuilt from the config rather than captured during training so it stays a pure
    function of (task, fraction, seed) -- and so it can be regenerated without a GPU.
    """
    from grok import tasks

    config = base.with_overrides({"seed": str(split_seed)})
    task = tasks.from_config(config, device="cpu")
    np.savez_compressed(
        outdir / "splits" / f"split_{split_seed}.npz",
        train_x=task.X_train.cpu().numpy(), train_y=task.Y_train.cpu().numpy(),
        val_x=task.X_val.cpu().numpy(), val_y=task.Y_val.cpu().numpy(),
        meta=json.dumps({"task": config.task, "p": config.p, "n": config.n,
                         "fraction": config.fraction, "seed": split_seed}),
    )


def _report(frame):
    """Attribute the variance in the gap to the split axis or the init axis."""
    usable = frame.dropna(subset=["gap"])
    print(f"\n{len(usable)}/{len(frame)} cells generalized")
    if usable.empty:
        return

    print(f"gap: min {usable.gap.min():.0f}  median {usable.gap.median():.0f}  "
          f"max {usable.gap.max():.0f}  sd {usable.gap.std():.0f}")

    by_split = usable.groupby("split_seed").gap
    by_init = usable.groupby("init_seed").gap
    print("\nmean gap by split seed:", {k: round(v) for k, v in by_split.mean().items()})
    print("mean gap by init seed :", {k: round(v) for k, v in by_init.mean().items()})

    # Between-group variance over total: how much of the spread each axis explains.
    total = usable.gap.var()
    if total and total > 0:
        for name, grouped in (("split", by_split), ("init", by_init)):
            counts = grouped.count()
            means = grouped.mean()
            between = float((counts * (means - usable.gap.mean()) ** 2).sum()
                            / max(len(usable) - 1, 1))
            print(f"variance explained by {name:<5}: {100 * between / total:5.1f}%")
    print("\n(If split >> init, the delay is a property of which examples were "
          "sampled,\n not of the optimization -- and should be predictable before "
          "training.)")


if __name__ == "__main__":
    raise SystemExit(main())

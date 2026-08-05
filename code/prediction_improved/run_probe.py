"""Train a registered grokking run with function-space probe logging.

Wraps ``grokking_train``'s proven loop -- the configs in ``runs.py`` are the ones that
produced the published logs -- and attaches :class:`probe.LogitProbe`, which is verified
non-invasive by ``verify_noninvasive.py`` (training logs are bit-identical with and
without it).

    python run_probe.py mod_wd1 --outdir /content/out
    python run_probe.py mod_wd0 --outdir /content/out --n-probe 128

Writes three files per run into ``--outdir``:

    <csv_name>                     the standard training log (unchanged format)
    <key>_probe.csv                per-logged-step projections and velocities
    <key>_probe_snapshots.npz      periodic full normalized-logit matrices
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import runs                                                 # noqa: E402
from grok import train                                      # noqa: E402
from probe import LogitProbe                                # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="a key from grokking_train/runs.py, e.g. mod_wd1")
    parser.add_argument("--outdir", default="probe_logs")
    parser.add_argument("--n-probe", type=int, default=256,
                        help="probe examples per source (default: 256)")
    parser.add_argument("--projections", type=int, default=16)
    parser.add_argument("--snapshot-every", type=int, default=20,
                        help="store the full logit matrix every N logged steps")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100,
                        help="heartbeat every N logged rows (0 to silence)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a RunConfig field (repeatable)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    overrides = {}
    for pair in args.overrides:
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    if args.device:
        overrides["device"] = args.device

    config = runs.get(args.run)
    if overrides:
        config = config.with_overrides(overrides)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe = LogitProbe(
        n_probe=args.n_probe,
        n_projections=args.projections,
        seed=args.probe_seed,
        snapshot_every=args.snapshot_every,
        progress_every=args.progress_every,
    )

    print(f"[{args.run}] {config.summary()}", flush=True)
    print(f"    probe: {args.n_probe} inputs x {args.projections} projections "
          f"from {', '.join(probe.sources)}; snapshot every {args.snapshot_every} rows",
          flush=True)

    started = time.perf_counter()
    df, path = train(config, outdir=outdir, progress=False, overwrite=args.force,
                     observer=probe)

    probe_csv = outdir / f"{args.run}_probe.csv"
    probe.save(probe_csv, npz_path=outdir / f"{args.run}_probe_snapshots.npz")
    rows = len(probe.to_frame())

    grokked = df[df["val_acc"] >= 0.95]["step"]
    print(f"\n    training log : {path}")
    print(f"    probe log    : {probe_csv} ({rows} rows)")
    print(f"    grokking     : "
          f"{'NOT REACHED' if len(grokked) == 0 else f'step {grokked.iloc[0]:.0f}'}")
    print(f"    total time   : {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

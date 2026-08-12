"""Train a grokking run and write the CSV that ``../grokking_analysis`` reads.

    python train.py --list                     # registered runs
    python train.py s5_wd1                     # one run -> ./grokking_logs/
    python train.py mod_wd1 mod_wd0            # several, in sequence
    python train.py --article                  # every run the paper's figures use
    python train.py sn --set n=6 --set max_steps=200000    # grok S_6
    python train.py s5_wd1 --dry-run           # resolve the config, train nothing

Overrides use the field names of ``grok.config.RunConfig``; ``--set columns=a,b,c``
and ``--set batch_size=none`` work as expected.  Output goes to ``./grokking_logs``
unless ``--outdir`` says otherwise -- point it at
``../grokking_analysis/grokking_logs`` to feed the figure pipeline directly.
"""

import argparse
import sys
from pathlib import Path

import runs


def _parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def run(key, outdir=None, overrides=None, progress=True, overwrite=False, dry_run=False):
    """Train one registered run, with optional ``RunConfig`` field overrides."""
    config = runs.get(key)
    if overrides:
        config = config.with_overrides(overrides)

    if dry_run:
        target = "(in memory)" if outdir is None else Path(outdir) / config.csv_name
        print(f"[{config.key}] {config.description}")
        print(f"    {config.summary()}")
        print(f"    columns  : {', '.join(config.columns)}")
        print(f"    rows     : {config.expected_rows}")
        print(f"    would write: {target}")
        return None, None

    from grok import train                      # deferred so --dry-run needs no torch

    return train(config, outdir=outdir, progress=progress, overwrite=overwrite)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", help="run keys to train")
    parser.add_argument("--list", action="store_true", help="list the registered runs and exit")
    parser.add_argument("--article", action="store_true",
                        help="train every run the article's figures are built from")
    parser.add_argument("--outdir", default=None,
                        help="output directory (default: ./grokking_logs)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a RunConfig field (repeatable)")
    parser.add_argument("--device", default=None, help="shorthand for --set device=...")
    parser.add_argument("--force", action="store_true", help="overwrite an existing CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configuration without training")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress bar")
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(k) for k in runs.RUNS)
        for key, config in runs.RUNS.items():
            mark = "*" if key in runs.ARTICLE_RUNS else " "
            print(f"{mark} {key:<{width}}  {config.description}")
            print(f"  {'':<{width}}  -> {config.csv_name}")
        print("\n* = a log the article's figures are built from")
        return 0

    keys = list(runs.ARTICLE_RUNS) if args.article else args.runs
    if not keys:
        parser.error("name at least one run, or pass --article / --list")
    unknown = [k for k in keys if k not in runs.RUNS]
    if unknown:
        parser.error(f"unknown run(s): {', '.join(unknown)}. Try --list.")

    overrides = _parse_overrides(args.overrides)
    if args.device:
        overrides["device"] = args.device

    outdir = Path(args.outdir) if args.outdir else runs.LOG_DIR
    for key in keys:
        run(key, outdir=outdir, overrides=overrides, progress=not args.quiet,
            overwrite=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

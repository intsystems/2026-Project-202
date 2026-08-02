"""Regenerate the figures of the ICOMP paper from the raw training logs.

    python reproduce_figures.py                 # every figure
    python reproduce_figures.py s5_wd1 mod_wd1  # a subset
    python reproduce_figures.py --list          # what is available
    python reproduce_figures.py --outdir out    # write elsewhere

Nothing here trains a network: the logs in ``grokking_logs/`` are the input.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import experiments
from edm import (
    grokking_step,
    load_logs,
    plot_dimension_vs_accuracy,
    plot_presentation_panels,
    plot_smoothed_accuracy,
    sliding_dimension,
)


def _run_panels(fig, df, outdir, seed, progress):
    trace = sliding_dimension(
        df,
        target_metric=fig.metric,
        method=fig.method,
        tau_selector=fig.tau_selector,
        window_size=fig.window_size,
        step_size=fig.step_size,
        include_last_window=fig.include_last_window,
        seed=seed,
        progress=progress,
    )
    written = plot_presentation_panels(trace, df, outdir, prefix=fig.key)
    return trace, list(written.values())


def _run_diagnostic(fig, df, outdir, seed, progress):
    trace = sliding_dimension(
        df,
        target_metric=fig.metric,
        method=fig.method,
        tau_selector=fig.tau_selector,
        window_size=fig.window_size,
        step_size=fig.step_size,
        include_last_window=fig.include_last_window,
        seed=seed,
        progress=progress,
    )
    suffix = ".pdf" if fig.article_files and fig.article_files[0].endswith(".pdf") else ".png"
    path = plot_dimension_vs_accuracy(trace, df, outdir / f"{fig.key}{suffix}")
    return trace, [path]


def _run_overview(fig, df, outdir, seed, progress):
    path = plot_smoothed_accuracy(df, outdir / f"{fig.key}.png", window=fig.smooth_window)
    return None, [path]


RUNNERS = {"panels": _run_panels, "diagnostic": _run_diagnostic, "overview": _run_overview}


def run_figure(key, outdir=None, seed=0, progress=True):
    """Produce one registered figure; returns ``(trace, [written paths])``."""
    fig = experiments.get(key)
    outdir = experiments.FIGURE_DIR if outdir is None else Path(outdir)

    if not fig.csv_path.exists():
        raise FileNotFoundError(f"missing log file {fig.csv_path}")

    df = load_logs(fig.csv_path, required=fig.required_columns)
    started = time.perf_counter()
    trace, paths = RUNNERS[fig.kind](fig, df, outdir, seed, progress)
    elapsed = time.perf_counter() - started

    grok = grokking_step(df)
    print(f"[{key}] {fig.description}")
    print(f"    log      : {fig.csv} ({len(df)} rows, steps {df['step'].min():.0f}..{df['step'].max():.0f})")
    print(f"    grokking : {'not reached' if grok is None else f'step {grok:.0f}'}")
    if trace is not None:
        valid = trace.dimension[~np.isnan(trace.dimension)]
        print(
            f"    E({fig.metric}) : {len(valid)} windows, "
            f"min={valid.min():.2f} max={valid.max():.2f} last={valid[-1]:.2f}"
        )
    for path in paths:
        print(f"    wrote    : {path}")
    print(f"    took     : {elapsed:.1f}s")
    return trace, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", help="figure keys to build (default: all)")
    parser.add_argument("--list", action="store_true", help="list the registered figures and exit")
    parser.add_argument("--outdir", default=None, help="output directory (default: ./figures)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the estimators' tie-breaking dither")
    parser.add_argument("--quiet", action="store_true", help="suppress per-window progress bars")
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(k) for k in experiments.FIGURES)
        for key, fig in experiments.FIGURES.items():
            article = ", ".join(fig.article_files) or "-"
            print(f"{key:<{width}}  {fig.description}\n{'':<{width}}  -> {article}")
        return 0

    keys = args.figures or list(experiments.FIGURES)
    unknown = [k for k in keys if k not in experiments.FIGURES]
    if unknown:
        parser.error(f"unknown figure(s): {', '.join(unknown)}. Try --list.")

    outdir = experiments.FIGURE_DIR if args.outdir is None else args.outdir
    for key in keys:
        run_figure(key, outdir=outdir, seed=args.seed, progress=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

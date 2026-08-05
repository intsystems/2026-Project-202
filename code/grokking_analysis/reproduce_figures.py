"""Regenerate the figures of the ICOMP paper from the raw training logs.

    python reproduce_figures.py                 # every figure
    python reproduce_figures.py s5_wd1 mod_wd1  # a subset
    python reproduce_figures.py --list          # what is available
    python reproduce_figures.py --outdir out    # write elsewhere
    python reproduce_figures.py --compare mle_mg  # overlay the MacKay-Ghahramani curve

Nothing here trains a network: the logs in ``grokking_logs/`` are the input.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import experiments
from edm import (
    ESTIMATORS,
    grokking_step,
    load_logs,
    plot_dimension_vs_accuracy,
    plot_presentation_panels,
    plot_smoothed_accuracy,
    sliding_dimension,
)


def _traces(fig, df, methods, seed, progress):
    return [
        sliding_dimension(
            df,
            target_metric=fig.metric,
            method=method,
            tau_selector=fig.tau_selector,
            window_size=fig.window_size,
            step_size=fig.step_size,
            include_last_window=fig.include_last_window,
            seed=seed,
            progress=progress,
        )
        for method in methods
    ]


def _run_panels(fig, df, outdir, methods, seed, progress, annotate_source=True):
    traces = _traces(fig, df, methods, seed, progress)
    written = plot_presentation_panels(traces, df, outdir, prefix=fig.key,
                                       annotate_source=annotate_source)
    return traces, list(written.values())


def _run_diagnostic(fig, df, outdir, methods, seed, progress, annotate_source=True):
    traces = _traces(fig, df, methods, seed, progress)
    suffix = ".pdf" if fig.article_files and fig.article_files[0].endswith(".pdf") else ".png"
    path = plot_dimension_vs_accuracy(traces, df, outdir / f"{fig.key}{suffix}",
                                      annotate_source=annotate_source)
    return traces, [path]


def _run_overview(fig, df, outdir, methods, seed, progress, annotate_source=True):
    path = plot_smoothed_accuracy(df, outdir / f"{fig.key}.png", window=fig.smooth_window)
    return [], [path]


RUNNERS = {"panels": _run_panels, "diagnostic": _run_diagnostic, "overview": _run_overview}


def run_figure(key, outdir=None, compare=(), seed=0, progress=True, annotate_source=True):
    """Produce one registered figure; returns ``(traces, [written paths])``.

    ``compare`` names extra estimators to overlay on the same axes (e.g.
    ``("mle_mg",)`` for the MacKay-Ghahramani correction).
    """
    fig = experiments.get(key)
    outdir = experiments.FIGURE_DIR if outdir is None else Path(outdir)
    methods = [fig.method, *compare]

    if not fig.csv_path.exists():
        raise FileNotFoundError(f"missing log file {fig.csv_path}")

    df = load_logs(fig.csv_path, required=fig.required_columns)
    started = time.perf_counter()
    traces, paths = RUNNERS[fig.kind](fig, df, outdir, methods, seed, progress, annotate_source)
    elapsed = time.perf_counter() - started

    grok = grokking_step(df)
    print(f"[{key}] {fig.description}")
    print(f"    log      : {fig.csv} ({len(df)} rows, steps {df['step'].min():.0f}..{df['step'].max():.0f})")
    print(f"    grokking : {'not reached' if grok is None else f'step {grok:.0f}'}")
    for trace in traces:
        valid = trace.dimension[~np.isnan(trace.dimension)]
        print(
            f"    E({fig.metric}, {trace.label}) : {len(valid)} windows, "
            f"min={valid.min():.2f} max={valid.max():.2f} last={valid[-1]:.2f}"
        )
    for path in paths:
        print(f"    wrote    : {path}")
    print(f"    took     : {elapsed:.1f}s")
    return traces, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", help="figure keys to build (default: all)")
    parser.add_argument("--list", action="store_true", help="list the registered figures and exit")
    parser.add_argument("--outdir", default=None, help="output directory (default: ./figures)")
    parser.add_argument(
        "--compare", nargs="+", default=(), metavar="METHOD", choices=sorted(ESTIMATORS),
        help="overlay extra estimators on the same axes (e.g. --compare mle_mg); "
             "output defaults to ./figures/comparison so the article-faithful set is kept",
    )
    parser.add_argument(
        "--article-exact", action="store_true",
        help="omit the 'estimated from ...' annotation, reproducing the published "
             "figures byte for byte")
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

    if args.outdir is not None:
        outdir = Path(args.outdir)
    elif args.compare:
        outdir = experiments.FIGURE_DIR / "comparison"
    else:
        outdir = experiments.FIGURE_DIR

    for key in keys:
        if args.compare and experiments.get(key).kind == "overview":
            print(f"[{key}] no dimensionality curve to compare -- skipped")
            continue
        run_figure(key, outdir=outdir, compare=args.compare, seed=args.seed,
                   progress=not args.quiet, annotate_source=not args.article_exact)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Local trajectory dimension: how many directions the optimizer explores in ~100 steps.

This asks a different question from `reproduce_figures.py` and `best_practice.py`.
Those estimate the dimension of the *attractor* the trajectory fills, which needs
the trajectory to return near its past states -- and the audit showed these logs
never do, at any window length.

Here the question is local: over a short run of consecutive iterations, how many
independent directions does the trajectory actually use? That is well posed on a
transient, needs no recurrence, and is answered by the singular spectrum of the
delay-embedded segment (Broomhead and King, 1986) rather than by nearest
neighbours. The summary used is the participation ratio ``1 / sum p_i^2`` of the
normalised squared singular values.

Two consequences of switching estimator:

* **The window can finally be short.** A 100-200 iteration segment is 10-40 logged
  samples -- far too few for any neighbour-based estimate, but ample for an SVD.
* **A redundant embedding is now correct.** ``tau = 1`` with ``E = W/2`` is the
  right choice, the opposite of what Levina-Bickel wants, because the SVD sorts
  the redundant directions out instead of being fooled by them.

The reference value is exact rather than empirical: a locally straight trajectory
gives **PR = 1**, a planar circular arc 2, and noise the full embedding dimension.
So "PR = 1" means "the optimizer moved along a line", with no calibration needed.

Every figure also carries the null model this has to beat -- :func:`edm.local_roughness`,
the residual of a plain linear fit, no embedding and no SVD -- because on these logs
it tracks the participation ratio closely and fires *earlier*. Each figure has three
panels: the two statistics against training step (their zeros aligned, so one dotted
line means "locally straight" for both), the train/val accuracies, and a rank-rank
scatter showing the Spearman correlation between them for what it is.

    python local_dimension.py                 # all runs -> figures/local_dimension/
    python local_dimension.py --iters 100     # window in optimization steps
    python local_dimension.py s5_wd1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import experiments
from best_practice import RUNS, BY_KEY
from edm import grokking_step, load_logs, local_roughness, sliding_dimension
from edm.plots import COLOR_TRAIN, COLOR_VAL, PAPER_STYLE, _save, source_label

OUT_DIR = experiments.FIGURE_DIR / "local_dimension"

WINDOW_ITERS = 200        # length of the trajectory segment, in optimization steps
PR_THRESHOLD = 1.10       # "more than a straight line"
PERSIST_ITERS = 500       # ignore transients shorter than this
STRAIGHT_LINE = 1.0       # exact PR of a locally linear trajectory

COLOR_PR = "#5b2c8d"        # singular spectrum
COLOR_BASELINE = "#0f7b6c"  # the null model it is compared against


def steps_per_sample(df):
    return int(np.median(np.diff(df["step"].to_numpy())))


def sustained_rise(steps, values, threshold=PR_THRESHOLD,
                   persist_iters=PERSIST_ITERS, sps=1):
    """First step of the earliest stretch that stays above ``threshold`` long enough.

    Causal, and deliberately insensitive to the initialisation transient: a single
    spike at step ~200 does not count, only a sustained departure from PR = 1 does.
    """
    need = max(1, persist_iters // sps)
    run = 0
    for i, value in enumerate(values):
        if np.isfinite(value) and value > threshold:
            run += 1
            if run >= need:
                return float(steps[i - need + 1])
        else:
            run = 0
    return None


def analyse(run, window_iters=WINDOW_ITERS):
    df = load_logs(run.csv_path)
    sps = steps_per_sample(df)
    window_size = max(6, window_iters // sps)

    trace = sliding_dimension(
        df, target_metric=run.metric, method="svd",
        window_size=window_size, step_size=1, label_position="right",
        clip=None, seed=0, progress=False, estimator_kwargs=dict(degenerate=np.nan),
    )
    fire = sustained_rise(trace.steps, trace.dimension, sps=sps)

    # Null model: does a linear detrend do the same job without the SVD?
    series = df[run.metric].to_numpy(dtype=np.float64)
    baseline = np.array([local_roughness(series[i:i + window_size])
                         for i in range(len(series) - window_size + 1)])
    baseline_fire = sustained_rise(trace.steps, baseline, threshold=0.10, sps=sps)
    ok = np.isfinite(trace.dimension) & np.isfinite(baseline)
    order = lambda v: np.argsort(np.argsort(v))          # Spearman via rank correlation
    rho = (float(np.corrcoef(order(trace.dimension[ok]), order(baseline[ok]))[0, 1])
           if ok.sum() > 2 else np.nan)

    return dict(df=df, trace=trace, fire=fire, sps=sps, window_size=window_size,
                window_iters=window_size * sps, baseline=baseline,
                baseline_fire=baseline_fire, baseline_rho=rho)


def _fire_marker(ax, step, colour, label):
    ax.axvline(step, color=colour, linestyle="--", linewidth=2, alpha=0.9)
    return matplotlib.lines.Line2D([], [], color=colour, linestyle="--", linewidth=2,
                                   label=label)


def _verdict(fire, grok, name):
    if fire is None:
        return None, f"{name}: silent" + (" (correct)" if grok is None else "  MISS")
    if grok is None:
        return fire, f"{name}: fires {fire:.0f}  FALSE POSITIVE"
    return fire, f"{name}: fires {fire:.0f}  ({grok - fire:+.0f} vs. grokking)"


def plot_run(run, result, outdir=OUT_DIR):
    """Three panels: the two statistics, the accuracies, and the rank correlation.

    The singular spectrum and its null model are given equal billing on purpose --
    on these logs the null model detects earlier, and a figure that showed only the
    participation ratio would hide that.
    """
    df, trace = result["df"], result["trace"]
    baseline, rho = result["baseline"], result["baseline_rho"]
    grok = grokking_step(df)
    pr = trace.dimension
    finite_pr = pr[np.isfinite(pr)]
    finite_bl = baseline[np.isfinite(baseline)]

    with matplotlib.rc_context({**PAPER_STYLE, "font.size": 12}):
        fig = plt.figure(figsize=(12.4, 6.2))
        grid = fig.add_gridspec(2, 2, width_ratios=[4.1, 1.0], height_ratios=[3.0, 1.15],
                                hspace=0.12, wspace=0.28)
        ax = fig.add_subplot(grid[0, 0])
        ax_acc = fig.add_subplot(grid[1, 0], sharex=ax)
        ax_rank = fig.add_subplot(grid[:, 1])

        # --- the two statistics, with their "perfectly straight" baselines aligned ---
        pr_line, = ax.plot(trace.steps, pr, color=COLOR_PR, linewidth=1.5,
                           label="local SVD participation ratio")
        ax_b = ax.twinx()
        bl_line, = ax_b.plot(trace.steps, baseline, color=COLOR_BASELINE, linewidth=1.5,
                             alpha=0.85, label="departure from local linearity (null model)")

        # 35% headroom so the legend sits above the curves instead of over them.
        # Scaling both tops by the same factor keeps PR = 1 aligned with residual = 0.
        pr_top = 1.0 + 1.35 * (max(finite_pr.max(), PR_THRESHOLD * 1.4) - 1.0)
        bl_top = 1.35 * max(finite_bl.max(), 0.05)
        margin = 0.06
        ax.set_ylim(STRAIGHT_LINE - margin * (pr_top - STRAIGHT_LINE), pr_top)
        ax_b.set_ylim(-margin * bl_top, bl_top)      # PR = 1 and roughness = 0 now coincide
        ax.axhline(STRAIGHT_LINE, color="0.35", linestyle=(0, (1, 3)), linewidth=1.5)

        ax.set_ylabel("participation ratio", color=COLOR_PR, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=COLOR_PR)
        ax_b.set_ylabel("residual after a linear fit", color=COLOR_BASELINE,
                        fontsize=11, fontweight="bold")
        ax_b.tick_params(axis="y", labelcolor=COLOR_BASELINE)
        ax_b.grid(False)
        ax.set_zorder(ax_b.get_zorder() + 1)
        ax.patch.set_visible(False)

        markers = [matplotlib.lines.Line2D(
            [], [], color="0.35", linestyle=(0, (1, 3)), linewidth=1.5,
            label="a locally straight trajectory (PR = 1, residual = 0)")]
        for axis, fire, colour, name in (
            (ax, result["fire"], COLOR_PR, "SVD"),
            (ax, result["baseline_fire"], COLOR_BASELINE, "null model"),
        ):
            step, label = _verdict(fire, grok, name)
            if step is None:
                markers.append(matplotlib.patches.Patch(facecolor="none", edgecolor="none",
                                                        label=label))
            else:
                markers.append(_fire_marker(axis, step, colour, label))

        for axis in (ax, ax_acc):
            if grok is not None:
                axis.axvline(grok, color="black", linestyle=":", linewidth=2)
        if grok is not None:
            ax.annotate("grokking", xy=(grok, 0.62), xycoords=("data", "axes fraction"),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=11, fontweight="bold", va="center")

        ax.legend(handles=[pr_line, bl_line, *markers], loc="upper right",
                  framealpha=0.95, fontsize=9)
        ax.tick_params(labelbottom=False)

        # --- accuracies, on their own panel so three y-axes are not needed ---
        ax_acc.plot(df["step"], df["train_acc"], color=COLOR_TRAIN, alpha=0.75,
                    linewidth=1.6, label="Train Acc")
        ax_acc.plot(df["step"], df["val_acc"], color=COLOR_VAL, linewidth=1.8, label="Val Acc")
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.set_ylabel("Accuracy", fontsize=11)
        ax_acc.set_xlabel("Optimization steps (right edge of the segment -- causal)",
                          fontsize=12, fontweight="bold")
        ax_acc.set_xlim(df["step"].min(), df["step"].max())
        ax_acc.legend(loc="center right", framealpha=0.9, fontsize=9)

        # --- Spearman shown as what it is: a correlation of ranks ---
        ok = np.isfinite(pr) & np.isfinite(baseline)
        rank = lambda v: np.argsort(np.argsort(v)) / max(1, len(v) - 1)
        ax_rank.scatter(rank(baseline[ok]), rank(pr[ok]), s=4, alpha=0.35,
                        color=COLOR_PR, edgecolors="none")
        ax_rank.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1.2)
        ax_rank.set_xlabel("rank, null model", fontsize=10)
        ax_rank.set_ylabel("rank, participation ratio", fontsize=10)
        ax_rank.set_title(rf"Spearman $\rho$ = {rho:+.2f}", fontsize=11, fontweight="bold")
        ax_rank.set_xlim(-0.02, 1.02)
        ax_rank.set_ylim(-0.02, 1.02)
        ax_rank.set_aspect("equal")

        fig.suptitle(
            f"{run.title}\n"
            f"local structure of the 1D series {source_label(run.metric)} over "
            f"{result['window_iters']} iterations ({result['window_size']} samples, "
            f"embedding {result['window_size'] // 2}, " + r"$\tau$=1)",
            fontsize=12.5, y=0.985,
        )
        fig.subplots_adjust(top=0.86)
        return _save(fig, Path(outdir) / f"{run.key}.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run keys to build (default: all)")
    parser.add_argument("--iters", type=int, default=WINDOW_ITERS,
                        help="segment length in optimization steps (default 200)")
    parser.add_argument("--outdir", default=OUT_DIR)
    args = parser.parse_args(argv)

    unknown = [key for key in args.runs if key not in BY_KEY]
    if unknown:
        parser.error(f"unknown run(s): {', '.join(unknown)}. Available: {', '.join(BY_KEY)}")

    for key in (args.runs or [run.key for run in RUNS]):
        run = BY_KEY[key]
        result = analyse(run, window_iters=args.iters)
        path = plot_run(run, result, outdir=args.outdir)
        grok = grokking_step(result["df"])
        values = result["trace"].dimension[np.isfinite(result["trace"].dimension)]
        fire = result["fire"]
        if fire is None:
            verdict = "silent" + (" (correct: no grokking)" if grok is None else "  MISS")
        elif grok is None:
            verdict = f"fires at {fire:.0f}  (FALSE POSITIVE)"
        else:
            verdict = f"fires at {fire:.0f}, lead {grok - fire:+.0f}"
        print(f"[{key}] {run.metric}: {result['window_iters']} iters "
              f"({result['window_size']} samples), PR {values.min():.2f}..{values.max():.2f}, "
              f"{np.mean(values > PR_THRESHOLD) * 100:3.0f}% above {PR_THRESHOLD}  ->  {verdict}")
        bfire, rho = result["baseline_fire"], result["baseline_rho"]
        bverdict = "silent" if bfire is None else (
            f"fires at {bfire:.0f}" + (f", lead {grok - bfire:+.0f}" if grok is not None else ""))
        print(f"    null model (linear detrend, no SVD): rho={rho:+.2f} with PR, {bverdict}")
        print(f"    wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

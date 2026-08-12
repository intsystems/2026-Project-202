"""Best-practice re-analysis: Levina-Bickel vs. MacKay-Ghahramani vs. MG + Theiler.

The audit established that no window length available in these runs makes the
*absolute* dimension identifiable (see `identifiability_ratio` in the README).
Paying for identifiability with a wider window therefore buys nothing and costs
the only thing that matters -- temporal localization. So this module optimises for
**detection**, not for measurement:

* the window is the *smallest* one that still supports the Theiler exclusion,
  which keeps the estimate local in training time;
* everything that is free is fixed -- MacKay-Ghahramani pooling, exact numerics,
  a delay from the delayed mutual information, causal labelling;
* the absolute level is never quoted; what is read off is the *change*.

===========================  =========================  ==================================
knob                         paper / `reproduce_figures`  here
===========================  =========================  ==================================
delay ``tau``                1 (redundant embedding)    first minimum of the delayed
                                                        mutual information
window ``W``                 300 samples                smallest W leaving >= 5k candidate
                                                        neighbours after Theiler exclusion
neighbours ``k``             5                          10
label position               window centre (acausal)    right edge (causal)
tie-breaking                 +N(0, 1e-9) dither         none; coincident points dropped
``sum log(r_k/r_j)`` floor   clamped at 1e-5            none
reported ``E``               clipped to [1, 30]         raw
variance-free window         fabricated ``E = 1``       ``NaN`` (absent from the plot)
control runs                 analysed separately        share the treatment's tau and W
===========================  =========================  ==================================

Each figure carries three diagnostics, because tuning cannot manufacture
information that is not in the data:

* **line constants** -- what the estimator returns for a perfectly straight
  trajectory at this ``k`` and Theiler window. A curve resting on its constant is
  tracking the smoothness of the series, not an attractor.
* **identifiability shading** -- windows where ``E(2*max_E) / E(max_E) > 1.25``,
  i.e. where the absolute level tracks the embedding space rather than the data.
* **causal detector** -- the first step at which the statistic has fallen 25 %
  below the running peak *it had seen by that step*, and its lead over grokking.

    python best_practice.py            # all experiments -> figures/best_practice/
    python best_practice.py s5_wd1     # one
    python best_practice.py --calibrate  # Lorenz-63 control, to show the pipeline works
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import experiments
from edm import (
    delayed_mutual_information,
    first_local_minimum,
    grokking_step,
    load_logs,
    mle_intrinsic_dimension,
    sliding_dimension,
)
from edm.plots import (
    E_SERIES_MARKERS, E_SERIES_STYLES, PAPER_STYLE, _save, add_accuracy_axis, source_label,
)

OUT_DIR = experiments.FIGURE_DIR / "best_practice"

K_NEIGHBOURS = 10
MAX_E = 10
CANDIDATE_SAFETY = 5      # require >= 5k candidates to survive the Theiler exclusion
MIN_WINDOW = 150
ESTIMATOR_KWARGS = dict(
    k_neighbors=K_NEIGHBOURS,
    max_E=MAX_E,
    dither=None,          # exact: no noise, no distance floor, no log-ratio clamp
    degenerate=np.nan,    # do not invent E = 1 for a variance-free window
    clamp_to_max_E=False,
)
METHODS = ("mle", "mle_mg", "mle_mg_theiler")
IDENTIFIABILITY_LIMIT = 1.25
DETECTOR_DROP = 0.25


@dataclass(frozen=True)
class Run:
    """One log + observable. Controls borrow their settings from the treatment run."""

    key: str
    csv: str
    metric: str
    title: str
    reference: str = ""   # key whose tau/W to adopt; "" means self

    @property
    def csv_path(self):
        return experiments.LOG_DIR / self.csv


RUNS = (
    Run("mod_wd1", experiments.MOD_WD1_CSV, "weight_norm",
        r"Modular addition, WD=1.0 -- $\|w\|_2$"),
    Run("mod_wd0", experiments.MOD_WD0_CSV, "weight_norm",
        r"Modular addition, WD=0.0 (control) -- $\|w\|_2$", reference="mod_wd1"),
    Run("s5_wd1", experiments.S5_WD1_CSV, "weight_norm",
        r"$S_5$ composition, WD=0.2 -- $\|w\|_2$"),
    Run("s5_wd0", experiments.S5_WD0_CSV, "weight_norm",
        r"$S_5$ composition, WD=0.0 (control) -- $\|w\|_2$", reference="s5_wd1"),
    Run("s5_wd1_val_loss", experiments.S5_WD1_CSV, "val_loss",
        r"$S_5$ composition, WD=0.2 -- $\mathcal{L}_{val}$"),
    Run("s5_wd0_val_loss", experiments.S5_WD0_CSV, "val_loss",
        r"$S_5$ composition, WD=0.0 (control) -- $\mathcal{L}_{val}$",
        reference="s5_wd1_val_loss"),
    Run("grokking_dimension", experiments.FULL_BATCH_CSV, "train_loss",
        r"Full-batch GD baseline -- $\mathcal{L}_{train}$"),
)
BY_KEY = {run.key: run for run in RUNS}


def select_tau(series, max_tau=40, bins=32):
    """First minimum of the delayed mutual information, over the whole series."""
    taus, dmi = delayed_mutual_information(series, max_tau=max_tau, bins=bins)
    return int(taus[first_local_minimum(dmi, abs_eps=0.0, drop_fraction=0.02)])


def window_for(tau, k=K_NEIGHBOURS, max_E=MAX_E, safety=CANDIDATE_SAFETY, floor=MIN_WINDOW):
    """Smallest window leaving ``safety * k`` candidates after the Theiler exclusion.

    Embedding costs ``(max_E-1)*tau`` samples and the exclusion removes ``2W_th+1``
    candidates per point, so anything smaller cannot support the Theiler estimate
    at all -- and anything larger only blurs the estimate in training time.
    """
    theiler = (max_E - 1) * tau
    needed = theiler + (2 * theiler + 1) + safety * k
    return max(floor, int(np.ceil(needed / 50.0) * 50))


def line_constant(k=K_NEIGHBOURS, theiler_window=0):
    """What the estimator returns for a perfectly straight, uniformly sampled path.

    The surviving neighbours then sit at ``|dt| = W+1 ...`` in both directions, so
    ``r_j`` is a known integer sequence and the estimate depends on nothing but
    ``k`` and ``W``.
    """
    offsets = np.repeat(np.arange(theiler_window + 1, theiler_window + k + 1), 2)
    r = np.sort(offsets)[:k]
    return (k - 1) / np.sum(np.log(r[-1] / r[:-1]))


def causal_detector(steps, values, rel_drop=DETECTOR_DROP):
    """First step where the statistic sits ``rel_drop`` below its running peak.

    Strictly causal: at every step it only uses estimates already available, and
    each estimate only uses samples up to its own (right-edge) label.
    """
    peak = -np.inf
    for step, value in zip(steps, values):
        if not np.isfinite(value):
            continue
        peak = max(peak, value)
        if value < (1.0 - rel_drop) * peak:
            return float(step)
    return None


def settings_for(run):
    """The (tau, W) a run analyses at -- inherited from its treatment run if it is a control."""
    source = BY_KEY[run.reference] if run.reference else run
    series = load_logs(source.csv_path)[source.metric].to_numpy(dtype=np.float64)
    tau = select_tau(series)
    return tau, window_for(tau)


def analyse(run, progress=False):
    """Three traces, a per-window identifiability ratio, and the detector verdicts."""
    df = load_logs(run.csv_path)
    tau, window_size = settings_for(run)
    step_size = max(5, window_size // 25)

    common = dict(
        target_metric=run.metric, window_size=window_size, step_size=step_size, tau=tau,
        label_position="right", clip=None, seed=0, progress=progress,
    )
    traces = {
        method: sliding_dimension(df, method=method, estimator_kwargs=ESTIMATOR_KWARGS, **common)
        for method in METHODS
    }
    doubled = sliding_dimension(
        df, method="mle_mg_theiler",
        estimator_kwargs={**ESTIMATOR_KWARGS, "max_E": 2 * MAX_E}, **common,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = doubled.dimension / traces["mle_mg_theiler"].dimension

    fires = {m: causal_detector(t.steps, t.dimension) for m, t in traces.items()}
    return dict(df=df, traces=traces, ratio=ratio, tau=tau, fires=fires,
                window_size=window_size, step_size=step_size)


def _shade_unidentifiable(ax, steps, ratio, limit=IDENTIFIABILITY_LIMIT):
    bad = np.asarray(ratio > limit)
    if not bad.any():
        return False
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        ax.axvspan(steps[start], steps[min(stop, len(steps) - 1)],
                   color="0.5", alpha=0.12, lw=0, zorder=0)
    return True


def plot_run(run, result, outdir=OUT_DIR):
    df, traces, ratio = result["df"], result["traces"], result["ratio"]
    tau, window_size = result["tau"], result["window_size"]
    grok = grokking_step(df)
    theiler = (MAX_E - 1) * tau
    const_plain, const_theiler = line_constant(), line_constant(theiler_window=theiler)
    span = (window_size - 1) * float(np.median(np.diff(df["step"].to_numpy())))

    with matplotlib.rc_context({**PAPER_STYLE, "font.size": 13}):
        fig, ax = plt.subplots(figsize=(10, 5.6))
        shaded = _shade_unidentifiable(ax, traces["mle"].steps, ratio)

        lines = []
        for method, style, marker in zip(METHODS, E_SERIES_STYLES, E_SERIES_MARKERS):
            trace = traces[method]
            line, = ax.plot(trace.steps, trace.dimension, linewidth=2.0, marker=marker,
                            markersize=3, markevery=3, label=f"$E$ ({trace.label})", **style)
            lines.append(line)

        # Robust limits: with the clamps removed Levina-Bickel can spike to O(100)
        # on a single near-degenerate window, which would flatten everything else.
        # Log scale: the three estimators differ by ~10x, so a linear axis would
        # flatten two of them. Robust limits because Levina-Bickel, with the clamps
        # removed, can spike to O(100) on a single near-degenerate window.
        finite = np.concatenate([t.dimension[np.isfinite(t.dimension)] for t in traces.values()])
        low, high = np.percentile(finite, [0.5, 99.5])
        ax.set_yscale("log")
        ax.set_ylim(low / 1.6, high * 1.25)
        off_scale = int(np.sum((finite < ax.get_ylim()[0]) | (finite > ax.get_ylim()[1])))

        constants = []
        for value, colour, name in ((const_plain, E_SERIES_STYLES[0]["color"], "no Theiler"),
                                    (const_theiler, E_SERIES_STYLES[2]["color"], "Theiler")):
            if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]:
                ax.axhline(value, color=colour, linestyle=(0, (1, 3)), linewidth=1.6, alpha=0.8)
            constants.append(matplotlib.lines.Line2D(
                [], [], color=colour, linestyle=(0, (1, 3)), linewidth=1.6,
                label=f"straight-line $E$ = {value:.2f} ({name})"))

        _, acc_lines = add_accuracy_axis(ax, df)

        if grok is not None:
            ax.axvline(grok, color="black", linestyle=":", linewidth=2)
            ax.annotate("grokking", xy=(grok, 0.965), xycoords=("data", "axes fraction"),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=11, fontweight="bold", va="top")

        # Causal detector: MG is the estimator to read, so mark its firing step.
        fire = result["fires"]["mle_mg"]
        markers = []
        if fire is not None:
            colour = E_SERIES_STYLES[1]["color"]
            ax.axvline(fire, color=colour, linestyle="--", linewidth=1.8, alpha=0.9)
            lead = "" if grok is None else f"  ({grok - fire:+.0f} vs. grokking)"
            markers.append(matplotlib.lines.Line2D(
                [], [], color=colour, linestyle="--", linewidth=1.8,
                label=f"MG detector fires at {fire:.0f}{lead}"))
        elif grok is None:
            markers.append(matplotlib.patches.Patch(
                facecolor="none", edgecolor="none", label="MG detector never fires (correct)"))

        ax.set_xlabel("Optimization steps (right edge of the window -- causal)",
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("MLE statistic $\\hat{E}$ (log scale)\nestimated from the 1D series "
                      + source_label(run.metric), fontsize=12, fontweight="bold")
        ax.set_xlim(df["step"].min(), df["step"].max())
        ax.set_title(
            f"{run.title}\n"
            f"dimension estimated from the 1D series  {source_label(run.metric)}\n"
            rf"$\tau$={tau} (DMI), $W$={window_size} samples ({span:.0f} steps), "
            rf"$k$={K_NEIGHBOURS}, $E_{{max}}$={MAX_E}, Theiler={theiler}, exact numerics",
            fontsize=12,
        )

        if off_scale:
            markers.append(matplotlib.patches.Patch(
                facecolor="none", edgecolor="none",
                label=f"{off_scale} estimate(s) off-scale (LB instability)"))

        handles = [*lines, *constants, *markers, *acc_lines]
        if shaded:
            handles.append(matplotlib.patches.Patch(
                facecolor="0.5", alpha=0.12,
                label=f"absolute level not identifiable (ratio > {IDENTIFIABILITY_LIMIT})"))
        ax.legend(handles=handles, loc="best", framealpha=0.92, fontsize=9)
        fig.tight_layout()
        return _save(fig, Path(outdir) / f"{run.key}.png")


def calibrate(n=12000):
    """Same estimators on Lorenz-63, where the answer is known to be ~2.06."""
    from test_edm import _lorenz_x

    series = _lorenz_x(n=n, burn_in=1000)
    tau = select_tau(series)
    rows = []
    for label, method_kwargs in (
        ("Levina-Bickel", dict(correction="levina_bickel", theiler_window=0)),
        ("MacKay-Ghahramani", dict(correction="mackay_ghahramani", theiler_window=0)),
        ("MG + Theiler", dict(correction="mackay_ghahramani", theiler_window="embedding")),
    ):
        kwargs = dict(k_neighbors=K_NEIGHBOURS, dither=None, clamp_to_max_E=False, **method_kwargs)
        low = mle_intrinsic_dimension(series, tau=tau, max_E=MAX_E, **kwargs)
        high = mle_intrinsic_dimension(series, tau=tau, max_E=2 * MAX_E, **kwargs)
        rows.append((label, low, high, high / low))

    print(f"Lorenz-63 calibration ({len(series)} samples, tau={tau}, "
          f"k={K_NEIGHBOURS}, true d ~ 2.06)")
    print(f"  {'estimator':22s} {'E@max_E':>9s} {'E@2max_E':>9s} {'ratio':>7s}")
    for label, low, high, ratio in rows:
        print(f"  {label:22s} {low:9.2f} {high:9.2f} {ratio:7.2f}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run keys to build (default: all)")
    parser.add_argument("--outdir", default=OUT_DIR)
    parser.add_argument("--calibrate", action="store_true",
                        help="run the Lorenz-63 control and exit")
    args = parser.parse_args(argv)

    if args.calibrate:
        calibrate()
        return 0

    unknown = [key for key in args.runs if key not in BY_KEY]
    if unknown:
        parser.error(f"unknown run(s): {', '.join(unknown)}. Available: {', '.join(BY_KEY)}")

    for key in (args.runs or list(BY_KEY)):
        run = BY_KEY[key]
        result = analyse(run)
        path = plot_run(run, result, outdir=args.outdir)
        grok = grokking_step(result["df"])
        steps = result["traces"]["mle"].steps
        bad = np.mean(np.asarray(result["ratio"]) > IDENTIFIABILITY_LIMIT) * 100
        print(f"[{key}] {run.metric}: tau={result['tau']}, W={result['window_size']}, "
              f"{len(steps)} windows from step {steps[0]:.0f}, {bad:.0f}% level not identifiable")
        lb, mg = (result["traces"]["mle"].dimension, result["traces"]["mle_mg"].dimension)
        with np.errstate(invalid="ignore", divide="ignore"):
            blowups = int(np.nansum(lb > 3 * mg))
        if blowups:
            print(f"    !! Levina-Bickel exceeds 3x MacKay-Ghahramani in {blowups} window(s) "
                  f"-- the arithmetic-mean instability the clamps used to hide")
        for method in METHODS:
            trace = result["traces"][method]
            values = trace.dimension[np.isfinite(trace.dimension)]
            fire = result["fires"][method]
            if fire is None:
                verdict = "never fires" + (" (correct: no grokking)" if grok is None else " (MISS)")
            else:
                verdict = (f"fires at {fire:.0f}" +
                           (f", lead {grok - fire:+.0f}" if grok is not None
                            else "  (FALSE POSITIVE)"))
            print(f"    {trace.label:22s} {values.min():5.2f} .. {values.max():5.2f}   {verdict}")
        print(f"    wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

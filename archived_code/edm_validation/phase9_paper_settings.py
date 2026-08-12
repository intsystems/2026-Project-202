"""Falsify the dimension-collapse signal under the paper's own settings (no Theiler window).

Section 3.1 of the report falsifies the published intrinsic-dimension collapse using a
Theiler exclusion. A fair objection is that the exclusion is itself the problem: the
original analysis did not use one, and under its own settings the collapse looked
convincing. This script answers that objection on its own terms. Every estimate here uses
W = 0, k = 5, max_E = 15 and a 300-sample window, which is the published configuration.

Two things are separated that the WD=0 control cannot separate.

**Level.** With W = 0 the k nearest neighbours of an interior point are its temporal
neighbours, so a locally straight window returns the closed-form tangent constant (1.227
for Levina-Bickel at k = 5). The WD=0 controls sit there permanently. That falsifies the
*level* but not the *shape*: a flat control cannot refute a claim about a rise and a fall,
because its weight norm never decays and so its local roughness never changes.

**Shape.** The control that can refute the shape holds weight decay fixed at 1.0 and
reduces the training fraction, so the same regulariser drives a non-stationary norm but the
model never generalises. The direction differs, and that is informative rather than a
defect of the control: the grokking norm falls from 42 to 30 while the lowdata norm rises
from 42 to 61. Local straightness does not care about the sign of the slope, so if the
mechanism is smoothness then both should collapse. If the rise-and-collapse survives in
runs that never generalise, it is not a precursor of generalisation.

**What the statistic actually measures.** At W = 0 the estimate is a deterministic
function of the local shape of the window. We test the strongest form of that claim: that
it is predicted by a one-line roughness ratio, std(diff(x)) / std(x), with no embedding,
no neighbour search and no maximum-likelihood step anywhere in it.

    python phase9_paper_settings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension                      # noqa: E402

RUNS = HERE.parent / "prediction_improved" / "results"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# Weight decay is 1.0 in every run below except wd0. The grokking and lowdata families
# differ only in the training fraction, which is the whole point of the comparison.
CONDITIONS = [
    ("grok", "grok_train.csv", "generalises"),
    ("grok_s1", "grok_seed1_train.csv", "generalises"),
    ("grok_s2", "grok_seed2_train.csv", "generalises"),
    ("lowdata15", "lowdata15_train.csv", "never generalises"),
    ("lowdata20", "lowdata20_train.csv", "never generalises"),
    ("wd0", "wd0_train.csv", "never generalises"),
]

WINDOW, STRIDE, K, MAX_E = 300, 10, 5, 15


def line_constant(k, theiler=0):
    """Closed-form Levina-Bickel estimate for a straight, uniformly sampled trajectory."""
    offsets = np.repeat(np.arange(theiler + 1, theiler + k + 1), 2)
    r = np.sort(offsets)[:k]
    return (k - 1) / np.sum(np.log(r[-1] / r[:-1]))


def roughness(window):
    """Local roughness: the step-to-step variation relative to the window's own spread.

    No embedding, no neighbour search, no likelihood. If this predicts the dimension
    estimate, the estimate carries no information the ratio does not already carry.
    """
    spread = np.std(window)
    return np.std(np.diff(window)) / spread if spread > 0 else np.nan


def trace(series, correction, theiler=0):
    """Sliding-window estimate and roughness, at the published window and stride."""
    centres, dims, roughs = [], [], []
    for start in range(0, len(series) - WINDOW + 1, STRIDE):
        w = series[start:start + WINDOW]
        d = mle_intrinsic_dimension(
            w, tau=1, max_E=MAX_E, k_neighbors=K, correction=correction,
            theiler_window=theiler, rng=np.random.default_rng(0))
        centres.append(start + WINDOW // 2)
        dims.append(d)
        roughs.append(roughness(w))
    return np.array(centres), np.array(dims), np.array(roughs)


def generalisation_step(frame, threshold=0.9):
    hit = np.flatnonzero(frame["val_acc"].to_numpy() >= threshold)
    return int(frame["step"].to_numpy()[hit[0]]) if len(hit) else None


def main():
    lb_constant = line_constant(K, 0)
    print("=== Paper settings: no Theiler window, Levina-Bickel, k=5, max_E=15, win=300 ===")
    print(f"Closed-form tangent constant at W=0, k=5: {lb_constant:.3f}\n")

    rows, traces = [], {}
    for name, filename, label in CONDITIONS:
        frame = pd.read_csv(RUNS / filename)
        series = frame["weight_norm"].to_numpy()
        steps = frame["step"].to_numpy()
        t_gen = generalisation_step(frame)

        centres, dims, roughs = trace(series, "levina_bickel", theiler=0)
        centre_steps = steps[np.clip(centres, 0, len(steps) - 1)]
        traces[name] = pd.DataFrame({"step": centre_steps, "dim_W0": dims,
                                     "roughness": roughs})

        finite = np.isfinite(dims)
        # Direction of the trend over the run. The published claim is a fall, so a
        # positive trend is not a weaker version of the claim, it is its opposite.
        trend = spearmanr(centre_steps[finite], dims[finite]).statistic
        third = len(dims) // 3
        early = np.nanmedian(dims[:third])
        late = np.nanmedian(dims[-third:])
        rho = spearmanr(dims[finite], roughs[finite]).statistic

        rows.append({"run": name, "outcome": label, "t_gen": t_gen,
                     "dim_early": early, "dim_late": late, "fall_ratio": early / late,
                     "trend_rho": trend, "spearman_dim_roughness": rho})
        gen = f"T_gen={t_gen}" if t_gen else "never generalises"
        arrow = "falls" if trend < -0.3 else ("rises" if trend > 0.3 else "flat ")
        print(f"  {name:<10} {gen:<20} early {early:5.2f} -> late {late:5.2f}"
              f"   {arrow} (trend {trend:+.2f})"
              f"   rho(dim, roughness) {rho:+.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "phase9_paper_settings_summary.csv", index=False)
    for name, frame in traces.items():
        frame.to_csv(OUT / f"phase9_trace_{name}.csv", index=False)

    grok = summary[summary.outcome == "generalises"]
    lowdata = summary[summary.run.str.startswith("lowdata")]
    print(f"\n  trend, generalising runs      "
          f"{grok.trend_rho.min():+.2f} to {grok.trend_rho.max():+.2f}")
    print(f"  trend, never-generalising WD=1 "
          f"{lowdata.trend_rho.min():+.2f} to {lowdata.trend_rho.max():+.2f}")
    rising = int((grok.trend_rho > 0.3).sum())
    print(f"  {rising} of {len(grok)} generalising runs show a rise, not the published fall.")
    print(f"  early/late ratio, generalising      "
          f"{grok.fall_ratio.min():.2f} - {grok.fall_ratio.max():.2f}")
    print(f"  early/late ratio, never-generalising "
          f"{lowdata.fall_ratio.min():.2f} - {lowdata.fall_ratio.max():.2f}")
    print(f"  late-training level across seeds of one condition: "
          f"{grok.dim_late.min():.2f} to {grok.dim_late.max():.2f} "
          f"(factor {grok.dim_late.max() / grok.dim_late.min():.1f})")

    print("\n=== What the W=0 estimate measures ===")
    pooled = pd.concat(traces.values())
    ok = np.isfinite(pooled.dim_W0) & np.isfinite(pooled.roughness)
    rho = spearmanr(pooled.dim_W0[ok], pooled.roughness[ok]).statistic
    print(f"  pooled Spearman(dimension at W=0, std(diff)/std) over all runs: {rho:+.3f}")
    print(f"  n = {int(ok.sum())} windows")
    within = summary.spearman_dim_roughness
    print(f"  within-run Spearman ranges {within.min():+.3f} to {within.max():+.3f}")
    print("  A one-line roughness ratio, with no embedding and no neighbour search,")
    print("  tracks the published statistic. The mapping is not identical across runs,")
    print("  so roughness is a strong correlate rather than a complete description.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

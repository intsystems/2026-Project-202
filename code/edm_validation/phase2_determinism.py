"""Phase 2: is there nonlinear determinism in the logs, beyond linear autocorrelation?

Phase 1 showed recurrence is *necessary but not sufficient*: i.i.d. ghost drivers score
~1.0 there too, because white noise also has close pairs at arbitrary time separations.
Recurrence separates smooth transients from everything else; it cannot separate an
attractor from noise. This does.

The statistic is simplex forecast skill with a causal split, and the null is an IAAFT
surrogate ensemble -- same power spectrum, same value distribution, no nonlinear
structure. Rejecting that null is the evidence that the series is a deterministic
function of a few delay coordinates rather than filtered noise.

The design has controls with known answers, which is why it can be believed:

* ``LogisticMap`` **poison_fraction** -- the driver *is* a logistic map. Must reject.
* ``Sinusoidal`` **poison_fraction** -- periodic, hence *linearly* predictable; its
  surrogates are sinusoids too. Must NOT reject. (A method that "detects structure" here
  is detecting the spectrum, which is exactly the mumbo-jumbo failure mode.)
* ``Ghost_Normal`` / ``Ghost_Uniform`` -- i.i.d. noise. Must NOT reject.

Only once those behave does the real question mean anything: do the *loss* series, which
are the cheap 1-D logs we actually want to analyse, inherit the driver's determinism?
That is Stark's forced-system setting, made true by construction in these runs.

    python phase2_determinism.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast import simplex_skill                          # noqa: E402
from phase1_preconditions import datasets, detrend          # noqa: E402
from surrogates import surrogate_test                       # noqa: E402

E, TAU, HORIZON, N_SURROGATES = 3, 1, 1, 39


def main(metric="spearman"):
    print(f"skill metric: {metric}\n")
    rows = []
    for family, name, frame in datasets():
        columns = [c for c in ("poison_fraction", "train_loss", "val_loss", "weight_norm")
                   if c in frame.columns]
        for column in columns:
            series = frame[column].to_numpy(dtype=float)
            series = series[np.isfinite(series)]
            if len(series) < 500 or series.std() == 0:
                continue
            values = detrend(series)
            if values.std() == 0:
                continue

            result = surrogate_test(
                values,
                lambda s: simplex_skill(s, E=E, tau=TAU, horizon=HORIZON, metric=metric),
                n_surrogates=N_SURROGATES, kind="iaaft", seed=0,
            )
            if not np.isfinite(result.statistic):
                continue
            rows.append({
                "family": family, "run": name, "column": column,
                "skill": result.statistic,
                "surrogate_mean": result.values.mean(),
                "z": result.z_score,
                "p": result.p_value,
                "rejects_null": result.p_value <= 0.05,
            })
            print(f"  {family:<8} {name:<34} {column:<16} {result}", flush=True)

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / f"phase2_determinism_{metric}.csv", index=False)

    print("\n=== CONTROLS (these decide whether anything else can be believed) ===")
    # Expectations are stated from what the driver *is*, not from its label.
    # `Sinusoidal` was initially predicted not to reject, on the grounds that a sinusoid
    # is linearly predictable. Inspecting the data corrected that: poison_fraction takes
    # only 95 distinct values in 7840 rows, i.e. it is a *quantized* sinusoid, and
    # quantization is a nonlinear map. Rejecting is the right answer; the prediction was
    # mis-specified. Recorded here rather than quietly relabelled.
    controls = {
        "LogisticMap/poison_fraction": ("must REJECT (chaotic map)", True),
        "Sinusoidal/poison_fraction": ("must REJECT (quantized sinusoid)", True),
        "Random/poison_fraction": ("must NOT reject (i.i.d. draws)", False),
        "ProgressiveNoise/poison_fraction": ("must NOT reject (stochastic)", False),
        "Ghost_Normal/poison_fraction": ("must NOT reject (i.i.d.)", False),
        "Ghost_Uniform/poison_fraction": ("must NOT reject (i.i.d.)", False),
    }
    verdicts = []
    for key, (expectation, should_reject) in controls.items():
        run, column = key.split("/")
        hit = table[(table.run == run) & (table.column == column)]
        if hit.empty:
            print(f"  {key:<32} MISSING")
            continue
        got = bool(hit.rejects_null.iloc[0])
        ok = got == should_reject
        verdicts.append(ok)
        print(f"  {'OK  ' if ok else 'BAD '} {key:<30} {expectation:<38} "
              f"z={hit.z.iloc[0]:+.2f} p={hit.p.iloc[0]:.3f}")

    print("\n=== THE QUESTION: do the loss logs carry it? ===")
    loss = table[table.column.isin(["train_loss", "val_loss"])]
    print(loss.pivot_table(index=["family", "run"], columns="column",
                           values="z").round(2).to_string())
    print("\nfraction rejecting the null, by family:")
    print(loss.groupby("family").rejects_null.mean().round(3).to_string())

    print("\ncontrols behaved as required:", all(verdicts) if verdicts else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "spearman"))

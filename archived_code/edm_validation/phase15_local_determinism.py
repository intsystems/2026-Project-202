"""Is the locally recurrent training loss deterministic, or is it noise that recurs?

Phase 14 establishes that inside an 800-step segment the training loss revisits its past
states, with a recurrence profile that plateaus like a chaotic attractor rather than
decaying like a transient. That clears the precondition of the embedding theorems, but it
does not by itself justify applying them, because recurrence does not separate an attractor
from noise: independent noise also produces close pairs at arbitrary temporal separations.

Two questions decide the matter, and both are asked against nulls rather than by eye.

**Is the local dynamic predictable beyond its linear structure?** Simplex projection is
compared against an IAAFT surrogate ensemble, which preserves the amplitude distribution
and the power spectrum while destroying any deterministic state dependence. A statistic
that does not beat that ensemble is reporting linear autocorrelation.

**Does the skill decay with the forecast horizon?** Deterministic chaos loses predictability
at a rate set by its Lyapunov exponent, so skill falls as the horizon grows. A smooth or
strongly periodic signal keeps its skill; noise has none to begin with. The shape of the
decay therefore separates the three cases where a single number cannot.

Both are calibrated on systems with known answers before being applied to the logs.

    python phase15_local_determinism.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecast import simplex_skill, skill_vs_horizon         # noqa: E402
from surrogates import surrogate_test                        # noqa: E402

DENSE = HERE / "results" / "dense"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

SEG = slice(2000, 2800)                 # the segment phase 14 measured
E, TAU, N_SURROGATES = 5, 1, 99
HORIZONS = (1, 2, 4, 8, 16, 32)
RUNS = ("grok_dense", "lowdata15_dense", "wd0_dense")
OBSERVABLES = ("train_loss", "weight_norm")


def lorenz_x(n, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn:])


def logistic_map(n, r=3.9, x0=0.4, burn=500):
    x, out = x0, []
    for i in range(n + burn):
        x = r * x * (1 - x)
        if i >= burn:
            out.append(x)
    return np.array(out)


def evaluate(name, series, rows):
    result = surrogate_test(series, lambda v: simplex_skill(v, E=E, tau=TAU, horizon=1),
                            n_surrogates=N_SURROGATES, kind="iaaft", seed=0)
    curve = skill_vs_horizon(series, E=E, tau=TAU, horizons=HORIZONS)
    decay = curve[HORIZONS[0]] - curve[HORIZONS[-1]]
    rows.append({"series": name, "skill_h1": result.statistic,
                 "surrogate_mean": float(result.values.mean()), "p": result.p_value,
                 **{f"h{h}": curve[h] for h in HORIZONS}, "decay": decay})
    print(f"    {name:<34} {result.statistic:6.3f}  {result.values.mean():6.3f}  "
          f"{result.p_value:6.3f}   "
          + " ".join(f"{curve[h]:5.2f}" for h in HORIZONS)
          + f"   {decay:+.2f}", flush=True)


def main():
    print("=== Local determinism of an 800-step segment ===")
    print("    skill: simplex projection at horizon 1, against IAAFT surrogates")
    print(f"    p floor is {1 / (N_SURROGATES + 1):.3f} with {N_SURROGATES} surrogates\n")
    header = ("    " + f"{'series':<34}" + " skill    surr       p   "
              + " ".join(f"{'h'+str(h):>5}" for h in HORIZONS) + "    decay")
    print(header)

    rows = []
    n = SEG.stop - SEG.start
    rng = np.random.default_rng(0)
    # Two Lorenz references, deliberately. At dt = 0.01 the trajectory is oversampled,
    # consecutive points are nearly identical, and an IAAFT surrogate is as predictable as
    # the original: the test has no power there, and reports p ~ 0.4 on a system that is
    # certainly deterministic. Sampling at dt = 0.1 restores it. Any series must therefore
    # be compared against a reference at a comparable sampling density, and a failure to
    # beat the null is only informative if the reference beats it.
    for label, series in (("Lorenz-63, dt=0.01 (oversampled)", lorenz_x(n)),
                          ("Lorenz-63, dt=0.1", lorenz_x(n * 10)[::10]),
                          ("logistic map (chaos)", logistic_map(n)),
                          ("sine (periodic)", np.sin(np.linspace(0, 40 * np.pi, n))),
                          ("white noise", rng.normal(size=n))):
        evaluate(label, series, rows)
    print()

    for run in RUNS:
        frame = pd.read_csv(DENSE / f"{run}.csv")
        for observable in OBSERVABLES:
            segment = frame[observable].to_numpy()[SEG]
            evaluate(f"{run.replace('_dense', '')} {observable}", segment, rows)
        print()

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase15_local_determinism.csv", index=False)

    print("    Read the row, not the number. Chaos beats the null and loses skill with")
    print("    the horizon; a periodic signal beats it and keeps skill; noise does")
    print("    neither. A log that beats the null without decaying is smooth, not chaotic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

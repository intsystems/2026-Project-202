"""Audit of our own identifiability diagnostic (report section 3.1, second observation).

The report argued that the weight-norm dimension estimate is non-identifiable because it
grows with the embedding dimension, whereas a Lorenz reference stays comparatively flat.
That comparison was confounded. The Lorenz series was 12000 samples long and the training
logs are 2000 to 3000, and the diagnostic is strongly length-dependent: a length-matched
Lorenz reference is *worse* than the logs it was supposed to outperform.

This script measures the length dependence directly, so the diagnostic is either used with
a matched reference or not used at all.

The conclusion that survives is weaker than the one originally written, and it is about the
data budget rather than about these particular series: at the lengths training logs
provide, the estimator cannot resolve even a textbook attractor, so no intrinsic dimension
is identifiable from them either way. The falsification of the published claim therefore
rests on the closed form of Proposition 1 and on the seed irreproducibility measured in
phase 9, neither of which depends on this diagnostic.

    python phase10_identifiability_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import embedding_dimension_scan                     # noqa: E402

LOGS = HERE.parent / "grokking_analysis" / "grokking_logs"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

MAX_ES = (5, 10, 15, 20, 25, 30)
LENGTHS = (1000, 2000, 3000, 6000, 12000, 24000)


def lorenz_x(n, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn:])


def ratio(series, tau):
    """Growth of the estimate from the smallest to the largest embedding dimension.

    A value near 1 means the estimate is a property of the data. A large value means it
    is a property of the embedding, and therefore that no dimension is identifiable.
    """
    scan = embedding_dimension_scan(series, MAX_ES, tau=tau)
    values = [scan[m] for m in MAX_ES]
    return values, values[-1] / values[0]


def main():
    print("=== The diagnostic depends on series length, which the report did not control ===")
    rows = []
    for n in LENGTHS:
        # tau=11 is the first autocorrelation minimum for Lorenz at dt=0.01; tau=1 for
        # noise, which has no correlation time to speak of.
        _, r_lorenz = ratio(lorenz_x(n), tau=11)
        _, r_noise = ratio(np.random.default_rng(0).normal(size=n), tau=1)
        rows.append({"n": n, "lorenz": r_lorenz, "white_noise": r_noise})
        print(f"  n={n:<6} Lorenz-63 ratio {r_lorenz:6.2f}    white noise ratio {r_noise:6.2f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "phase10_length_dependence.csv", index=False)

    print("\n=== Where the training logs actually sit ===")
    series = {
        "mod_wd1": "grokking_modular_addition_logs_to_flat_grokking_with_stochastic.csv",
        "s5_wd1": "grokking_modular_addition_logs_S_5_with_stochastic.csv",
    }
    log_rows = []
    for name, filename in series.items():
        values = pd.read_csv(LOGS / filename)["weight_norm"].to_numpy()
        _, r = ratio(values, tau=1)
        log_rows.append({"run": name, "n": len(values), "ratio": r})
        print(f"  {name:<10} n={len(values):<6} ratio {r:6.2f}")
    pd.DataFrame(log_rows).to_csv(OUT / "phase10_log_ratios.csv", index=False)

    n_logs = int(np.median([r["n"] for r in log_rows]))
    nearest = min(LENGTHS, key=lambda n: abs(n - n_logs))
    matched = frame[frame.n == nearest].iloc[0]
    print(f"\n=== Verdict at the matched length (n={nearest}) ===")
    print(f"  Lorenz-63    {matched.lorenz:6.2f}")
    print(f"  white noise  {matched.white_noise:6.2f}")
    print(f"  logs         {min(r['ratio'] for r in log_rows):6.2f}"
          f" - {max(r['ratio'] for r in log_rows):6.2f}")
    verdict = ("HAS NO POWER" if matched.lorenz >= min(r["ratio"] for r in log_rows)
               else "separates them")
    print(f"\n  At the length training logs provide, the diagnostic {verdict}:")
    print("  a known attractor scores no better than the series it should outperform.")
    print("  The report must not use it to argue that the logs behave like noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

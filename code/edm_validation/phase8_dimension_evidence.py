"""Evidence for report section 3.1: the intrinsic-dimension estimate is a line constant.

Two claims are asserted in the report and are checked here rather than taken on trust.

**Claim 1 (closed form).** For a locally straight, uniformly sampled trajectory the k
nearest neighbours of an interior point lie at temporal offsets +-1, +-2, ... , so with a
Theiler exclusion of W the sorted neighbour distances are proportional to the multiset
{W+1, W+1, W+2, W+2, ...} truncated to k. The Levina-Bickel estimate

    d = (k - 1) / sum_{j<k} log(r_k / r_j)

therefore depends only on k and W and not on the data at all. The prediction is verified
against the estimator on a synthetic straight line and against the published WD=0 control
runs.

**Claim 2 (non-identifiability).** An intrinsic dimension exists only if the estimate is
insensitive to the dimension of the space it is measured in. Sweeping the embedding
dimension separates a genuine attractor, where the estimate is flat, from a point cloud
with no resolvable manifold, where the estimate simply tracks the embedding.

    python phase8_dimension_evidence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import embedding_dimension_scan, mle_intrinsic_dimension   # noqa: E402

LOGS = HERE.parent / "grokking_analysis" / "grokking_logs"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def line_constant(k, theiler=0):
    """Closed-form Levina-Bickel estimate for a straight, uniformly sampled trajectory."""
    offsets = np.repeat(np.arange(theiler + 1, theiler + k + 1), 2)
    r = np.sort(offsets)[:k]
    return (k - 1) / np.sum(np.log(r[-1] / r[:-1]))


def lorenz_x(n=12000, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn:])


def main():
    line = np.linspace(0.0, 1.0, 3000)

    print("=== Claim 1: the estimate is a closed-form constant of a straight line ===")
    rows = []
    for k in (5, 10, 20):
        for theiler in (0, 14):
            predicted = line_constant(k, theiler)
            measured = mle_intrinsic_dimension(
                line, tau=1, max_E=10, k_neighbors=k, correction="mackay_ghahramani",
                dither=None, clamp_to_max_E=False, theiler_window=theiler)
            rows.append({"k": k, "theiler": theiler, "closed_form": predicted,
                         "measured_on_line": measured,
                         "rel_error": abs(measured - predicted) / predicted})
            print(f"  k={k:<3} W={theiler:<3} closed form {predicted:7.3f}   "
                  f"estimator on a line {measured:7.3f}   "
                  f"rel. error {rows[-1]['rel_error']:.1%}")
    pd.DataFrame(rows).to_csv(OUT / "phase8_line_constants.csv", index=False)

    print("\n=== and it is what the published control runs report ===")
    controls = {
        "mod_wd0": "grokking_modular_addition_logs_to_flat_grokking_with_stochastic_"
                   "without_wight_decay.csv",
        "s5_wd0": "grokking_modular_addition_logs_S_5_with_stochastic_without_wight_decay.csv",
    }
    control_rows = []
    for name, filename in controls.items():
        series = pd.read_csv(LOGS / filename)["weight_norm"].to_numpy()
        for theiler, label in ((0, "no exclusion"), (14, "Theiler W=14")):
            values = []
            for start in range(0, len(series) - 300, 150):
                d = mle_intrinsic_dimension(
                    series[start:start + 300], tau=1, max_E=15, k_neighbors=5,
                    theiler_window=theiler, rng=np.random.default_rng(0))
                if np.isfinite(d):
                    values.append(d)
            predicted = line_constant(5, theiler)
            control_rows.append({"run": name, "theiler": theiler,
                                 "measured_median": float(np.median(values)),
                                 "closed_form": predicted})
            print(f"  {name:<8} {label:<14} measured median {np.median(values):6.2f}   "
                  f"straight-line constant {predicted:6.2f}")
    pd.DataFrame(control_rows).to_csv(OUT / "phase8_control_plateaus.csv", index=False)

    print("\n=== Claim 2: the estimate is not identifiable on these logs ===")
    max_es = (5, 10, 15, 20, 25, 30)
    scans = {}
    lorenz = lorenz_x()
    scans["Lorenz-63 (11000 samples)"] = embedding_dimension_scan(lorenz, max_es, tau=11)
    for name, filename in (("mod_wd1", "grokking_modular_addition_logs_to_flat_grokking_"
                            "with_stochastic.csv"),
                           ("s5_wd1", "grokking_modular_addition_logs_S_5_with_stochastic.csv")):
        series = pd.read_csv(LOGS / filename)["weight_norm"].to_numpy()
        scans[f"{name} weight norm"] = embedding_dimension_scan(series, max_es, tau=1)
    noise = np.random.default_rng(0).normal(size=3000)
    scans["white noise"] = embedding_dimension_scan(noise, max_es, tau=1)

    frame = pd.DataFrame(scans).T
    frame.columns = [f"E_max={m}" for m in max_es]
    frame["ratio"] = frame.iloc[:, -1] / frame.iloc[:, 0]
    print(frame.round(2).to_string())
    frame.to_csv(OUT / "phase8_identifiability.csv")
    print("\nA flat row means the number is a property of the data; a rising row means it")
    print("is a property of the embedding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

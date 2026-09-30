"""E9 config choice on the hand-written TARGETS only (no network is looked at).

The MG configuration for E9 is chosen here, on the target signals themselves: a sum of
q sinusoids is a q-torus by construction. The config that reads these best is frozen
and then applied, unchanged, to the trained networks.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from generator import DT, target, target_spec  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
CONFIGS = {
    "E20_tau1_k20": EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding"),
    "E20_tau4_k20": EstimatorConfig(max_E=20, tau=4, k_neighbors=20, theiler="embedding", theiler_cap=156),
    "E20_tau8_k20": EstimatorConfig(max_E=20, tau=8, k_neighbors=20, theiler="embedding", theiler_cap=320),
    "E20_acorr_k20": EstimatorConfig(max_E=20, tau="acorr", k_neighbors=20, theiler="autocorr", theiler_cap=320),
}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(L) * DT
    print("config           " + " ".join(f"{k}{q}" for k, q in
                                         [("T", 1), ("T", 2), ("T", 3), ("T", 4), ("T", 5), ("H", 4)]))
    for name, cfg in CONFIGS.items():
        vals = []
        t0 = time.perf_counter()
        for kind, q in [("torus", 1), ("torus", 2), ("torus", 3), ("torus", 4), ("torus", 5), ("harmonic", 4)]:
            x = target(t, target_spec(kind, q, 1))
            x = np.tanh(0.8 * x + 0.3)                     # a nonlinear read-out, like a neuron
            vals.append(estimate(x + 1e-3 * rng.normal(size=L), cfg).MG)
        print(f"{name:16s} " + " ".join(f"{v:5.2f}" for v in vals), f"({time.perf_counter()-t0:.0f}s)",
              flush=True)

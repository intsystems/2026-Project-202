"""E9 pilot, seed 0 only: does FORCE learn each target, and what does Lyapunov say?

No MG is computed here. Only learning success (spectral fidelity), the chaos test and
the number of near-zero Lyapunov exponents are looked at, to fix N and the training
length before the confirmatory run.
"""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import target_spec, train, rollout, spectral_fidelity, peak_count  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    with threadpool_limits(limits=4):
        for kind, q in [("torus", 1), ("torus", 2), ("torus", 3), ("torus", 4), ("torus", 5),
                        ("harmonic", 4)]:
            spec = target_spec(kind, q, 0)
            t0 = time.perf_counter()
            j, u, w, x, ttrain = train(0, n, spec, steps)
            obs, z, lyap, tt = rollout(j, u, w, x, burn=5000, length=16384, lyap_len=16384)
            fid, fmin = spectral_fidelity(z, spec)
            n0 = int((lyap >= -0.005).sum())
            print(f"{kind:8s} q={q} N={n}: fidelity {fid:.3f} (min line {fmin:.3f}) peaks {peak_count(z)} "
                  f"lyap top {np.round(lyap[:7], 4)} near-zero(>=-0.005) {n0}  "
                  f"train {ttrain:.0f}s total {time.perf_counter()-t0:.0f}s", flush=True)

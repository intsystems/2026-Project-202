"""E9 pilot 2, seed 0: multi-output FORCE (q read-outs, one sinusoid each). No MG here."""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import target_spec, train_multi, rollout, fidelity_multi, peak_count  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    gain = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    qs = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [3, 4, 5, 1, 2]
    arms = [("torus", q) for q in qs] + ([("harmonic", 4)] if len(sys.argv) <= 4 else [])
    if len(sys.argv) > 5:
        arms = [(k, int(v)) for k, v in (a.split(":") for a in sys.argv[5].split(","))]
    with threadpool_limits(limits=4):
        for kind, q in arms:
            spec = target_spec(kind, q, 0)
            j, u, w, x, ttrain = train_multi(0, n, spec, steps, gain=gain)
            obs, z, lyap, tt = rollout(j, u, w, x, burn=5000, length=16384, lyap_len=16384)
            fmin, shares = fidelity_multi(tt["zs"], spec)
            print(f"{kind:8s} q={q} N={n} g={gain}: min fidelity {fmin:.3f} {shares} peaks(sum) {peak_count(z)} "
                  f"lyap top {np.round(lyap[:7], 4)} near-zero {int((lyap >= -0.005).sum())} "
                  f"train {ttrain:.0f}s", flush=True)

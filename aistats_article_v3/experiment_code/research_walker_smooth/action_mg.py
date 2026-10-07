"""Post-hoc MG analysis on action logs for the temporal-smoothing experiment.

The training and primary state-space results are untouched.  This script tests
whether the scalar estimator tracks the property that was explicitly changed:
temporal complexity of the policy's actions.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "research_walker_phase_wide"))
from features import measure
W = 2048
TAU = 8
ENDS = (2048, 3072, 4096)

def scalar_signals(actions):
    a = np.asarray(actions, dtype=float)
    da = np.diff(a, axis=0, prepend=a[:1])
    return {**{f"action_{j}": a[:, j] for j in range(a.shape[1])},
            "action_norm": np.linalg.norm(a, axis=1),
            "delta_action_norm": np.linalg.norm(da, axis=1),
            "mean_action": a.mean(axis=1)}

def mg(x):
    values = []
    for end in ENDS:
        r = measure(x[end-W:end], W, TAU)
        if np.isfinite(r.get("MG", np.nan)) and not r.get("degenerate", True):
            values.append(float(r["MG"]))
    return float(np.median(values)) if values else np.nan

def main():
    rows = []
    for seed in range(221, 226):
        pair = json.loads((H / f"pair{seed}.json").read_text())
        for reset in pair["common_resets"]:
            roots = [H/f"seed{seed}_lambda0/step1048576/reset{reset}",
                     H/f"seed{seed}_lambda1/step1048576/reset{reset}"]
            if not all((r/"trajectory.npz").exists() for r in roots):
                continue
            z0, z1 = [np.load(r/"trajectory.npz") for r in roots]
            s0, s1 = scalar_signals(z0["actions"]), scalar_signals(z1["actions"])
            for name in s0:
                m0, m1 = mg(s0[name]), mg(s1[name])
                rows.append(dict(seed=seed, reset=reset, signal=name,
                                 MG_control=m0, MG_smooth=m1,
                                 MG_ratio=m1/m0 if np.isfinite(m0) and m0 else np.nan))
    frame = pd.DataFrame(rows)
    frame.to_csv(H/"action_mg.csv", index=False)
    summary=[]
    for (seed, signal), g in frame.groupby(["seed", "signal"]):
        r=g.MG_ratio.dropna()
        summary.append(dict(seed=int(seed), signal=signal, n=len(r),
                            ratio=float(r.median()),
                            fraction_decrease=float((r<1).mean())))
    out=pd.DataFrame(summary)
    out.to_csv(H/"action_mg_summary.csv", index=False)
    print(out.pivot(index="seed", columns="signal", values="ratio").round(3).to_string())

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

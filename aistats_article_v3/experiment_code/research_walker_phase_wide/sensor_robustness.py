"""Post-hoc sensor robustness check; does not change training or primary results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from features import measure

H = Path(__file__).resolve().parent
WINDOW = 2048
TAU = 8

def signals(z):
    q = z["qpos"]
    right, left = q[:, 4], q[:, 7]
    return {
        "knee_mean": (right + left) / 2.0,
        "knee_difference": right - left,
    }

def one(x):
    rows = []
    for end in (WINDOW, 3072, 4096):
        rows.append(measure(x[end-WINDOW:end], WINDOW, TAU) | {"end": end})
    frame = pd.DataFrame(rows)
    good = np.isfinite(frame.MG) & ~frame.degenerate
    return float(frame.loc[good, "MG"].median()) if good.any() else np.nan

def main():
    rows = []
    for seed in range(271, 276):
        a = pd.read_csv(H/f"seed{seed}_lambda0/test.csv").set_index("reset")
        b = pd.read_csv(H/f"seed{seed}_lambda3/test.csv").set_index("reset")
        pair = json.loads((H/f"pair{seed}.json").read_text())
        for reset in pair["common_resets"]:
            if reset not in a.index or reset not in b.index:
                continue
            roots = [H/f"seed{seed}_lambda0/step1048576/reset{reset}",
                     H/f"seed{seed}_lambda3/step1048576/reset{reset}"]
            if not all((r/"trajectory.npz").exists() for r in roots):
                continue
            az, bz = [np.load(r/"trajectory.npz") for r in roots]
            sa, sb = signals(az), signals(bz)
            for name in sa:
                mg0, mg1 = one(sa[name]), one(sb[name])
                rows.append(dict(seed=seed, reset=reset, sensor=name,
                                 MG_control=mg0, MG_tracking=mg1,
                                 MG_ratio=mg1/mg0 if np.isfinite(mg0) and mg0 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(H/"sensor_robustness.csv", index=False)
    summary = []
    for (seed, sensor), g in out.groupby(["seed", "sensor"]):
        r = g.MG_ratio.dropna()
        summary.append(dict(seed=int(seed), sensor=sensor, n=len(r),
                            ratio=float(r.median()),
                            fraction_decrease=float((r < 1).mean())))
    s = pd.DataFrame(summary)
    s.to_csv(H/"sensor_robustness_summary.csv", index=False)
    print(s.pivot(index="seed", columns="sensor", values="ratio").round(3).to_string())

if __name__ == "__main__":
    main()

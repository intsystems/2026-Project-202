"""Re-score the saved logs of pilot.py under several MG configurations.

The question is specificity: an arm whose trajectory dimension falls (freeze,
lr_drop) must move MG, and an arm that changes only the observer's time scale
(smooth) or the noise amplitude (batch_up) must not. The fixed lag tau=1 fails
the smoothing control, so the adaptive lag is the candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.embedding import resolve_tau  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
W, S, HALF = 500, 250, 3000
CONFIGS = {
    "t1_E10": EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding"),
    "acorr_E10": EstimatorConfig(max_E=10, tau="acorr", k_neighbors=10, theiler="embedding"),
    "acorr_E10_auto": EstimatorConfig(max_E=10, tau="acorr", k_neighbors=10, theiler="autocorr"),
    "t4_E10": EstimatorConfig(max_E=10, tau=4, k_neighbors=10, theiler="embedding"),
}
OBS = ("probe_loss", "batch_loss", "grad_norm")


def main() -> None:
    ref = pd.concat([pd.read_csv(RES / "windows.csv")])[["arm", "seed", "start", "traj_PR"]]
    rows = []
    for f in sorted(RES.glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        logs = np.load(f)
        for a in range(0, len(logs["probe_loss"]) - W + 1, S):
            row = {"arm": arm, "seed": int(seed), "start": a}
            for name, cfg in CONFIGS.items():
                for o in OBS:
                    seg = logs[o][a:a + W]
                    row[f"{name}|{o}"] = estimate(seg, cfg).MG
                    if o == "probe_loss":
                        row[f"{name}|tau"] = resolve_tau(cfg, seg)
            rows.append(row)
    d = pd.DataFrame(rows).merge(ref, on=["arm", "seed", "start"])
    d.to_csv(RES / "reanalysed.csv", index=False)

    pre = d[(d.start + W <= HALF) & (d.start >= 1000)]
    post = d[d.start >= HALF + W]
    r = (post.groupby(["arm", "seed"]).median(numeric_only=True)
         / pre.groupby(["arm", "seed"]).median(numeric_only=True))
    keep = ["traj_PR"] + [c for c in r.columns if "|probe_loss" in c or "|tau" in c]
    pd.set_option("display.width", 200)
    print(r.groupby("arm")[keep].median().round(2).to_string())
    r.to_csv(RES / "ratios.csv")


if __name__ == "__main__":
    main()

"""E13 analysis.

PROTOCOL (written before the runs finished). Probe window: steps [500, 1500) of each log.
Runs whose probe contains a non-finite value are dropped by every rule (any practitioner
sees a divergence). Statistics of the probe window of each log (param_norm, batch_loss,
grad_norm): MG (E=20, tau=1, k=20) and every competitor of cnn_competitors.EXTRA; plus the
standard probe criteria: mean mini-batch loss over steps [1300, 1500) and its drop from
steps [0, 200).
Calibration seeds 0-2: for every (statistic, log) choose the sign maximising the mean
within-seed Spearman correlation with final test accuracy. Test seeds 3-5: in each seed pick
the setting with the highest signed statistic; regret = best final accuracy minus the
picked one. Reported: mean test regret, mean test Spearman.
Prediction: MG on the best log (chosen on calibration) has lower mean test regret than the
probe-loss rule.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "research_trajectory_reference"))
from cnn_competitors import EXTRA  # noqa: E402
import detector as D  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

R = HERE / "results_lr"
LOGS = ("param_norm", "batch_loss", "grad_norm")
CAL, TEST = (0, 1, 2), (3, 4, 5)


def feats(f):
    rec = json.load(open(f))
    z = np.load(str(f).replace("eval_", "logs_").replace(".json", ".npz"))
    row = {k: rec[k] for k in ("lr", "batch", "seed", "final_acc")}
    bl = z["batch_loss"]
    row["probe_ok"] = bool(all(np.all(np.isfinite(z[k][:1500])) for k in LOGS))
    if not row["probe_ok"]:
        return row
    row["probe_loss"] = -float(np.mean(bl[1300:1500]))          # lower loss = better, so negate
    row["probe_drop"] = float(np.mean(bl[:200]) - np.mean(bl[1300:1500]))
    for lg in LOGS:
        seg = z[lg][500:1500].astype(float)
        row[f"MG|{lg}"] = estimate(seg, D.CFG).MG
        for k, fn in EXTRA.items():
            try:
                row[f"{k}|{lg}"] = float(fn(seg))
            except Exception:
                row[f"{k}|{lg}"] = np.nan
    return row


def main():
    files = sorted(R.glob("eval_*.json"))
    with Pool(8) as p:
        df = pd.DataFrame(p.map(feats, files))
    df.to_csv(R / "features.csv", index=False)
    ok = df[df.probe_ok]
    stats = [c for c in ok.columns if c not in ("lr", "batch", "seed", "final_acc", "probe_ok")]
    rows = []
    for c in stats:
        cal_rho = np.nanmean([spearmanr(g[c], g.final_acc)[0] for _, g in ok[ok.seed.isin(CAL)].groupby("seed")])
        s = 1 if not np.isfinite(cal_rho) or cal_rho >= 0 else -1
        reg, rho = [], []
        for _, g in ok[ok.seed.isin(TEST)].groupby("seed"):
            sc = (s * g[c]).fillna(-np.inf)
            reg.append(g.final_acc.max() - g.final_acc.iloc[int(np.argmax(sc.to_numpy()))])
            rho.append(spearmanr(s * g[c], g.final_acc)[0])
        rows.append({"stat": c, "sign": s, "cal_rho": abs(cal_rho), "test_regret": np.mean(reg),
                     "test_rho": np.nanmean(rho)})
    res = pd.DataFrame(rows).sort_values("test_regret")
    res.to_csv(R / "selection.csv", index=False)
    print(df.groupby(["lr", "batch"]).final_acc.agg(["mean", "min", "max"]).round(3).to_string())
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""E6 detector with stronger competitors, on the same saved logs and the same protocol.

Competitors are computed per window (W=1000, stride 500) on the parameter-norm log; the
delay-space ones use MG's reconstruction (E=20, tau=1). Each statistic gets the E6
calibration: on E4 seeds 0-3 choose (M, B, sign) and delta (no alarm on no-event runs + 0.02);
test on E5 seeds 10-13. Nothing here was used to tune MG.
"""
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(REF)); sys.path.insert(0, str(HERE))
import detector as D  # noqa: E402
import baselines as BL  # noqa: E402

EXTRA = {"spectral_entropy": BL.spectral_entropy, "perm_entropy": partial(BL.perm_entropy, lag=1),
         "self_repeat": BL.self_repeat, "roughness": BL.roughness,
         "recurrence_rate": partial(BL.recurrence_rate, tau=1), "corr_dim": partial(BL.corr_dim, tau=1),
         "twonn": partial(BL.twonn_fit, tau=1), "linear_pr": partial(BL.linear_pr, tau=1)}
OUT = HERE / "results_cnn"


def windows_of(args):
    path, arm_override = args
    arm, seed = path.stem[5:].rsplit("_s", 1)
    x = np.load(path)["param_norm"]
    variants = {arm: x}
    if arm_override and arm == "base":
        sc = x.copy(); sc[D.EVENT:] *= 10
        sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[D.EVENT:] = c[D.EVENT:]
        variants.update({"scale": sc, "smooth": sm})
    rows = []
    for name, v in variants.items():
        for a in range(0, len(v) - D.W + 1, D.S):
            seg = v[a:a + D.W]
            rows.append({"arm": name, "seed": int(seed), "start": a,
                         **{k: f(seg) for k, f in EXTRA.items()}})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    cal_files = [(f, False) for f in sorted((REF / "results_cifar").glob("logs_*_s*.npz"))
                 if f.stem[5:].rsplit("_s", 1)[0] in D.CAL_EVENTS + D.CAL_NULL]
    test_files = [(f, True) for f in sorted((REF / "results_graded").glob("logs_*_s*.npz"))]
    with Pool(12) as p:
        cal_x = pd.DataFrame([r for rr in p.map(windows_of, cal_files) for r in rr])
        test_x = pd.DataFrame([r for rr in p.map(windows_of, test_files) for r in rr])
    cal = pd.read_csv(REF / "results_detector" / "cal_windows.csv").merge(cal_x, on=["arm", "seed", "start"])
    test = pd.read_csv(REF / "results_detector" / "test_windows.csv").merge(test_x, on=["arm", "seed", "start"])
    cal.to_csv(OUT / "cal_windows.csv", index=False); test.to_csv(OUT / "test_windows.csv", index=False)

    stats = list(D.STATS) + list(EXTRA)
    orig_cal = D.calibrate

    def calibrate_any(cal, stat):           # every statistic may alarm on drops or rises
        D_stat = stat
        if stat == "MG":
            return orig_cal(cal, stat)
        return orig_cal(cal, D_stat)
    chosen = {s: calibrate_any(cal, s) for s in stats}
    strong = ["lr10", "lr100", "freeze_head", "freeze_bias", "prune50", "prune80", "prune95"]
    weak = ["lr3", "freeze12"]
    rows = []
    for s, r in chosen.items():
        pr = D.evaluate(test, s, r["M"], r["B"], r["delta"], r["sign"], D.TEST_NULL, D.EVENT)
        rows.append({"stat": s, "sign": r["sign"], "cal_hit": r["cal_hit"],
                     "strong_hit": f"{int(pr[pr.arm.isin(strong)].hit.sum())}/{int(pr.arm.isin(strong).sum())}",
                     "weak_hit": f"{int(pr[pr.arm.isin(weak)].hit.sum())}/{int(pr.arm.isin(weak).sum())}",
                     "null_alarm": f"{int(pr[pr.arm.isin(D.TEST_NULL)].false_alarm.sum())}/{int(pr.arm.isin(D.TEST_NULL).sum())}",
                     "early_alarm": int(pr[pr.event].false_alarm.sum()),
                     "delay": float(np.nanmedian(pr[pr.arm.isin(strong)].delay))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "detector_competitors.csv", index=False)
    json.dump(chosen, open(OUT / "chosen_rules.json", "w"), indent=1, default=float)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

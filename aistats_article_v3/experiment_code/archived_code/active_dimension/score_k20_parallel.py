"""Parallel scorer for the already generated k=1..20 calibration trajectories.

The original scorer was intentionally simple and repeated every KD-tree query in
one process.  This version uses the estimator configuration already frozen on
CAL_R and distributes independent trajectory/observer jobs across CPU cores.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibration_k20 import CAL_R, OBSERVERS, OUT, TEST_R, load_npz
from mg import MGConfig, summarise


def tags(path: Path, spec):
    tag = spec.get("tag", spec["mode"])
    for prefix, value in (("qp_slow_", "qp_slow"),
                          ("qp_scale2_", "qp_scale2"),
                          ("qp_rotate_", "qp_rotate")):
        if path.stem.startswith(prefix):
            tag = value
    return tag


def one(job):
    path_s, observer, cfg_dict = job
    path = Path(path_s)
    cfg = MGConfig(**cfg_dict)
    logs, _, _, info, spec = load_npz(path)
    x = np.asarray(logs[observer], float)
    if len(x) < cfg.window or x.std() <= 1e-12:
        return None
    x = (x-x.mean())/x.std()
    rec = summarise(x, cfg, seed=int(spec["seed"]))
    return dict(file=path.name, tag=tags(path, spec), mode=spec["mode"],
                r=int(spec["r"]), seed=int(spec["seed"]), observer=observer,
                available=info.get("available", 20),
                functional_rank=info.get("func_rank"),
                functional_pr=info.get("func_PR"),
                traj_rank=info.get("traj_rank"), traj_pr=info.get("traj_PR"),
                update_pr=info.get("upd_PR"), **rec)


def main():
    frozen = json.loads((OUT/"frozen_k20.json").read_text())
    cfg_dict = frozen["config"]
    paths = [p for p in sorted((OUT/"trajectories").glob("*.npz"))
             if not p.stem.startswith("smoke")]
    jobs = [(str(p), o, cfg_dict) for p in paths for o in OBSERVERS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [x for x in ex.map(one, jobs, chunksize=1) if x is not None]
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"scores_frozen.csv", index=False)

    # Calibrate absolute MG values using calibration ranks only.  Raw MG is also
    # retained: correlation does not require this affine rescaling.
    cal = df[(df.tag == "qp") & df.r.isin(CAL_R)]
    maps = {o: np.polyfit(g.MG, g.traj_pr, 1) for o, g in cal.groupby("observer")}
    clean = df[(df.tag == "qp") & df.r.isin(TEST_R)].copy()
    clean["MG_cal"] = [maps[o][0]*v + maps[o][1]
                       for o, v in zip(clean.observer, clean.MG)]
    summary = []
    for o, g in clean.groupby("observer"):
        summary.append(dict(observer=o,
            rho_raw=float(spearmanr(g.traj_pr, g.MG).statistic),
            mae_raw=float(np.mean(np.abs(g.MG-g.traj_pr))),
            rho_cal=float(spearmanr(g.traj_pr, g.MG_cal).statistic),
            mae_cal=float(np.mean(np.abs(g.MG_cal-g.traj_pr))),
            max_error_cal=float(np.max(np.abs(g.MG_cal-g.traj_pr))),
            degenerate=float(g.frac_degenerate.mean())))
    pd.DataFrame(summary).sort_values("mae_cal").to_csv(
        OUT/"heldout_qp_summary.csv", index=False)

    med = (df.groupby(["tag", "mode", "observer", "r"])
           .agg(MG=("MG", "median"), LB=("LB", "median"),
                TwoNN=("TwoNN", "median"), PRdelay=("PRdelay", "median"),
                traj_pr=("traj_pr", "median"), update_pr=("update_pr", "median"),
                degenerate=("frac_degenerate", "mean")).reset_index())
    med.to_csv(OUT/"frozen_per_r.csv", index=False)

    # Pair each control with qp/r=10/seed=0.  Comparing a one-seed control to
    # the three-seed median would confound invariance with seed variability.
    ctrl = []
    base = df[(df.tag == "qp") & (df.r == 10) & (df.seed == 0)]
    for tag in ("qp_scale2", "qp_rotate"):
        other = df[(df.tag == tag) & (df.r == 10) & (df.seed == 0)]
        for o in OBSERVERS:
            a, b = base[base.observer == o], other[other.observer == o]
            if len(a) and len(b):
                ctrl.append(dict(control=tag, observer=o,
                                 delta_MG=float(b.MG.iloc[0]-a.MG.iloc[0])))
    pd.DataFrame(ctrl).to_csv(OUT/"invariance_controls.csv", index=False)
    print(pd.read_csv(OUT/"heldout_qp_summary.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""E7b -- the Theiler comparison of E7 on a subset, for when the full sweep is too slow.

Same question, same trajectories, same two exclusions; seven ranks spanning the range instead
of all twenty, and the two observers the twenty-direction configuration scores best on.

    python e7b_theiler_quick.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import mg as MG

HERE = Path(__file__).resolve().parent
TRAJ = HERE / "results" / "k20_calibration" / "trajectories"
OUT = HERE / "results" / "e7_theiler"
OUT.mkdir(parents=True, exist_ok=True)

R_SUBSET = (2, 5, 8, 11, 14, 17, 20)
OBSERVERS = ("w_fro", "c_norm", "g_fro")
ARMS = (("capped_150", 150), ("span_624", 624))


def participation_ratio(M):
    X = np.asarray(M, float)
    X = X - X.mean(0, keepdims=True)
    s2 = np.linalg.svd(X, compute_uv=False) ** 2
    if not np.isfinite(s2).all() or s2.sum() <= 0:
        return np.nan
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def job(path, label, theiler):
    MG.THEILER_CAP = 10 ** 9          # must be lifted inside the worker
    d = np.load(path, allow_pickle=True)
    spec = json.loads(str(d["spec_json"]))
    r, seed = int(spec["r"]), int(spec["seed"])
    pr_pos = participation_ratio(d["C"])
    cfg = MG.MGConfig(max_E=40, tau=16, k_neighbors=20, theiler=int(theiler),
                      window=8000, stride=4000, dither=1e-9)
    rows = []
    for o in OBSERVERS:
        x = np.asarray(d[f"log__{o}"], float)
        if not np.isfinite(x).all() or x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()
        rec = MG.summarise(z, cfg)
        rows.append(dict(arm=label, theiler=int(theiler), r=r, seed=seed, observer=o,
                         truth=pr_pos, MG=rec.get("MG"), LB=rec.get("LB"),
                         PRdelay=rec.get("PRdelay"),
                         theiler_used=rec.get("theiler_used"),
                         degenerate=float(rec.get("frac_degenerate", 1.0)) > 0.5))
    return rows


def main():
    paths = [p for p in sorted(TRAJ.glob("qp_r*_s*.npz"))
             if all(t not in p.name for t in ("slow", "rotate", "scale2"))
             and int(p.name.split("_r")[1][:2]) in R_SUBSET]
    jobs = [(p, lab, th) for p in paths for lab, th in ARMS]
    print(f"{len(paths)} trajectories x {len(ARMS)} exclusions = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=10, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "theiler_quick_raw.csv", index=False)

    lines = [f"ranks {R_SUBSET}, observers {OBSERVERS}, 3 seeds"]
    for lab, th in ARMS:
        a = df[(df.arm == lab) & (~df.degenerate)]
        med = (a.groupby(["observer", "r"])
                 .agg(MG=("MG", "median"), truth=("truth", "median")).reset_index())
        maes, rhos, tops = [], [], []
        for o, g in med.groupby("observer"):
            g = g.sort_values("r")
            maes.append(float(np.abs(g.MG - g.truth).mean()))
            rhos.append(float(spearmanr(g.MG, g.truth).statistic))
            tops.append(float(g.MG.iloc[-1]))
        used = sorted(a.theiler_used.unique())
        lines.append(f"{lab:>12}  used={used}  median MAE {np.median(maes):6.3f}  "
                     f"median rho {np.median(rhos):+.3f}  median MG at r=20 {np.median(tops):6.2f}")
        med.to_csv(OUT / f"quick_by_rank_{lab}.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "quick_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

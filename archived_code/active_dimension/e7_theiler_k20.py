"""E7 -- does the twenty-direction result survive its own Theiler rule?

The frozen twenty-direction configuration asks for ``theiler="autocorr"``, which
``edm.dimension.resolve_theiler_window`` resolves to *at least the full embedding span*
``(max_E - 1) * tau``.  At ``max_E = 40`` and ``tau = 16`` that span is 624 samples.  But
``mg.THEILER_CAP`` truncates every Theiler window at 150, because the neighbour query cost is
linear in it, so the number actually used is 150 -- less than a quarter of the span the
configuration asks for.

Delay vectors 150 samples apart therefore still overlap in time: they are built from
different samples (150 is not a multiple of 16), but they are drawn from windows that share
474 of their 624 samples' worth of history, and on a smooth series that is enough for them to
be near-neighbours for reasons of continuity rather than of recurrence.  That is exactly the
inflation Theiler (1986) introduced the exclusion to prevent, and it would bias the estimate
*downwards* towards the dimension of the local tangent.

This re-scores the stored trajectories at both exclusions -- the capped 150 and the full
embedding span 624 -- and reports what changes.  It is pure re-analysis: no trajectory is
regenerated, so nothing about the systems can drift.

The cost is superlinear in the exclusion, because ``_neighbor_distances`` queries
``k + 2 * theiler + 1`` neighbours per point: the 624 arm over all twenty ranks did not
finish in an hour on ten cores.  ``e7b_theiler_quick.py`` runs the same comparison on seven
ranks and three observers in about three minutes, and is the version the paper reports.

    python e7_theiler_k20.py            # slow; prefer e7b_theiler_quick.py
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import mg as MG

# the point of this experiment is to exceed the cap, so lift it here and record the
# exclusion that was actually used on every row
MG.THEILER_CAP = 10 ** 9

HERE = Path(__file__).resolve().parent
TRAJ = HERE / "results" / "k20_calibration" / "trajectories"
OUT = HERE / "results" / "e7_theiler"
OUT.mkdir(parents=True, exist_ok=True)

#: the eight state-only observers the twenty-direction configuration was frozen on
OBSERVERS = ("w_fro", "c_norm", "g_fro", "g_proj", "c_proj1", "fn_fro", "fn_proj1")

#: (label, Theiler exclusion in samples).  None means "as the frozen config asks", i.e.
#: capped at THEILER_CAP.
ARMS = (("capped_150", 150), ("span_624", 624))


def participation_ratio(M):
    """PR of the covariance spectrum of the rows of M, mean-removed."""
    X = np.asarray(M, float)
    X = X - X.mean(0, keepdims=True)
    s2 = np.linalg.svd(X, compute_uv=False) ** 2
    if not np.isfinite(s2).all() or s2.sum() <= 0:
        return np.nan
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def job(path, theiler_label, theiler):
    # joblib's workers re-import mg, so the cap has to be lifted inside the worker;
    # lifting it in the parent alone leaves every arm at 150 and they come out identical
    MG.THEILER_CAP = 10 ** 9
    d = np.load(path, allow_pickle=True)
    spec = json.loads(str(d["spec_json"]))
    r = int(spec["r"])
    seed = int(spec["seed"])
    # the ground truth the paper scores against: PR of the position and of the update
    # covariance, measured on the trajectory rather than taken from the label r
    pr_pos = participation_ratio(d["C"])
    pr_upd = participation_ratio(d["D"])

    cfg = MG.MGConfig(max_E=40, tau=16, k_neighbors=20, theiler=int(theiler),
                      window=8000, stride=4000, dither=1e-9)
    rows = []
    for o in OBSERVERS:
        key = f"log__{o}"
        if key not in d:
            continue
        x = np.asarray(d[key], float)
        if not np.isfinite(x).all() or x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()
        rec = MG.summarise(z, cfg)
        rows.append(dict(arm=theiler_label, theiler=int(theiler), r=r, seed=seed,
                         observer=o, pr_pos=pr_pos, pr_upd=pr_upd,
                         MG=rec.get("MG"), LB=rec.get("LB"),
                         PRdelay=rec.get("PRdelay"),
                         theiler_used=rec.get("theiler_used"),
                         n_windows=rec.get("n_windows"),
                         degenerate=float(rec.get("frac_degenerate", 1.0)) > 0.5))
    return rows


def main():
    paths = sorted(TRAJ.glob("qp_r*_s*.npz"))
    paths = [p for p in paths if "slow" not in p.name and "rotate" not in p.name
             and "scale2" not in p.name]
    if not paths:
        raise SystemExit(f"no trajectories under {TRAJ}")
    jobs = [(p, lab, th) for p in paths for lab, th in ARMS]
    print(f"{len(paths)} trajectories x {len(ARMS)} exclusions = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=10, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "theiler_raw.csv", index=False)

    # score exactly as the paper does: median over seeds at each rank, per observer,
    # against the measured position PR
    lines = []
    for lab, th in ARMS:
        a = df[(df.arm == lab) & (~df.degenerate)]
        if a.empty:
            lines.append(f"{lab:>16}  all windows degenerate")
            continue
        med = (a.groupby(["observer", "r"])
                 .agg(MG=("MG", "median"), truth=("pr_pos", "median")).reset_index())
        per_obs = []
        for o, g in med.groupby("observer"):
            g = g.sort_values("r")
            mae = float(np.abs(g.MG - g.truth).mean())
            rho = float(spearmanr(g.MG, g.truth).statistic)
            per_obs.append((o, mae, rho, float(g.MG.iloc[-1]), float(g.truth.iloc[-1])))
        per_obs.sort(key=lambda t: t[1])
        mae_med = float(np.median([p[1] for p in per_obs]))
        rho_med = float(np.median([p[2] for p in per_obs]))
        top = float(np.median([p[3] for p in per_obs]))
        lines.append(f"{lab:>16}  Theiler={th:>4}  median MAE {mae_med:6.3f}  "
                     f"median rho {rho_med:+.3f}  median MG at r=20 {top:6.2f}  "
                     f"(n_obs {len(per_obs)})")
        pd.DataFrame(per_obs, columns=["observer", "MAE", "rho", "MG_r20", "truth_r20"]) \
            .to_csv(OUT / f"by_observer_{lab}.csv", index=False)
        med.to_csv(OUT / f"by_rank_{lab}.csv", index=False)

    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n=== recovery against the measured position PR ===")
    print(txt)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

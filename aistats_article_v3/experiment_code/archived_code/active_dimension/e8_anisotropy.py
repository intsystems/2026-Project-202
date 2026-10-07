"""E8 -- when the manifold dimension and the covariance participation ratio disagree,
which one does the estimator report?

Every construction in this study equalises the drive amplitudes, so the r-torus it produces is
isotropic: its manifold dimension is r *and* the participation ratio of its trajectory
covariance is r.  The two agree by construction, and an experiment in which they agree cannot
say which of them the estimator measures.

They are different quantities.  The manifold dimension counts the coordinates needed to specify
a point; the participation ratio is an effective rank, and anisotropy depresses it.  An r-torus
with geometrically decaying amplitudes still needs r coordinates, but its covariance PR falls
well below r.  This experiment separates them on the system of section 5.1, which has no
optimiser and no feedback, so nothing but the amplitude profile changes:

    W(t) = diag(b_i + delta_i(t)),  delta_l(t) = a_l sin(2 pi f_l t + phi_l),  a_l = rho**l

    manifold dimension = r                       (held fixed along each block)
    covariance PR      = (sum a^2)^2 / sum a^4   (falls as rho falls)

observed through the squared Frobenius norm, and estimated with the frozen eight-direction
configuration.  Two outcomes are informative and they point in opposite directions.  If MG
follows r, it is estimating the manifold dimension and the paper's ground truth is misnamed.
If MG follows the PR, it is estimating an effective rank and the delay-embedding language is
the wrong justification for it.

    python e8_anisotropy.py            # ~3 min on 10 cores
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from e1_calibration import load_frozen
from generators import frequencies

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e8_anisotropy"
OUT.mkdir(parents=True, exist_ok=True)

R_VALUES = (2, 4, 6, 8)
#: a_l = RHO**l.  1.0 is the isotropic construction the rest of the paper uses.
RHOS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
SEEDS = (0, 1, 2)
T = 12_000
F0 = 1 / 16.0          # F_FAST: the regime in which the estimator is accurate
BAND = 2.0             # one octave for every r, so bandwidth is not a confound


def participation_ratio(M):
    X = np.asarray(M, float)
    X = X - X.mean(0, keepdims=True)
    s2 = np.linalg.svd(X, compute_uv=False) ** 2
    if not np.isfinite(s2).all() or s2.sum() <= 0:
        return np.nan
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def job(r, rho, seed, cfg):
    rng = np.random.default_rng(9001 * seed + 17 * r + int(1000 * rho))
    f = frequencies(r, F0, BAND)
    ph = rng.uniform(0, 2 * np.pi, r)
    a = np.array([rho ** l for l in range(r)], float)
    t = np.arange(T, dtype=float)

    # the r oscillating coordinates: an r-torus for every rho > 0
    D = a[None, :] * np.sin(2 * np.pi * np.outer(t, f) + ph[None, :])
    b = 1.0 + 0.5 * rng.random(r)                 # non-zero offsets, as in section 5.1
    x = ((b[None, :] + D) ** 2).sum(1)            # squared Frobenius norm of the diagonal

    pr_pos = participation_ratio(D)
    # what a diagonal amplitude profile alone predicts, as a check on the construction
    a2 = a ** 2
    pr_pred = float(a2.sum() ** 2 / (a2 ** 2).sum())

    z = (x - x.mean()) / x.std()
    rec = MG.summarise(z, cfg)
    return dict(r=r, rho=rho, seed=seed, manifold=r, pr_pos=pr_pos, pr_pred=pr_pred,
                MG=rec.get("MG"), LB=rec.get("LB"), PRdelay=rec.get("PRdelay"),
                specPR256=rec.get("specPR256"), roughness=rec.get("roughness"),
                frac_degenerate=rec.get("frac_degenerate"))


def main():
    cfg, _ = load_frozen()
    print(f"frozen config: {cfg}", flush=True)
    jobs = [(r, rho, s, cfg) for r in R_VALUES for rho in RHOS for s in SEEDS]
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=10, verbose=5, batch_size=2)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "aniso_raw.csv", index=False)

    ok = df[df.frac_degenerate.fillna(1.0) < 0.5]
    g = (ok.groupby(["r", "rho"])
           .agg(pr_pos=("pr_pos", "median"), pr_pred=("pr_pred", "median"),
                MG=("MG", "median"), MG_sd=("MG", "std"),
                PRdelay=("PRdelay", "median"), n=("MG", "size")).reset_index())
    g.to_csv(OUT / "aniso_summary.csv", index=False)
    print("\n=== r is the manifold dimension and is fixed within a block; "
          "pr_pos falls with rho ===")
    print(g.round(2).to_string(index=False))

    lines = []
    for r, blk in g.groupby("r"):
        e_r = float(np.abs(blk.MG - r).mean())
        e_pr = float(np.abs(blk.MG - blk.pr_pos).mean())
        lines.append(f"  r={r}:  mean |MG - r| = {e_r:5.2f}   "
                     f"mean |MG - PR| = {e_pr:5.2f}   -> closer to "
                     f"{'the manifold dimension' if e_r < e_pr else 'the PR'}")
    # and over the whole grid
    e_r = float(np.abs(g.MG - g.r).mean())
    e_pr = float(np.abs(g.MG - g.pr_pos).mean())
    lines.append(f"  all  :  mean |MG - r| = {e_r:5.2f}   mean |MG - PR| = {e_pr:5.2f}")
    txt = "\n".join(lines)
    (OUT / "verdict.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n=== which target does MG follow? ===")
    print(txt)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

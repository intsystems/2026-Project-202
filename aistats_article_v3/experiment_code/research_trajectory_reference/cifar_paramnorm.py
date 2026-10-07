"""Supplement to E4 (not in the protocol): fair competitors on the parameter-norm log.

The protocol scored MG on the parameter norm but computed the simple competitors and the
IAAFT surrogate only for the two losses. After the primary analysis showed MG on the
parameter norm separating the events, this script computes the same competitors and the
surrogate ratio for that log from the saved logs, with the same windows and controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE))
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.surrogates import iaaft  # noqa: E402
from cifar_events import CFG, CFG2, W, S, simple  # noqa: E402

RES = HERE / "results_cifar"
HALF = 4000


def main():
    rows = []
    for f in sorted(RES.glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        x = np.load(f)["param_norm"]
        variants = {arm: x}
        if arm == "base":
            sc = x.copy(); sc[HALF:] *= 10
            sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[HALF:] = c[HALF:]
            variants.update({"scale": sc, "smooth": sm})
        for name, v in variants.items():
            for a in range(0, len(v) - W + 1, S):
                seg = v[a:a + W]
                rng = np.random.default_rng(a)
                row = {"arm": name, "seed": int(seed), "start": a,
                       "MG": estimate(seg, CFG).MG, "MG20": estimate(seg, CFG2).MG,
                       "MGs": float(np.median([estimate(iaaft(seg, rng=rng), CFG).MG for _ in range(3)]))}
                row.update(simple(seg))
                rows.append(row)
    d = pd.DataFrame(rows)
    d["rel"], d["ident"] = d.MG / d.MGs, d.MG20 / d.MG
    d.to_csv(RES / "paramnorm_windows.csv", index=False)
    pre = d[(d.start >= 2000) & (d.start + W <= HALF)]
    post = d[d.start >= HALF + W]
    cols = ["MG", "MGs", "rel", "ident", "crossings", "lag1", "det_std"]
    r = post.groupby(["arm", "seed"])[cols].median() / pre.groupby(["arm", "seed"])[cols].median()
    r.to_csv(RES / "paramnorm_ratios.csv")
    pd.set_option("display.width", 220)
    print(r.groupby("arm")[cols].agg(["median", "min", "max"]).round(2).to_string())
    print("\nlevels: MG/surr median", round(d.rel.median(), 3), " ident median", round(d.ident.median(), 3))
    rr = r.reset_index()
    real = rr[rr.arm.isin(["base", "batch_up", "lr_step", "freeze", "prune"])]
    lab = real.arm.isin(["lr_step", "freeze", "prune"])
    for c in cols:
        a, b = real[lab][c].values, real[~lab][c].values
        print(f"AUC lower-in-events {c:10s} {np.mean([(x < y) + .5 * (x == y) for x in a for y in b]):.2f}")


if __name__ == "__main__":
    main()

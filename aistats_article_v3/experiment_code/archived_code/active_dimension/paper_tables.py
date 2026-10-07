"""Regenerate the article's result tables from the committed CSVs, and check them.

Written because the audit found that ``tab:alts`` -- table 7 of ``icomp_v2/report.tex``, the
estimator against the alternatives -- existed only as numbers in the LaTeX source.  The
aggregation that produces it was in nobody's script.  In particular the twenty-direction
column of that table is **not** produced by the recipe that produces its neighbour column,
and the caption does not say so:

* the eight-direction column scores the four *withheld* ranks r in {1, 3, 5, 8} of the E2
  sweep, following the convention stated in the appendix D preamble;
* the twenty-direction column scores **all twenty** ranks of the k20 calibration, calibration
  ranks (2, 6, 10, 14, 18) and test ranks together.

Restricting the twenty-direction side to the ten ranks that ``tab:k20`` prints gives MG 1.554,
LB 1.313, TwoNN 3.311, PRdelay 1.833, roughness 8.790 -- which is not what the table says.  So
the two columns are not the same measurement at two ranges, and this script is the record of
what each one actually is.

Sources, both committed:

    results/e2_rank_sweep/sweep_raw.csv          eight directions  (12 observers x 4 seeds
                                                 x 7 ranks x 11 arms)
    results/k20_calibration/scores_frozen.csv    twenty directions (8 observers x 3 seeds
                                                 x 20 ranks x 7 arms)

Run:

    python paper_tables.py            # print the tables and check every cell
    python paper_tables.py --csv      # also write results/paper_tables/*.csv

Exit status is non-zero if any regenerated cell disagrees with the value printed in the
article beyond the tolerance implied by that value's own rounding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# ------------------------------------------------------------------ conventions
# tab:alts / tab:obs / fig:observers all drop these two, for the reasons appendix D gives:
# probe accuracy is quantised and degenerate in 100% of its windows, and the instantaneous
# mini-batch loss fails requirement 4 (it still reads 0.91-6.93 at zero learning rate).
DROP_OBS = ("acc_probe", "loss_step")

# The four ranks withheld from the E1 calibration.
HELD_OUT_R8 = (1, 3, 5, 8)

# The ten ranks tab:k20 prints.  NOT the ranks tab:alts scores -- see the module docstring.
K20_PRINTED_R = (1, 2, 4, 6, 8, 10, 12, 14, 17, 20)

STATS = ("MG", "LB", "TwoNN", "PRdelay", "specPR256", "specPR1024", "specPR0", "roughness")


def _score(med, truth):
    """MAE and Spearman of one median-collapsed series against the measured truth."""
    ok = np.isfinite(med) & np.isfinite(truth)
    if ok.sum() < 3:
        return np.nan, np.nan
    return (float(np.mean(np.abs(med[ok] - truth[ok]))),
            float(spearmanr(med[ok], truth[ok]).statistic))


def eight_direction_column():
    """tab:alts columns 2-3.  E2 sweep, recurrent arm, the four withheld ranks.

    One value per rank: the median over 4 seeds x 10 observers, exactly the aggregation the
    appendix D preamble describes ("the error of the single series left after taking the
    median over seeds and observers at each rank").  Keeping all twelve observers instead
    gives MG 0.455 and TwoNN 1.752, so the two exclusions are load-bearing.
    """
    d = pd.read_csv(RES / "e2_rank_sweep" / "sweep_raw.csv")
    d = d[(d.arm == "qp") & (~d.eta_zero) & (~d.observer.isin(DROP_OBS))]
    d = d[d.r.isin(HELD_OUT_R8)]
    g = d.groupby("r")
    truth = g.traj_PR.median().values
    return {s: _score(g[s].median().values, truth) for s in STATS}, truth, sorted(HELD_OUT_R8)


def twenty_direction_column():
    """tab:alts columns 4-5.  k20 calibration, recurrent arm, ALL TWENTY ranks.

    Note the asymmetry with the eight-direction column: there is no held-out restriction
    here, so five of the twenty ranks (2, 6, 10, 14, 18) are the ranks the k20 estimator
    configuration was itself selected on.  That is what reproduces the printed numbers.
    """
    d = pd.read_csv(RES / "k20_calibration" / "scores_frozen.csv")
    d = d[d.tag == "qp"]
    g = d.groupby("r")
    truth = g.traj_pr.median().values
    return {s: _score(g[s].median().values, truth) for s in STATS}, truth, sorted(d.r.unique())


def table_k20():
    """tab:k20.  Median over three seeds and eight observers at the ten printed ranks.

    The "spectral PR" row is specPR256, not the native-resolution specPR0.  The last row is
    the spread of MG across the three seeds at fixed rank, taken as the median over observers
    of (max - min).
    """
    d = pd.read_csv(RES / "k20_calibration" / "scores_frozen.csv")
    d = d[(d.tag == "qp") & d.r.isin(K20_PRINTED_R)]
    g = d.groupby("r")
    out = {"MG": g.MG.median(), "PRdelay": g.PRdelay.median(),
           "specPR256": g.specPR256.median()}
    spread = (d.groupby(["observer", "r"]).MG.agg(lambda s: s.max() - s.min())
               .groupby("r").median())
    out["spread across seeds"] = spread
    return pd.DataFrame(out).T[list(K20_PRINTED_R)]


# ------------------------------------------------------------------ what the article prints
PUBLISHED_ALTS = {                    # statistic: (mae8, rho8, mae20, rho20)
    "MG":         (0.457, 1.00, 1.62, 0.97),
    "LB":         (0.44,  1.00, 1.36, 0.98),
    "TwoNN":      (1.81,  1.00, 3.48, 0.89),
    "PRdelay":    (0.89,  1.00, 2.19, 0.98),
    "specPR256":  (1.60,  1.00, 5.29, 0.99),
    "specPR1024": (1.35,  1.00, 4.47, 0.99),
    "specPR0":    (1.03,  0.80, 2.78, 0.93),
    "roughness":  (3.65,  0.20, 9.90, -0.57),
}

PUBLISHED_K20 = {
    "MG":                 [0.89, 2.35, 4.41, 6.66, 7.38, 8.99, 10.61, 11.40, 13.50, 15.09],
    "PRdelay":            [2.00, 2.29, 4.54, 6.81, 7.95, 9.20, 10.79, 12.15, 11.71, 13.51],
    "specPR256":          [1.01, 1.25, 2.58, 3.68, 4.18, 5.48, 5.98, 6.53, 8.11, 8.38],
    "spread across seeds": [0.00, 0.20, 1.49, 2.15, 2.11, 2.95, 1.50, 2.84, 2.76, 3.33],
}


def _ulp(printed):
    """One unit in the last printed decimal place."""
    s = f"{printed:.10f}".rstrip("0")
    dp = len(s.split(".")[1]) if "." in s else 0
    return 10.0 ** (-dp)


def _classify(got, printed):
    """'' if the cell rounds correctly, 'round' if it is off by <=1 ulp, 'DATA' otherwise.

    The distinction matters.  A 'round' cell means the article's printed digits are not what
    the committed CSV rounds to, but the underlying measurement agrees -- a typesetting
    defect.  A 'DATA' cell means the number cannot be reproduced at all.
    """
    if not np.isfinite(got):
        return "DATA"
    u = _ulp(printed)
    d = abs(got - printed)
    if d <= 0.5 * u + 1e-9:
        return ""
    return "round" if d <= u + 1e-9 else "DATA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write results/paper_tables/")
    args = ap.parse_args()

    c8, truth8, r8 = eight_direction_column()
    c20, truth20, r20 = twenty_direction_column()
    k20 = table_k20()

    print(f"eight directions : {len(r8)} ranks {r8}, truth {np.round(truth8, 3).tolist()}")
    print(f"twenty directions: {len(r20)} ranks {r20[0]}..{r20[-1]}")
    print()
    print("=== tab:alts (table 7) ===")
    print(f"{'statistic':12s} {'MAE8':>8s} {'rho8':>7s} {'MAE20':>8s} {'rho20':>7s}   "
          f"{'published':>28s}")
    bad, rounding = [], []
    for s in STATS:
        (m8, p8), (m20, p20) = c8[s], c20[s]
        pub = PUBLISHED_ALTS[s]
        got = (m8, p8, m20, p20)
        flags = ""
        for name, g, p in zip(("MAE8", "rho8", "MAE20", "rho20"), got, pub):
            kind = _classify(g, p)
            if kind:
                flags += f" {name}:{kind}"
                (bad if kind == "DATA" else rounding).append((s, name, g, p))
        print(f"{s:12s} {m8:8.3f} {p8:+7.3f} {m20:8.3f} {p20:+7.3f}   "
              f"{pub[0]:6.3f} {pub[1]:+5.2f} {pub[2]:5.2f} {pub[3]:+5.2f}{flags}")

    print()
    print("=== tab:k20 (table 6) ===")
    print(k20.round(2).to_string())
    for row, pub in PUBLISHED_K20.items():
        got = k20.loc[row].values
        for r, g, p in zip(K20_PRINTED_R, got, pub):
            kind = _classify(float(g), p)
            if kind:
                (bad if kind == "DATA" else rounding).append(
                    (f"tab:k20 {row}", f"r={r}", float(g), p))

    if args.csv:
        out = RES / "paper_tables"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({s: dict(zip(("mae_8dir", "rho_8dir"), c8[s])) |
                         dict(zip(("mae_20dir", "rho_20dir"), c20[s])) for s in STATS}) \
          .T.to_csv(out / "tab_alts.csv")
        k20.to_csv(out / "tab_k20.csv")
        print(f"\nwrote {out}")

    print()
    if rounding:
        print(f"{len(rounding)} cell(s) differ only in the last printed digit "
              f"(measurement agrees; the article's rounding does not):")
        for what, cell, g, p in rounding:
            print(f"  {what:22s} {cell:8s} regenerated {g:.4f} -> should print "
                  f"{round(g, len(f'{p:.10f}'.rstrip('0').split('.')[1]))}, article prints {p}")
        print()
    if bad:
        print(f"MISMATCH in {len(bad)} cell(s) -- these do not reproduce at all:")
        for what, cell, g, p in bad:
            print(f"  {what:22s} {cell:8s} regenerated {g:.4f}  printed {p}")
        return 1
    print("every cell reproduces from the committed CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

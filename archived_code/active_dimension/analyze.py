"""Turn the raw CSVs into the tables the report quotes.  Run after the experiments.

Every table scores MG against the **measured** active dimension, and every table carries the
nulls beside it, because the question is never "does MG correlate with r" -- a compressive
monotone function of r correlates with r -- but "does MG carry information that the linear
participation ratio, the roughness, and a same-spectrum surrogate do not".

    python analyze.py > results/tables.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mg as MG
from runner import mae, spearman

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TEST_R = (1, 3, 5, 8)          # disjoint from the calibration r = {2, 4, 6}
pd.set_option("display.width", 200)


def _load(p):
    q = RES / p
    return pd.read_csv(q) if q.exists() else None


def table_e2_observers():
    """Per arm x observer: does MG track the active dimension, and does anything else?"""
    df = _load("e2_rank_sweep/sweep_raw.csv")
    if df is None:
        return None
    out = []
    for (arm, obs), g in df.groupby(["arm", "observer"]):
        if g["flat"].all():
            continue
        m = g.groupby("r").agg(
            MG=("MG", "median"), truth=("traj_PR", "median"), rough=("roughness", "median"),
            prd=("PRdelay", "median"), spc=("specPR0", "median"),
            spc256=("specPR256", "median"),
            ident=("ident_ratio", "median"), lb=("LB", "median"),
            deg=("frac_degenerate", "mean"), sd=("MG", "std")).reset_index()
        if m.truth.nunique() < 3:
            continue
        out.append(dict(
            arm=arm, observer=obs, family=g.family.iloc[0],
            rho_MG=spearman(m.MG, m.truth),
            rho_rough=spearman(m.rough, m.truth),
            rho_PRdelay=spearman(m.prd, m.truth),
            rho_specPR=spearman(m.spc, m.truth),
            mae_specPR=mae(m.spc, m.truth),
            rho_specPR256=spearman(m.spc256, m.truth),
            slope=np.polyfit(m.truth, m.MG, 1)[0] if m.MG.notna().sum() > 2 else np.nan,
            mae_raw=mae(m.MG, m.truth),

            ident=float(np.nanmedian(m.ident)),
            seed_sd=float(np.nanmean(m.sd)),
            degen=float(m.deg.mean())))
    return pd.DataFrame(out)


def table_e2_calibrated():
    """Absolute recovery, with the calibration fitted on a split disjoint in seed AND r."""
    df = _load("e2_rank_sweep/sweep_raw.csv")
    cal = _load("e1_calibration/calibration_raw.csv")
    if df is None or cal is None:
        return None
    # NOT `cal.cfg_id.mode()`: every cfg_id appears the same number of times, so the mode is
    # a 108-way tie and `.iloc[0]` silently returns config 0 (max_E=10, theiler=0) -- the very
    # setting this package exists to avoid.  Take the frozen id.
    from e1_calibration import load_frozen
    _, meta = load_frozen()
    cal = cal[cal.cfg_id == int(meta["cfg_id"])]
    out = []
    for (arm, obs), g in df.groupby(["arm", "observer"]):
        c = cal[cal.observer == obs]
        if len(c) < 3 or g["flat"].all():
            continue
        for kind in ("isotonic", "affine", "identity"):
            try:
                f = MG.Calibration(kind).fit(c.MG.values, c.traj_PR.values)
            except (ValueError, np.linalg.LinAlgError):
                continue
            t = g[g.r.isin(TEST_R)]
            if not len(t):
                continue
            pred = f.predict(t.MG.values)
            out.append(dict(arm=arm, observer=obs, calibration=kind,
                            mae=mae(pred, t.traj_PR.values),
                            bias=float(np.nanmean(pred - t.traj_PR.values)),
                            n=int(np.isfinite(pred).sum())))
    return pd.DataFrame(out)


def main():
    print("=" * 100)
    print("E0 -- identifiability atlas")
    a = _load("e0_atlas/atlas_raw.csv")
    if a is not None:
        d = a[(a.max_E == 20) & (a.observer == "generic")]
        print("\nMG vs r  (rows: family x scale parameter)")
        print(d.pivot_table(index=["family", "tau_c", "N"], columns="r", values="MG",
                            dropna=False).round(2).to_string())
        i = _load("e0_atlas/identifiability_ratio.csv")
        if i is not None:
            print("\nidentifiability ratio MG(2E)/MG(E)  -- ~1 identifiable, >1.15 not")
            print(i.pivot_table(index="family", columns="r", values="ident_ratio")
                  .round(2).to_string())

    print("\n" + "=" * 100)
    print("E1 -- frozen estimator configuration")
    r = _load("e1_calibration/config_ranking.csv")
    if r is not None:
        print(r.head(8)[["cfg_id", "max_E", "tau", "k_neighbors", "theiler", "window",
                         "mae", "rho", "sd", "score"]].round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("E2 -- MG against the MEASURED active dimension, with every null beside it")
    t = table_e2_observers()
    if t is not None:
        t.to_csv(RES / "e2_rank_sweep" / "observer_scores.csv", index=False)
        for arm, g in t.groupby("arm"):
            print(f"\n--- arm: {arm} ---")
            print(g.sort_values("rho_MG", ascending=False)
                  [["observer", "family", "rho_MG", "rho_rough", "rho_PRdelay", "rho_specPR",
                    "slope", "mae_raw", "mae_specPR", "ident", "seed_sd", "degen"]]
                  .round(3).to_string(index=False))
        print("\n--- arm summary (median over observers that are not flat) ---")
        print(t.groupby("arm")[["rho_MG", "rho_rough", "rho_PRdelay", "rho_specPR", "slope",
                                "mae_raw", "mae_specPR", "ident"]].median().round(3).to_string())

    c = table_e2_calibrated()
    if c is not None:
        c.to_csv(RES / "e2_rank_sweep" / "calibrated_mae.csv", index=False)
        print("\n--- absolute recovery on r in {1,3,5,8}, calibration fitted on "
              "seeds 90-92 and r in {2,4,6} ---")
        print(c.pivot_table(index=["arm", "calibration"], values=["mae", "bias"],
                            aggfunc="median").round(3).to_string())

    print("\n" + "=" * 100)
    print("E3 -- controlled simplification")
    tr = _load("e3_transitions/transitions_raw.csv")
    if tr is not None:
        tr["seen_drop"] = tr.level0 - tr.level1
        tr["true_drop"] = tr.truth0 - tr.truth1
        g = tr.groupby(["mode", "observer"]).agg(
            true_drop=("true_drop", "median"), seen_drop=("seen_drop", "median"),
            lvl_err0=("level0", lambda s: np.nan), lag_down=("lag_down", "median"),
            lag_up=("lag_up", "median"), n=("seed", "size")).reset_index()
        g["lvl_err0"] = (tr.groupby(["mode", "observer"])
                         .apply(lambda s: float(np.nanmean(np.abs(s.level0 - s.truth0))),
                                include_groups=False).values)
        print(g.round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("E4 -- false alarms when the number of active directions does NOT change")
    cs = _load("e4_controls/controls_scored.csv")
    if cs is not None:
        print("\nfalse-alarm rate (fraction of run x observer cells that fire)")
        print(cs.pivot_table(index="control", columns="mode", values="fires",
                             aggfunc="mean").round(3).to_string())
        print("\nmedian |MG shift| vs the baseline level")
        print(cs.pivot_table(index="control", columns="mode", values="between",
                             aggfunc="median").round(3).to_string())

    print("\n" + "=" * 100)
    print("E5 -- this project's own 120k-step logs, on the atlas axes")
    s = _load("e5_real_logs/real_logs_summary.csv")
    if s is not None:
        print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

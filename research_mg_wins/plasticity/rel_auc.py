"""EXPLORATORY (post hoc, after the main test): detection AUC of the RELATIVE change of each
monitor against its own run's early level, r_j = v_j / median(v at the first 2 available
tasks) - 1, same target as eval_s1.detection_auc. Sign chosen on calibration runs."""
import numpy as np, pandas as pd
import eval_s1 as E
out = []
for fam in ("F1", "F2"):
    df = E.load(fam)
    res = {}
    for split, conds in (("cal", E.CAL_CONDS), ("test", E.CAL_CONDS + E.UNSEEN)):
        pools, mons = E.runs_of(df, split, conds)
        rel = {}
        for c, pool in pools.items():
            newpool = []
            for acc, vals, s in pool:
                nv = {}
                for m, v in vals.items():
                    ok = np.flatnonzero(np.isfinite(v))
                    ref = np.median(v[ok[:2]]) if len(ok) >= 2 else np.nan
                    nv[m] = v / ref - 1 if np.isfinite(ref) and ref != 0 else v * np.nan
                    nv[m][ok[:2]] = np.nan
                newpool.append((acc, nv, s))
            rel[c] = newpool
        res[split] = E.detection_auc(rel, mons)[0]
    for m in mons:
        a, t = res["cal"][m], res["test"][m]
        s = 1 if a >= 0.5 else -1
        out.append({"fam": fam, "monitor": m, "stat": E.stat_of(m), "auc_cal": max(a, 1 - a),
                    "auc_test": t if s == 1 else 1 - t})
o = pd.DataFrame(out)
o.to_csv("results/rel_detection_auc.csv", index=False)
best = o.sort_values("auc_cal", ascending=False).groupby(["fam", "stat"]).head(1)
for fam in ("F1", "F2"):
    print(fam); print(best[best.fam == fam].sort_values("auc_test", ascending=False)[["stat", "monitor", "auc_cal", "auc_test"]].round(3).to_string(index=False))
print(o[o.monitor.str.startswith("MG|pnorm")].round(3).to_string(index=False))

"""Fresh-seed confirmation of T1 (seeds 20, 21). Every selector FROZEN as calibrated in
results_main/T1_config_ranking.csv (log + sign chosen on calibration seeds 0, 1; for MG:
grad_norm, sign +). No re-calibration. Reports test Spearman, regret@1/@3 and a paired
bootstrap over configurations of rho(MG) - rho(other), on seeds 20-21 alone and pooled with
test seeds 10-12.

usage: python confirm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

frozen = pd.read_csv(HERE / "results_main" / "T1_config_ranking.csv")
frozen = frozen[frozen.col.notna()][["stat", "col", "sign"]]
meta, ev, feats = A.load(HERE / "runs_confirm", HERE / "feats_confirm")
new = A.table(meta, ev, feats, A.FINAL)
old = pd.read_csv(HERE / "results_main" / "test_final.csv")
out = HERE / "results_confirm"
out.mkdir(exist_ok=True)
new.to_csv(out / "confirm_final.csv", index=False)
pooled = pd.concat([old, new[old.columns.intersection(new.columns)]], ignore_index=True)
print("new seeds:", sorted(new.seed.unique()), "runs:", len(new), "bad:", int(new.bad.sum()))

mg_col, mg_sign = frozen.set_index("stat").loc["MG", ["col", "sign"]]
res = []
for label, df in (("new_20_21", new), ("pooled_10_12_20_21", pooled)):
    for _, r in frozen.iterrows():
        m = A.sel_metrics(df, r.col, r.sign)
        row = {"set": label, "stat": r.stat, "col": r.col, "sign": r.sign, "rho": m.rho.mean(),
               "rho_sd": m.rho.std(), "regret1": m.regret1.mean(), "regret3": m.regret3.mean(),
               "n_seeds": len(m)}
        if r.stat != "MG":
            p, lo, hi = A.boot_diff(df, mg_col, mg_sign, r.col, r.sign, n=1000)
            row.update({"P_MG_better": p, "diff_lo": lo, "diff_hi": hi})
        res.append(row)
    rr = []
    for s, g in df.groupby("seed"):
        rr.append(g.acc.max() - g.acc.mean())
    res.append({"set": label, "stat": "random_choice", "rho": 0.0, "regret1": float(np.mean(rr))})
res = pd.DataFrame(res)
res.to_csv(out / "T1_confirm.csv", index=False)
pd.set_option("display.width", 220)
for label, g in res.groupby("set", sort=False):
    print("===", label)
    g = g.copy()
    g["col"] = g["col"].astype(str).str.split("|").str[-1]
    print(g.drop(columns="set").sort_values("rho", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

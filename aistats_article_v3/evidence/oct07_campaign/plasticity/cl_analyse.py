"""Closed-loop confirmation summary (fresh seeds 20, 21; real resets)."""
import glob, json, numpy as np, pandas as pd
from eval_s1 import boot_ci, CAL_CONDS
rows = [json.load(open(f)) for f in glob.glob("closed_loop/F*_s*_*.json")]
d = pd.DataFrame(rows)
out = []
for fam, g in d.groupby("fam"):
    w = g.pivot_table(index=["cond", "seed"], columns="policy", values="util")
    r = g.pivot_table(index=["cond", "seed"], columns="policy", values="resets")
    w = w.dropna()
    for p in w.columns:
        dd = (w["MG"] - w[p]).to_numpy()
        lo, hi = boot_ci(dd) if p != "MG" else (0, 0)
        seen = w.index.get_level_values(0).isin(CAL_CONDS)
        out.append({"fam": fam, "policy": p, "util": w[p].mean(), "util_seen": w[p][seen].mean(),
                    "util_unseen": w[p][~seen].mean(), "resets": r.loc[w.index, p].mean(),
                    "MG_minus": dd.mean(), "ci_lo": lo, "ci_hi": hi, "MG_wins": int((dd > 1e-9).sum()),
                    "MG_losses": int((dd < -1e-9).sum()), "n": len(dd)})
o = pd.DataFrame(out).sort_values(["fam", "util"], ascending=[True, False])
o.to_csv("results/closed_loop_summary.csv", index=False)
pd.set_option("display.width", 220)
print(o.round(4).to_string(index=False))
print("seconds per run (mean):", d.groupby("policy").seconds.mean().round(0).to_dict())

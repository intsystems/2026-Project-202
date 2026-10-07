"""Head-to-head on the SAME log and window slot: MG vs each scalar statistic (decision utility
on test with each monitor's own calibrated rule, and detection AUC on test)."""
import pandas as pd, numpy as np
rows = []
for fam in ("F1", "F2"):
    a = pd.read_csv(f"results/{fam}_all_monitors.csv")
    d = pd.read_csv(f"results/{fam}_detection_auc.csv").set_index("monitor")
    a = a[a.monitor.str.count(r"\|") == 2]
    for (L, w), g in a.groupby(["log", "window"]):
        g = g.set_index("stat")
        mg_u = g.loc["MG", "test_util"]
        mg_auc = d.loc[f"MG|{L}|{w}", "auc_test"]
        comp = g.drop(index=["MG", "MG_t4"])
        aucs = np.array([d.loc[f"{s}|{L}|{w}", "auc_test"] for s in comp.index])
        rows.append({"fam": fam, "log": L, "window": w, "MG_util": mg_u,
                     "best_comp_util": comp.test_util.max(), "best_comp": comp.test_util.idxmax(),
                     "MG_util_rank": int((comp.test_util > mg_u).sum()) + 1,
                     "MG_auc": mg_auc, "best_comp_auc": aucs.max(), "best_comp_a": comp.index[aucs.argmax()],
                     "MG_auc_rank": int((aucs > mg_auc).sum()) + 1, "n_comp": len(comp)})
t = pd.DataFrame(rows)
t.to_csv("results/slot_head_to_head.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))

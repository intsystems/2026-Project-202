"""Competitors on the saved E9 records (neuron 0, four windows of 8192, median per record)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import ALL

R = Path(__file__).resolve().parents[1] / "research_generator" / "results"
DIM = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "H2": 1, "H4": 1, "M4": 2}
rows = []
with threadpool_limits(4):
    for f in sorted(R.glob("obs_*_s*.npz")):
        arm, seed = f.stem[4:].rsplit("_s", 1)
        if arm not in DIM:
            continue
        x = np.load(f)["obs"][:, 0]
        row = {"arm": arm, "seed": int(seed), "d": DIM[arm]}
        for name, fn in ALL.items():
            if name == "MG":
                continue
            row[name] = float(np.median([fn(x[a:a + 8192]) for a in range(0, 32768, 8192)]))
        rows.append(row)
        print(arm, seed, flush=True)
df = pd.DataFrame(rows)
mg = pd.DataFrame([r for f in R.glob("runs_s*.json") for r in json.load(open(f))])[["arm", "seed", "MG_n0"]]
df = df.merge(mg.rename(columns={"MG_n0": "MG"}), on=["arm", "seed"])
df.to_csv(Path(__file__).parent / "e9_competitors.csv", index=False)
out = []
for c in [c for c in df.columns if c not in ("arm", "seed", "d")]:
    w = df.pivot(index="seed", columns="arm", values=c)
    out.append({"stat": c, "spearman_d": spearmanr(df[c], df.d)[0],
                "H4<M4<T4": int(((w.H4 < w.M4) & (w.M4 < w.T4)).sum()),
                "H2<T2": int((w.H2 < w.T2).sum()),
                "H4<T2": int((w.H4 < w.T2).sum()),
                "M4<T3": int((w.M4 < w.T3).sum())})
print(pd.DataFrame(out).round(3).to_string())
print(df.groupby("arm")[[c for c in df.columns if c not in ("arm", "seed", "d")]].median().round(3).to_string())

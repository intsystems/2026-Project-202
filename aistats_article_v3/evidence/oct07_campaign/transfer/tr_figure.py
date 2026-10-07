"""Figure: Youden J (hit rate - null false-alarm rate) per monitor and setup, rules frozen on S."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
J = pd.read_csv(HERE / "results" / "table_J.csv", index_col=0)
J = J[[c for c in J.columns if not c.startswith("J_T0")]]
J = J[~J.index.str.endswith("_dt")].sort_values("J_T1..T9", ascending=False)
fig, ax = plt.subplots(figsize=(1.0 + 0.62 * J.shape[1], 0.32 * J.shape[0] + 1.4))
im = ax.imshow(J.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(J.shape[1])); ax.set_xticklabels(J.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(J.shape[0])); ax.set_yticklabels(J.index, fontsize=8)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        v = J.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "black")
ax.set_title("Youden J = hit rate - null false-alarm rate (rules calibrated on source only)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03)
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_J_heatmap.png", dpi=130)
print("saved")

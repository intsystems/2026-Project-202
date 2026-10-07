"""Pilot inspection: loss / internals curves and spike onsets (truth.py). No MG here."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import truth
d = HERE / (sys.argv[1] if len(sys.argv) > 1 else "pilot")
files = sorted(d.glob("*.npz"))
keys = ("loss", "grad_norm", "update_norm", "attn_max", "attn_ent", "logz")
fig, ax = plt.subplots(len(files), len(keys), figsize=(4 * len(keys), 2.2 * len(files)), squeeze=False)
for i, f in enumerate(files):
    r = np.load(f)
    sp = truth.spikes(r["loss"])
    n = int(np.isfinite(r["loss"]).sum())
    fv = r["val_loss"][-1] if len(r["val_loss"]) else np.nan
    print(f.stem, "steps", n, "div", int(r["diverged_at"]), "final val", round(float(fv), 3),
          "spikes", sp, "sec", round(float(r["seconds"])))
    for j, k in enumerate(keys):
        a = ax[i, j]
        y = r[k]
        a.plot(y, lw=0.4)
        if k in ("grad_norm", "update_norm", "attn_max"):
            a.set_yscale("log")
        if k == "loss":
            a.set_ylim(np.nanmin(y) * 0.9, min(np.nanmax(y), 5))
        for s, e in sp:
            a.axvline(s, color="r", lw=0.6)
        a.set_title(f"{f.stem[2:]} {k}", fontsize=7)
fig.tight_layout()
fig.savefig(d / "pilot_curves.png", dpi=70)

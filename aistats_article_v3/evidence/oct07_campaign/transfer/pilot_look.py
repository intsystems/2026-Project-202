import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
for f in sorted((HERE / "pilot").glob("*.npz")):
    r = np.load(f); m = json.load(open(f.with_suffix(".json")))
    t = r["truth_t"]; te = m["t_e"] or 5000
    def med(k, a, b):
        s = (t > a) & (t <= b); return np.median(r[f"truth_{k}"][s])
    print(f.stem, "P", m["P"], "t_e", m["t_e"], "t_r", m["t_r"], "acc %.3f wall %.0f" % (m["test_acc"], m["wall_s"]))
    for k in ("frac_moving", "upr", "erank", "srank", "dormant", "probe_acc"):
        print("   %-12s pre %.4g  post(+0..1000) %.4g  post(+1000..2000) %.4g  end %.4g" % (k, med(k, te-1000, te), med(k, te, te+1000), med(k, te+1000, te+2000), med(k, 9000, 10000)))
    pn = r["param_norm_raw"]
    print("   pn at 1000,te-1,te+1,te+1000,end: ", np.round(pn[[1000, te-1, te+1, min(te+1000, 9999), -1]], 4), " loss med pre/post", np.median(r["batch_loss"][te-1000:te]).round(4), np.median(r["batch_loss"][te:te+1000]).round(4))

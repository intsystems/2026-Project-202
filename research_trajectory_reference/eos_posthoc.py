"""Post-hoc (NOT pre-registered) variants of the cheap statistic on the EoS logs.

At the edge of stability the full-batch loss flips every step (period-2), and the
pre-registered MG (tau=1) ran opposite to the Hessian mode count. Tested here: remove
the flip before embedding. Every variant is reported; none was chosen by its result.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
from actdim.estimator.surrogates import iaaft

RES = Path(__file__).resolve().parent / "results_eos"
W, S, BURN = 1000, 500, 2000
base = EstimatorConfig(max_E=10, tau=1, k_neighbors=10, theiler="embedding")

def detr(x):
    t = np.arange(len(x)); return x - np.polyval(np.polyfit(t, x, 1), t)

VARIANTS = {
    "tau1 (pre-registered)": lambda s: (s, base),
    "tau2": lambda s: (s, base.replace(tau=2)),
    "even steps only": lambda s: (s[::2], base),
    "odd steps only": lambda s: (s[1::2], base),
    "flip-demodulated (-1)^t": lambda s: (detr(s) * (-1.0) ** np.arange(len(s)), base),
    "pair mean (x_t+x_t+1)/2": lambda s: ((s[:-1:2] + s[1::2]) / 2, base),
    "adaptive lag": lambda s: (s, base.replace(tau="acorr")),
}
ck = pd.read_csv(RES / "checkpoints_all.csv")
rows = []
for f in sorted(RES.glob("log_sweep_*.npz")):
    tag = f.stem[len("log_sweep_"):]
    e, seed = tag.split("_s"); eta = float(e.split("-")[0]); seed = int(seed)
    x = np.load(f)["loss"]
    for a in range(BURN, len(x) - W + 1, S):
        seg = x[a:a + W]
        c = ck[(ck.stage == "sweep") & (ck.eta0 == eta) & (ck.seed == seed) & (ck.step >= a) & (ck.step < a + W)]
        row = {"eta": eta, "seed": seed, "start": a, "n_unstable": c.n_unstable.median()}
        for name, fn in VARIANTS.items():
            y, cfg = fn(seg)
            row[name] = estimate(y, cfg).MG
            row[name + "|surr"] = np.median([estimate(iaaft(y, rng=np.random.default_rng(i)), cfg).MG for i in range(2)])
        rows.append(row)
d = pd.DataFrame(rows); d.to_csv(RES / "posthoc.csv", index=False)
lvl = d.groupby(["eta", "seed"]).median(numeric_only=True).reset_index()
eos = lvl[lvl.eta >= 0.1]
out = []
for name in VARIANTS:
    out.append({"variant": name,
                "rho all runs": spearmanr(lvl.n_unstable, lvl[name])[0],
                "rho EoS runs": spearmanr(eos.n_unstable, eos[name])[0],
                "rho windows": spearmanr(d.n_unstable, d[name])[0],
                "MG/surr": (d[name] / d[name + "|surr"]).median(),
                **{f"eta {e:g}": v for e, v in lvl.groupby("eta")[name].median().items()}})
o = pd.DataFrame(out).round(2); o.to_csv(RES / "posthoc_summary.csv", index=False)
pd.set_option("display.width", 250); print(o.to_string(index=False))

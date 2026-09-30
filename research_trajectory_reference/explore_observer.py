"""Exploration on the E4 logs (seeds 0-3), used ONLY to choose E5's observer and window.

E5 then runs on fresh seeds with the choice frozen in its protocol, so this search
cannot leak into E5's confirmatory numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

RES = HERE / "results_cifar"
HALF, EVENTS, CTRL = 4000, ("lr_step", "freeze", "prune"), ("base", "batch_up")


def detrend(x):
    t = np.arange(len(x)); return x - np.polyval(np.polyfit(t, x, 2), t)


TRANSFORMS = {
    "norm": lambda x: x,
    "diff": lambda x: np.diff(x),
    "norm_sq": lambda x: x ** 2,
}
CONFIGS = {f"E{E}_t{t}_W{W}": (EstimatorConfig(max_E=E, tau=t, k_neighbors=10, theiler="embedding"), W)
           for E in (10,) for t in (1, 4) for W in (500, 1000)}
CONFIGS["E20_t1_W1000"] = (EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding"), 1000)


def main():
    rows = []
    for f in sorted(RES.glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        x0 = np.load(f)["param_norm"]
        for tn, tf in TRANSFORMS.items():
            x = tf(x0)
            for cn, (cfg, W) in CONFIGS.items():
                for a in range(0, len(x) - W + 1, W // 2):
                    rows.append({"arm": arm, "seed": int(seed), "obs": tn, "cfg": cn, "start": a, "W": W,
                                 "MG": estimate(x[a:a + W], cfg).MG})
    d = pd.DataFrame(rows)
    out = []
    for (tn, cn), g in d.groupby(["obs", "cfg"]):
        W = g.W.iloc[0]
        pre = g[(g.start >= HALF - 2000) & (g.start + W <= HALF)].groupby(["arm", "seed"]).MG.median()
        post = g[g.start >= HALF + 500].groupby(["arm", "seed"]).MG.median()
        late = g[g.start >= HALF + 2000].groupby(["arm", "seed"]).MG.median()
        for per, p in (("post", post), ("late", late)):
            r = (p / pre).reset_index(name="r")
            ev, ct = r[r.arm.isin(EVENTS)].r, r[r.arm.isin(CTRL)].r
            out.append({"obs": tn, "cfg": cn, "period": per,
                        "ev_median_%": 100 * (ev.median() - 1), "ctrl_median_%": 100 * (ct.median() - 1),
                        "gap": ct.min() - ev.max(),
                        "auc": np.mean([(a < b) + .5 * (a == b) for a in ev for b in ct])})
    o = pd.DataFrame(out).sort_values(["auc", "gap"], ascending=False).round(3)
    o.to_csv(RES / "explore_observer.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()

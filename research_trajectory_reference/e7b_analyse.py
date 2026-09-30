"""E7b: other observers on the E7 ResNet-18 runs, analysed locally.

PROTOCOL (written before observers_*.npz was looked at).

Why. In E7 the parameter-norm log of ResNet-18 (11 M parameters) is a smooth
monotone curve: two trend crossings per window, detrended std ~ 0. On such a curve
MG sits at the value the configuration gives a smooth transient (~5) and cannot
respond; only freezing the whole network but the head, which changes the shape of the
curve itself, moved it. A norm over 11 M coordinates averages every fluctuation away.

Observers tested (all free to log, none needs a forward pass):
  primary     proj0 -- one fixed sparse random projection of theta_t - theta_0
              (running sum of CountSketch coordinate 0 of the updates), chosen as
              "the first coordinate" before looking;
  secondary   the median over the 16 projections of each window's MG;
              norms of fc, layer4, stem (reported, not tested).
Everything else is frozen from E5/E6: MG E=20, tau=1, k=20, Theiler = embedding span,
windows 1 000 / stride 500; before = windows inside [2 000, 4 000), after = windows
starting in [5 000, 9 000]; detector rule from results_detector/chosen_rules.json.
Every window also gets the trend-crossing count, so smooth-curve windows are visible.

Predictions, same as E7: strong events (lr10, lr100, freeze_head, prune80, prune95)
drop MG below -8.8 %, controls (base, batch_up, scale, smooth) stay above -5 %; the
detector catches >= 75 % of strong-event runs with <= 1 false alarm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "code"))
sys.path.insert(0, str(HERE / "colab_resnet"))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402
from analyse_resnet import drop_series  # noqa: E402

CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
W, S, EVENT = 1000, 500, 4000
ROOT = HERE / "results_resnet"
STRONG = ["lr10", "lr100", "freeze_head", "prune80", "prune95"]
NULL = ["base", "batch_up", "scale", "smooth"]
OBS = ["proj0"] + [f"proj{j}" for j in range(1, 16)] + ["norm_fc", "norm_layer4", "norm_stem", "param_norm"]


def windows(sc):
    z = np.load(ROOT / f"observers_{sc}.npz")
    runs = sorted({k.split("|")[0] for k in z.files})
    rows = []
    for run in runs:
        arm, seed = run.rsplit("_s", 1)
        for o in OBS:
            x = z[f"{run}|{o}"]
            variants = {arm: x}
            if arm == "base":
                sc_ = x.copy(); sc_[EVENT:] *= 10
                sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[EVENT:] = c[EVENT:]
                variants.update({"scale": sc_, "smooth": sm})
            for name, v in variants.items():
                for a in range(0, len(v) - W + 1, S):
                    seg = v[a:a + W]
                    rows.append({"scenario": sc, "arm": name, "seed": int(seed), "obs": o, "start": a,
                                 "end": a + W, "MG": estimate(seg, CFG).MG,
                                 "crossings": trend_crossings(seg)})
    return pd.DataFrame(rows)


def main():
    rule = json.load(open(HERE / "results_detector" / "chosen_rules.json"))["MG"]
    cache = ROOT / "e7b_windows.csv"
    w = pd.read_csv(cache) if cache.exists() else pd.concat([windows(s) for s in ("scratch", "finetune")])
    w.to_csv(cache, index=False)
    # the secondary observer: per-window median over the 16 projections
    pm = (w[w.obs.str.startswith("proj")].groupby(["scenario", "arm", "seed", "start", "end"])
          [["MG", "crossings"]].median().reset_index())
    pm["obs"] = "proj_median16"
    w = pd.concat([w, pm])
    out = []
    for (sc, o), g in w.groupby(["scenario", "obs"]):
        pre = g[(g.start >= 2000) & (g.end <= EVENT)].groupby(["arm", "seed"]).MG.median()
        post = g[(g.start >= 5000) & (g.start <= 9000)].groupby(["arm", "seed"]).MG.median()
        chg = (100 * (post / pre - 1)).rename("chg").reset_index()
        hits, fas = [], []
        for (arm, seed), gg in g.groupby(["arm", "seed"]):
            t = next((e for e, d in drop_series(gg, "MG", rule["M"], rule["B"], 1)
                      if np.isfinite(d) and d < -rule["delta"]), None)
            if arm in NULL:
                fas.append(t is not None)
            elif arm in STRONG:
                hits.append(t is not None and EVENT < t <= EVENT + 5000)
        row = {"scenario": sc, "observer": o,
               "crossings/window (base, before)": g[(g.arm == "base") & (g.end <= EVENT)].crossings.median(),
               "detector hits strong": f"{sum(hits)}/{len(hits)}",
               "false alarms": f"{sum(fas)}/{len(fas)}"}
        for a in NULL + STRONG:
            row[a] = chg[chg.arm == a].chg.median()
        out.append(row)
    o = pd.DataFrame(out)
    o.to_csv(ROOT / "e7b_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(o.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

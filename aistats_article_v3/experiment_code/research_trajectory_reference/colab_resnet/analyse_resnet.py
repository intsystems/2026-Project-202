"""Score the E7 logs: before/after table, update-PR reference, frozen online detector.

Nothing here is tuned on E7. The MG configuration and the detector rule
(M, B, delta) are read from the CPU experiments (detector_rule.json, copied from
results_detector/chosen_rules.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE / "code", HERE.parents[1] / "code"):
    if p.exists():
        sys.path.insert(0, str(p))
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402
from actdim.estimator.diagnostics import trend_crossings  # noqa: E402

CFG = EstimatorConfig(max_E=20, tau=1, k_neighbors=20, theiler="embedding")
W, S, WARM, HORIZON = 1000, 500, 1500, 5000
NULL = ("base", "batch_up", "scale", "smooth")


def pr(X):
    X = X - X.mean(0)
    s = np.clip(np.linalg.eigvalsh(X @ X.T), 0, None)
    return float(s.sum() ** 2 / (s ** 2).sum())


def simple(seg):
    t = np.arange(len(seg)); r = seg - np.polyval(np.polyfit(t, seg, 1), t)
    return {"crossings": trend_crossings(seg), "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            "det_std": float(r.std() / (abs(seg.mean()) + 1e-12))}


def windows(root: Path, event: int):
    rows = []
    for f in sorted(root.glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        z = np.load(f)
        x, U = z["param_norm"], z["update_sketch"].astype(np.float64)
        variants = {arm: x}
        if arm == "base":
            sc = x.copy(); sc[event:] *= 10
            sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[event:] = c[event:]
            variants.update({"scale": sc, "smooth": sm})
        for name, v in variants.items():
            for a in range(0, len(v) - W + 1, S):
                seg = v[a:a + W]
                rows.append({"arm": name, "seed": int(seed), "start": a, "end": a + W,
                             "MG": estimate(seg, CFG).MG, "update_PR": pr(U[a + 1:a + W]),
                             "update_size": float(np.abs(U[a + 1:a + W]).mean()), **simple(seg)})
    return pd.DataFrame(rows)


def drop_series(g, stat, M, B, sign):
    g = g.sort_values("start")
    v, ends = g[stat].to_numpy(), g.end.to_numpy()
    ok = ends > WARM
    v, ends = v[ok], ends[ok]
    return [(ends[k], sign * (np.median(v[k - M + 1:k + 1]) / np.median(v[k - M - B + 1:k - M + 1]) - 1))
            for k in range(M + B - 1, len(v))]


def detect(w, rules, event):
    rows = []
    for stat, r in rules.items():
        for (arm, seed), g in w.groupby(["arm", "seed"]):
            t = next((e for e, dd in drop_series(g, stat, r["M"], r["B"], r["sign"])
                      if np.isfinite(dd) and dd < -r["delta"]), None)
            null = arm in NULL
            rows.append({"stat": stat, "arm": arm, "seed": seed,
                         "false_alarm": (t is not None) if null else (t is not None and t <= event),
                         "hit": (not null) and t is not None and event < t <= event + HORIZON,
                         "delay": (t - event) if (not null and t is not None and t > event) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_resnet/scratch")
    ap.add_argument("--event", type=int, default=4000)
    args = ap.parse_args()
    root = Path(args.root)
    logs = sorted(root.glob("logs_*_s*.npz"))
    print(f"{root}: {len(logs)} finished runs", flush=True)
    if not logs:
        raise SystemExit(f"no logs_*.npz in {root} -- nothing to analyse yet "
                         "(is $OUT set? did the training cell finish any run?)")
    # The cache is used only if it covers exactly the runs on disk. A cache written
    # before any run had finished was empty and crashed the next analysis; one
    # written halfway through would silently leave the later runs out.
    cache = root / "windows.csv"
    w = None
    if cache.exists() and cache.stat().st_size > 0:
        c = pd.read_csv(cache)
        have = {f"{a}_s{s}" for a, s in zip(c.arm, c.seed)}
        want = {f.stem[5:] for f in logs}
        if want <= have:
            w = c
    if w is None:
        w = windows(root, args.event)
        w.to_csv(cache, index=False)
    ev = args.event
    pre = w[(w.start >= ev - 2000) & (w.end <= ev)].groupby(["arm", "seed"])
    post = w[(w.start >= ev + 1000) & (w.start <= ev + 5000)].groupby(["arm", "seed"])
    cols = ["MG", "update_PR", "update_size", "crossings", "lag1", "det_std"]
    r = (post[cols].median() / pre[cols].median()).reset_index()
    r.to_csv(root / "per_run.csv", index=False)
    g = r.groupby("arm")
    tab = pd.DataFrame({"MG change %": 100 * (g.MG.median() - 1),
                        "min %": 100 * (g.MG.min() - 1), "max %": 100 * (g.MG.max() - 1),
                        "update PR change %": 100 * (g.update_PR.median() - 1),
                        "update size change %": 100 * (g.update_size.median() - 1), "n": g.MG.count()})
    rules = json.load(open(HERE / "detector_rule.json"))
    det = detect(w, rules, ev)
    det.to_csv(root / "detector_per_run.csv", index=False)
    summ = det.groupby(["stat", "arm"]).agg(hit=("hit", "mean"), false_alarm=("false_alarm", "mean"),
                                            delay=("delay", "median")).reset_index()
    summ.to_csv(root / "detector_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string())
    print(summ.pivot(index="arm", columns="stat", values=["hit", "false_alarm"]).round(2).to_string())
    json.dump({"table": tab.round(3).reset_index().to_dict("records")}, open(root / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

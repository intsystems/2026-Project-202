"""E15: configuration of each monitor chosen on E4 calibration logs only, then tested everywhere.

PROTOCOL (fixed before any test result of a variant was computed).
Families and variants (window 1 000, stride 500, E6 change detector):
  MG           (E, tau, k) in (10,1,20) (20,1,20) (20,1,50) (20,2,20) (20,4,20) (10,2,50) (20,4,50) (40,1,20)
  self_repeat  lag range in (20,250) (5,100) (10,500) (50,500) (1,50)
  recurrence   radius fraction in 0.05, 0.1, 0.2
  roughness    one variant
For each variant the E6 calibration (M, B, sign, delta) is run on E4 seeds 0-3. The variant
of a family is the one with the highest calibration hit rate, ties broken by the larger
relative margin: mean over calibration event runs of (largest post-event drop - delta)/delta.
Chosen variants are then scored, unchanged, on:
  E5      seeds 10-13: 28 strong events, 8 weak, 8 training controls, 8 observer controls
  E11     six logging changes applied to E5 logs: hits /28, alarms on base+batch_up /8
  E14     20 operational-event runs (no event): alarms /20
  ResNet  E7 logs: freezing /6, LR+prune /24, training controls /12, observer controls /12
Prediction: the chosen MG variant has the fewest total false alarms on real training runs
(E5 controls, E11 controls, E14, ResNet training controls) among the chosen variants while
keeping >= 24/28 E5 strong hits.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "research_trajectory_reference"
sys.path.insert(0, str(REF)); sys.path.insert(0, str(HERE))
import detector as D  # noqa: E402
import baselines as BL  # noqa: E402
from observer_shift import transform, CHANGES  # noqa: E402
from actdim.estimator.config import EstimatorConfig  # noqa: E402
from actdim.estimator.mle import estimate  # noqa: E402

OUT = HERE / "results_variants"
STRONG = ["lr10", "lr100", "freeze_head", "freeze_bias", "prune50", "prune80", "prune95"]
WEAK = ["lr3", "freeze12"]


def mg_fn(E, tau, k):
    cfg = EstimatorConfig(max_E=E, tau=tau, k_neighbors=k, theiler="embedding")
    return lambda x: estimate(x, cfg).MG


def sr_fn(lo, hi):
    return lambda x: BL.self_repeat(x, lo, hi)


VARIANTS = {**{f"MG_E{E}_t{t}_k{k}": ("MG", mg_fn(E, t, k)) for E, t, k in
               [(10, 1, 20), (20, 1, 20), (20, 1, 50), (20, 2, 20), (20, 4, 20), (10, 2, 50), (20, 4, 50), (40, 1, 20)]},
            **{f"SR_{lo}_{hi}": ("self_repeat", sr_fn(lo, hi)) for lo, hi in
               [(20, 250), (5, 100), (10, 500), (50, 500), (1, 50)]},
            **{f"RR_{f}": ("recurrence", partial(BL.recurrence_rate, tau=1, frac=f)) for f in (0.05, 0.1, 0.2)},
            "rough": ("roughness", BL.roughness)}


def logs():
    L = []
    for f in sorted((REF / "results_cifar").glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        if arm in D.CAL_EVENTS + D.CAL_NULL:
            L.append(("cal", arm, int(seed), np.load(f)["param_norm"].astype(float)))
    for f in sorted((REF / "results_graded").glob("logs_*_s*.npz")):
        arm, seed = f.stem[5:].rsplit("_s", 1)
        x = np.load(f)["param_norm"].astype(float)
        L.append(("e5", arm, int(seed), x))
        if arm == "base":
            sc = x.copy(); sc[D.EVENT:] *= 10
            sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[D.EVENT:] = c[D.EVENT:]
            L += [("e5", "scale", int(seed), sc), ("e5", "smooth", int(seed), sm)]
        for ch in CHANGES:
            rng = np.random.default_rng(zlib.crc32(f"{seed}|{arm}|{ch}".encode()))
            L.append((f"e11_{ch}", arm, int(seed), transform(x, ch, rng)))
    for f in sorted((HERE / "results_ops").glob("pn_*_s*.npy")):
        arm, seed = f.stem[3:].rsplit("_s", 1)
        L.append(("e14", arm, int(seed), np.load(f).astype(float)))
    for scen in ("scratch", "finetune"):
        z = np.load(REF / "results_resnet" / f"observers_{scen}.npz")
        for key in sorted({k.split("|")[0] for k in z.keys()}):
            arm, seed = key.rsplit("_s", 1)
            x = z[f"{key}|param_norm"].astype(float)
            sd = int(seed) + (100 if scen == "finetune" else 0)
            L.append(("resnet", arm, sd, x))
            if arm == "base":
                sc = x.copy(); sc[4000:] *= 10
                sm = x.copy(); c = np.convolve(x, np.ones(16) / 16, mode="full")[:len(x)]; sm[4000:] = c[4000:]
                L += [("resnet", "scale", sd, sc), ("resnet", "smooth", sd, sm)]
    return L


def windows(item):
    suite, arm, seed, x = item
    rows = []
    for a in range(0, len(x) - D.W + 1, D.S):
        seg = x[a:a + D.W]
        r = {"suite": suite, "arm": arm, "seed": seed, "start": a, "end": a + D.W}
        for name, (_, fn) in VARIANTS.items():
            try:
                r[name] = float(fn(seg))
            except Exception:
                r[name] = np.nan
        rows.append(r)
    return rows


def margin(cal, stat, rule):
    out = []
    for (arm, seed), g in cal[cal.arm.isin(D.CAL_EVENTS)].groupby(["arm", "seed"]):
        ds = [d for end, d in D.drop_series(g, stat, rule["M"], rule["B"], rule["sign"])
              if end > D.EVENT and np.isfinite(d)]
        if ds:
            out.append((-min(ds) - rule["delta"]) / rule["delta"])
    return float(np.mean(out)) if out else -np.inf


def score(Wd, stat, r):
    res = {}
    ev = lambda w, nulls, event: D.evaluate(w, stat, r["M"], r["B"], r["delta"], r["sign"], nulls, event)  # noqa: E731
    e5 = ev(Wd[Wd.suite == "e5"], ("base", "batch_up", "scale", "smooth"), D.EVENT)
    res["E5 strong"] = int(e5[e5.arm.isin(STRONG)].hit.sum())
    res["E5 weak"] = int(e5[e5.arm.isin(WEAK)].hit.sum())
    res["E5 FA train"] = int(e5[e5.arm.isin(["base", "batch_up"])].false_alarm.sum())
    res["E5 FA observer"] = int(e5[e5.arm.isin(["scale", "smooth"])].false_alarm.sum())
    e11h = e11f = 0
    for ch in CHANGES:
        p = ev(Wd[Wd.suite == f"e11_{ch}"], ("base", "batch_up"), D.EVENT)
        e11h += int(p[p.arm.isin(STRONG)].hit.sum()); e11f += int(p[p.arm.isin(["base", "batch_up"])].false_alarm.sum())
    res["E11 hits /168"], res["E11 FA /48"] = e11h, e11f
    e14 = Wd[Wd.suite == "e14"]
    p = ev(e14, tuple(e14.arm.unique()), 4000)
    res["E14 FA /20"] = int(p.false_alarm.sum())
    p = ev(Wd[Wd.suite == "resnet"], ("base", "batch_up", "scale", "smooth"), 4000)
    res["ResNet freeze /6"] = int(p[p.arm == "freeze_head"].hit.sum())
    res["ResNet other /24"] = int(p[p.arm.isin(["lr10", "lr100", "prune80", "prune95"])].hit.sum())
    res["ResNet FA train /12"] = int(p[p.arm.isin(["base", "batch_up"])].false_alarm.sum())
    res["ResNet FA observer /12"] = int(p[p.arm.isin(["scale", "smooth"])].false_alarm.sum())
    res["FA real training"] = res["E5 FA train"] + res["E11 FA /48"] + res["E14 FA /20"] + res["ResNet FA train /12"]
    return res


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / "windows.csv"
    if path.exists():
        Wd = pd.read_csv(path)
    else:
        items = logs()
        print(len(items), "logs", flush=True)
        with Pool(14) as p:
            Wd = pd.DataFrame([r for rr in p.imap(windows, items, chunksize=4) for r in rr])
        Wd.to_csv(path, index=False)
    cal = Wd[Wd.suite == "cal"]
    rows, rules = [], {}
    for name, (fam, _) in VARIANTS.items():
        r = D.calibrate(cal, name) if fam != "MG" else D.calibrate(cal, name)
        rules[name] = r
        rows.append({"variant": name, "family": fam, "cal_hit": r["cal_hit"], "margin": margin(cal, name, r),
                     **score(Wd, name, r)})
    R = pd.DataFrame(rows)
    R["chosen"] = False
    for fam, g in R.groupby("family"):
        best = g.sort_values(["cal_hit", "margin"], ascending=False).index[0]
        R.loc[best, "chosen"] = True
    R.to_csv(OUT / "variants.csv", index=False)
    json.dump(rules, open(OUT / "rules.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 300)
    print(R.round(2).to_string(index=False))
    print("\nCHOSEN ON CALIBRATION:")
    print(R[R.chosen].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

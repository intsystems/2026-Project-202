"""Calibration/test evaluation of label-free selectors (protocol: see ssl_main.py docstring).

usage: python analyze.py <rundir> <featdir> <outdir>   (seeds split by configs.CAL_SEEDS/TEST_SEEDS)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import LOGS  # noqa: E402

SCALAR = ["MG", "MG_t4k50", "spectral_entropy", "self_repeat", "roughness", "perm_entropy",
          "recurrence_rate", "corr_dim", "twonn", "linear_pr", "crossings", "lag1", "det_std",
          "mean", "rel_std", "rel_change"]
INTERNAL = ["rankme_h", "rankme_z", "alpha_h", "abs_alpha1", "lidar_z", "out_std"]
FINAL, EARLY = 3000, 1500
BAD_GAP = 0.02          # set from the pilot: "bad" = final lin_acc < seed's random-init lin_acc + BAD_GAP


def family(name):
    return name.split("_")[0]


def load(rundir, featdir):
    meta, ev = [], []
    for jf in sorted(Path(rundir).glob("*.json")):
        m = json.load(open(jf))
        name, seed = jf.stem.rsplit("_s", 1)
        for r in m["evals"]:
            ev.append({"run": jf.stem, "name": name, "seed": int(seed), **r})
        meta.append({"run": jf.stem, "name": name, "seed": int(seed), "family": family(name),
                     "t_train": m["t_train"], **{"c_" + k: v for k, v in m["cost"].items()}})
    ev = pd.DataFrame(ev)
    ev["abs_alpha1"] = -(ev["alpha_h"] - 1).abs()
    feats = pd.concat([pd.read_csv(f) for f in sorted(Path(featdir).glob("*.csv"))], ignore_index=True)
    return pd.DataFrame(meta), ev, feats


def table(meta, ev, feats, end):
    """One row per run: candidate scores computed from information available at step `end`,
    plus the final ground truth."""
    truth = ev[ev.step == FINAL][["run", "lin_acc", "knn_acc", "rankme_h"]].rename(
        columns={"lin_acc": "acc", "knn_acc": "knn", "rankme_h": "rankme_final"})
    init = ev[ev.step == 0][["run", "lin_acc"]].rename(columns={"lin_acc": "acc0"})
    at = ev[ev.step == end][["run"] + INTERNAL].copy()
    f = feats[feats.end == end]
    wide = f.pivot(index="run", columns="log", values=[s for s in SCALAR if s in f.columns])
    wide.columns = [f"{s}|{lg}" for s, lg in wide.columns]
    df = meta.merge(truth, on="run").merge(init, on="run").merge(at, on="run").merge(
        wide.reset_index(), on="run", how="left")
    seed_init = df.groupby("seed").acc0.transform("mean")
    df["bad"] = (df.acc < seed_init + BAD_GAP).astype(int)
    return df


def candidates(df, stat):
    if stat in INTERNAL:
        return [stat]
    return [f"{stat}|{lg}" for lg in LOGS if f"{stat}|{lg}" in df.columns]


def signed(g, col, sign):
    """sign * score; NaN/inf (undefined statistic, e.g. constant log) ranked as the worst."""
    v = sign * np.asarray(g[col].values, float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.zeros(len(v))
    v[~ok] = v[ok].min() - 1.0
    return v


def sel_metrics(df, col, sign, minn=5):
    out = []
    for s, g in df.groupby("seed"):
        if g[col].notna().sum() < minn:
            continue
        sc = signed(g, col, sign)
        acc = g.acc.values
        o = np.argsort(-sc)
        best = df[df.seed == s].acc.max()
        out.append({"seed": s, "rho": spearmanr(sc, acc)[0], "regret1": best - acc[o[0]],
                    "regret3": best - acc[o[:3]].max(), "n": len(g)})
    return pd.DataFrame(out)


def auc_metric(df, col, sign, label="bad"):
    g = df
    if g[label].nunique() < 2 or g[col].notna().sum() < 5:
        return float("nan")
    return roc_auc_score(g[label], -signed(g, col, sign))   # high score = good run -> low flags bad


def calibrate(cal, stat, objective):
    best = None
    for col in candidates(cal, stat):
        for sign in (1, -1):
            if objective == "rho":
                m = sel_metrics(cal, col, sign)
                v = m.rho.mean() if len(m) else -np.inf
            else:
                v = auc_metric(cal, col, sign)
            if np.isfinite(v) and (best is None or v > best[0]):
                best = (v, col, sign)
    return best


def boot_diff(test, colA, sA, colB, sB, n=2000, seed=0):
    """Paired bootstrap over configurations (same resample in every test seed) of mean rho A - B."""
    rng = np.random.default_rng(seed)
    names = sorted(test["name"].unique())
    piv = {s: g.set_index("name") for s, g in test.groupby("seed")}
    d = []
    for _ in range(n):
        pick = rng.choice(names, len(names), replace=True)
        r = []
        for s, g in piv.items():
            gg = g.loc[[p for p in pick if p in g.index]]
            a = spearmanr(signed(gg, colA, sA), gg.acc)[0]
            b = spearmanr(signed(gg, colB, sB), gg.acc)[0]
            r.append(a - b)
        d.append(np.nanmean(r))
    d = np.array(d)
    return float(np.mean(d > 0)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main():
    from configs import CAL_SEEDS, TEST_SEEDS
    meta, ev, feats = load(sys.argv[1], sys.argv[2])
    sd = lambda d, S: d[d.run.str.rsplit("_s", n=1).str[1].astype(int).isin(S)].copy()  # noqa: E731
    cal_meta, cal_ev, cal_f = sd(meta, CAL_SEEDS), sd(ev, CAL_SEEDS), sd(feats, CAL_SEEDS)
    te_meta, te_ev, te_f = sd(meta, TEST_SEEDS), sd(ev, TEST_SEEDS), sd(feats, TEST_SEEDS)
    out = Path(sys.argv[3]); out.mkdir(exist_ok=True)
    stats = SCALAR + INTERNAL
    res = {}
    # ---- T1: configuration ranking at the final checkpoint
    cal, test = table(cal_meta, cal_ev, cal_f, FINAL), table(te_meta, te_ev, te_f, FINAL)
    cal.to_csv(out / "cal_final.csv", index=False); test.to_csv(out / "test_final.csv", index=False)
    rows = []
    chosen = {}
    for st in stats:
        b = calibrate(cal, st, "rho")
        if b is None:
            continue
        v, col, sign = b
        chosen[st] = (col, sign)
        m = sel_metrics(test, col, sign)
        fam = []
        for fm, g in test.groupby("family"):
            if g["name"].nunique() >= 4:
                mm = sel_metrics(g, col, sign, minn=4)
                if len(mm):
                    fam.append(mm.rho.mean())
        rows.append({"stat": st, "col": col, "sign": sign, "cal_rho": v, "test_rho": m.rho.mean(),
                     "test_rho_sd": m.rho.std(), "regret1": m.regret1.mean(), "regret3": m.regret3.mean(),
                     "within_family_rho": np.nanmean(fam) if fam else np.nan,
                     "auc_bad": auc_metric(test, col, sign)})
    # trivial rules
    rng = np.random.default_rng(0)
    rr = []
    for s, g in test.groupby("seed"):
        best = g.acc.max()
        rr.append({"r1": best - g.acc.mean(),
                   "r3": np.mean([best - g.acc.values[rng.choice(len(g), 3, replace=False)].max() for _ in range(2000)])})
    rr = pd.DataFrame(rr)
    rows.append({"stat": "random_choice", "test_rho": 0.0, "regret1": rr.r1.mean(), "regret3": rr.r3.mean()})
    # default config of the most common method family
    d = test[test["name"] == "ss_base"]
    if len(d):
        rows.append({"stat": "default_ss_base", "regret1": float((test.groupby("seed").acc.max() - d.set_index("seed").acc).mean())})
    t1 = pd.DataFrame(rows).sort_values("test_rho", ascending=False)
    t1.to_csv(out / "T1_config_ranking.csv", index=False)
    res["T1"] = t1
    print("=== T1 configuration ranking (final checkpoint), test seeds ===")
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # bootstrap MG vs every other stat (test)
    bs = []
    if "MG" in chosen:
        for st, (col, sign) in chosen.items():
            if st == "MG":
                continue
            p, lo, hi = boot_diff(test, *chosen["MG"], col, sign, n=500)
            bs.append({"vs": st, "P(MG better)": p, "diff_lo": lo, "diff_hi": hi})
        bs = pd.DataFrame(bs)
        bs.to_csv(out / "T1_bootstrap_MG_vs.csv", index=False)
        print(bs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- T2: collapse/failed-run detection (AUC), own calibration
    rows = []
    for st in stats:
        b = calibrate(cal, st, "auc")
        if b is None:
            continue
        v, col, sign = b
        rows.append({"stat": st, "col": col, "sign": sign, "cal_auc": v, "test_auc": auc_metric(test, col, sign)})
    t2 = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    t2.to_csv(out / "T2_bad_run_auc.csv", index=False)
    print(f"=== T2 failed-run AUC at final checkpoint; test bad = {int(test.bad.sum())}/{len(test)} ===")
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- T3: early warning: information up to step EARLY predicts final failure
    cal_e, test_e = table(cal_meta, cal_ev, cal_f, EARLY), table(te_meta, te_ev, te_f, EARLY)
    rows = []
    for st in stats:
        b = calibrate(cal_e, st, "auc")
        if b is None:
            continue
        v, col, sign = b
        rows.append({"stat": st, "col": col, "sign": sign, "cal_auc": v, "test_auc": auc_metric(test_e, col, sign)})
    t3 = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    t3.to_csv(out / "T3_early_warning_auc.csv", index=False)
    print(f"=== T3 early warning (window ending at step {EARLY}) AUC for final failure ===")
    print(t3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- T4: checkpoint choice within run (checkpoints 1000..4000 step 500 with probes)
    def ck_table(meta, ev, feats):
        parts = []
        for end in range(1000, FINAL + 1, 500):
            t = table(meta, ev, feats, end)
            acc_end = ev[ev.step == end][["run", "lin_acc"]].rename(columns={"lin_acc": "acc_end"})
            parts.append(t.merge(acc_end, on="run").assign(end=end))
        return pd.concat(parts, ignore_index=True)
    cal_c, test_c = ck_table(cal_meta, cal_ev, cal_f), ck_table(te_meta, te_ev, te_f)

    def ck_regret(df, col, sign):
        r = []
        for run, g in df.groupby("run"):
            if g[col].notna().sum() < 1:
                continue
            r.append(g.acc_end.max() - g.acc_end.values[np.argmax(signed(g, col, sign))])
        return float(np.mean(r)) if r else np.inf
    rows = []
    for st in stats:
        best = None
        for col in candidates(cal_c, st):
            for sign in (1, -1):
                v = ck_regret(cal_c, col, sign)
                if best is None or v < best[0]:
                    best = (v, col, sign)
        rows.append({"stat": st, "col": best[1], "sign": best[2], "cal_regret": best[0],
                     "test_regret": ck_regret(test_c, best[1], best[2])})
    last = float(np.mean([g.acc_end.max() - g[g.end == FINAL].acc_end.iloc[0] for _, g in test_c.groupby("run")]))
    rows.append({"stat": "last_checkpoint", "test_regret": last})
    t4 = pd.DataFrame(rows).sort_values("test_regret")
    t4.to_csv(out / "T4_checkpoint_regret.csv", index=False)
    print("=== T4 within-run checkpoint selection, mean regret (lin acc) ===")
    print(t4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- costs
    tf = pd.concat([cal_f, te_f])
    costs = {c[2:]: float(tf[c].mean() * 1000) for c in tf.columns if c.startswith("t_")}
    mm = pd.concat([cal_meta, te_meta])
    n_ev = 1 + FINAL // 250
    costs.update({"rankme_with_embed_per_eval": float((mm.c_rankme + mm.c_embed).mean() / n_ev * 1000),
                  "lidar_per_eval": float(mm.c_lidar.mean() / n_ev * 1000),
                  "alpha_per_eval": float((mm.c_alpha + mm.c_embed).mean() / n_ev * 1000),
                  "train_ms_per_step": float(mm.t_train.mean() / FINAL * 1000)})
    json.dump(costs, open(out / "costs_ms.json", "w"), indent=1)
    print("costs (ms):", json.dumps({k: round(v, 2) for k, v in costs.items()}))


if __name__ == "__main__":
    main()

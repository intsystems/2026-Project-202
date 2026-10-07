"""Calibrate every decay rule on the calibration units, score on the test units (see main.py)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules as R  # noqa: E402
from stats import STATS, SIMPLE, LOGS  # noqa: E402

RES = HERE / "results"
ALL_STATS = list(STATS) + list(SIMPLE)
CALONLY = "calonly" in sys.argv


def load_runs():
    runs = []
    for f in sorted(RES.glob("res_*.json")):
        res = json.load(open(f))
        if not res.get("done"):
            continue
        tag = f.stem[4:]
        wf = RES / f"win_{tag}.csv"
        if not wf.exists():
            continue
        c = res["cond"]
        lg = dict(np.load(RES / f"logs_{tag}.npz"))
        win = pd.read_csv(wf)
        run = {"tag": tag, "name": c["name"], "split": c["split"], "seed": res["seed"], "T": c["T"],
               "lr": c["lr"], "res": res, "logs": lg, "ev": json.load(open(RES / f"ev_{tag}.json"))["ev"],
               "win": {L: {k: g[k].to_numpy() for k in ["end"] + ALL_STATS}
                       for L, g in win.groupby("log")}}
        run["sasa"] = R.sasa_terms(run)
        runs.append(run)
    return runs


def families(quant):
    fam = {"fixed_step": (R.fixed_grid(), R.fixed),
           "pflug_chee_toulis": (R.pflug_grid(), R.pflug),
           "sasa": (R.sasa_grid(), R.sasa),
           "pesme_distance": (R.pesme_grid(), R.pesme),
           "plateau_train_loss": (R.plateau_grid(), R.plateau_train),
           "plateau_val_loss": (R.plateau_grid(), R.plateau_val)}
    for s in ALL_STATS:
        grid = [dict(p, log=L) for L in LOGS for p in R.stat_grid(quant)]
        fam[s] = (grid, (lambda s: lambda run, p: R.stat_rule(run, p["log"], s, p, quant))(s))
    return fam


def score(runs, fn, p, key="test_acc"):
    vals = [R.outcome(r["res"], fn(r, p))[key] for r in runs]
    return float(np.mean(vals))


def calibrate(cal, grid, fn, restrict=None):
    best = None
    for p in grid:
        if restrict and not restrict(p):
            continue
        s = score(cal, fn, p)
        if best is None or s > best[0] + 1e-12:
            best = (s, p)
    return best


def per_run(runs, fn, p):
    rows = []
    for r in runs:
        a = fn(r, p) if fn is not None else None
        o = R.outcome(r["res"], a)
        rows.append({"tag": r["tag"], "cond": r["name"], "seed": r["seed"], "alarm": a,
                     "decay": R.decay_step(r["res"], a), "T": r["T"],
                     "test_acc": o["test_acc"], "test_loss": o["test_loss"]})
    return pd.DataFrame(rows)


def oracle_rows(runs):
    rows = []
    for r in runs:
        opts = [(r["res"]["none"], None)] + [(r["res"]["branches"][str(t)], t) for t in r["res"]["grid"]]
        o, t = max(opts, key=lambda z: z[0]["test_acc"])
        rows.append({"tag": r["tag"], "cond": r["name"], "seed": r["seed"], "decay": t, "T": r["T"],
                     "test_acc": o["test_acc"], "test_loss": o["test_loss"]})
    return pd.DataFrame(rows)


def main(restrict_tmin=None, suffix=""):
    runs = load_runs()
    cal = [r for r in runs if r["split"] == "cal"]
    test = [] if CALONLY else [r for r in runs if r["split"] == "test"]
    print(f"calibration units {len(cal)}, test units {len(test)}")
    quant = {}
    for L in LOGS:
        for s in ALL_STATS:
            v = np.concatenate([r["win"][L][s] for r in cal])
            v = v[np.isfinite(v)]
            quant[(L, s)] = {q: float(np.quantile(v, q)) for q in R.QS}
    fam = families(quant)
    restrict = (lambda p: p.get("tmin", None) in (None, restrict_tmin)) if restrict_tmin else None
    chosen, rows, per = {}, [], []
    orc_t = oracle_rows(test)
    orc_c = oracle_rows(cal)
    for name, (grid, fn) in fam.items():
        s, p = calibrate(cal, grid, fn, restrict if name != "fixed_step" else None)
        chosen[name] = p
        if CALONLY:
            pc = per_run(cal, fn, p)
            print(f"{name:20s} cal_acc {s:.4f} regret {(orc_c.test_acc.values - pc.test_acc.values).mean():.4f} {p}", flush=True)
            continue
        pr = per_run(test, fn, p)
        pr["rule"] = name
        per.append(pr)
        pc = per_run(cal, fn, p)
        rows.append({"rule": name, "params": json.dumps(p), "cal_acc": s,
                     "cal_regret": float((orc_c.test_acc.values - pc.test_acc.values).mean()),
                     "test_acc": pr.test_acc.mean(), "test_loss": pr.test_loss.mean(),
                     "test_regret": float((orc_t.test_acc.values - pr.test_acc.values).mean()),
                     "never_decays": float(pr.decay.isna().mean()),
                     "median_decay_frac": float(np.nanmedian(pr.decay / pr["T"]))})
    if CALONLY:
        return None
    for name, key in (("constant_lr", "none"), ("cosine", "cosine")):
        pr = pd.DataFrame([{"tag": r["tag"], "cond": r["name"], "seed": r["seed"], "T": r["T"],
                            "test_acc": r["res"][key]["test_acc"], "test_loss": r["res"][key]["test_loss"]}
                           for r in test])
        pr["rule"] = name
        per.append(pr)
        rows.append({"rule": name, "test_acc": pr.test_acc.mean(), "test_loss": pr.test_loss.mean(),
                     "test_regret": float((orc_t.test_acc.values - pr.test_acc.values).mean())})
    orc_t["rule"] = "oracle_best_decay_time"
    per.append(orc_t)
    rows.append({"rule": "oracle_best_decay_time", "test_acc": orc_t.test_acc.mean(),
                 "test_loss": orc_t.test_loss.mean(), "test_regret": 0.0})
    tab = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    per = pd.concat(per)
    # paired bootstrap of MG minus each rule (test units)
    rng = np.random.default_rng(0)
    piv = per.pivot_table(index="tag", columns="rule", values="test_acc")
    B = rng.integers(0, len(piv), (5000, len(piv)))
    for rname in piv.columns:
        d = (piv["MG"] - piv[rname]).to_numpy()
        bs = d[B].mean(1)
        tab.loc[tab.rule == rname, "MG_minus_rule"] = d.mean()
        tab.loc[tab.rule == rname, "ci_lo"] = np.quantile(bs, 0.025)
        tab.loc[tab.rule == rname, "ci_hi"] = np.quantile(bs, 0.975)
        tab.loc[tab.rule == rname, "MG_wins/ties/losses"] = f"{(d > 0.0005).sum()}/{(abs(d) <= 0.0005).sum()}/{(d < -0.0005).sum()}"
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    out = HERE / "results"
    tab.to_csv(out / f"test_table{suffix}.csv", index=False)
    per.to_csv(out / f"test_per_run{suffix}.csv", index=False)
    json.dump(chosen, open(out / f"chosen_rules{suffix}.json", "w"), indent=1, default=str)
    print(tab.drop(columns=["params"]).round(4).to_string(index=False))
    print(tab[["rule", "params"]].to_string(index=False))
    # per-condition means for the key rules
    keyr = ["oracle_best_decay_time", "cosine", "constant_lr", "fixed_step", "MG", "self_repeat",
            "plateau_val_loss", "plateau_train_loss", "pflug_chee_toulis", "sasa", "pesme_distance"]
    pc = per[per.rule.isin(keyr)].pivot_table(index="cond", columns="rule", values="test_acc")
    print(pc[[k for k in keyr if k in pc.columns]].round(3).to_string())
    pc.to_csv(out / f"test_by_condition{suffix}.csv")
    return tab


if __name__ == "__main__":
    main()
    if len(sys.argv) > 1 and sys.argv[1] == "pure":
        print("\n==== pure diagnostics: warm-up fixed at 2/16 T (no time prior) ====")
        main(restrict_tmin=2 / 16, suffix="_pure")

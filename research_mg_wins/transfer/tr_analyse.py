"""Analysis of setting T (protocol in tr_main.py docstring). Works on whatever runs exist."""
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
import tr_monitors as Mo  # noqa: E402
from tr_main import TARGETS, EVENTS, RUNS  # noqa: E402

OUT = HERE / "results"


def load_runs():
    runs = []
    for js in sorted(RUNS.glob("*.json")):
        csv = js.with_name(js.stem + "_win.csv")
        if not csv.exists():
            continue
        m = json.load(open(js))
        z = np.load(js.with_suffix(".npz"))
        runs.append({"setup": m["setup"], "seed": m["seed"], "event": m["event"], "t_e": m["t_e"],
                     "t_r": m["t_r"], "wall": m["wall_s"], "acc": m["test_acc"], "P": m["P"],
                     "win": _win(csv), "gn": z["log_gn"], "loss": z["batch_loss"],
                     "truth": {k: z[f"truth_{k}"] for k in ("t", "frac_moving", "upr", "erank",
                                                            "srank", "dormant", "probe_acc")}})
    return runs


def _win(csv):
    w = pd.read_csv(csv)
    dt = csv.with_name(csv.name.replace("_win.csv", "_wind.csv"))
    if dt.exists():
        w = w.merge(pd.read_csv(dt), on="start", how="left")
    for k in Mo.SCALAR:
        if f"{k}_dt" not in w.columns:
            w[f"{k}_dt"] = np.nan
    return w


SECONDARY = {f"{k}_dt": ("ratio", f"{k}_dt") for k in Mo.SCALAR}


def confirm(r):
    if r["t_e"] is None:
        return None
    T, te = r["truth"], r["t_e"]
    t = T["t"]
    med = lambda k, a, b: float(np.median(T[k][(t > a) & (t <= b)]))  # noqa: E731
    if r["event"] in ("freeze", "prune"):
        return med("frac_moving", te, te + 1000) / med("frac_moving", te - 1000, te) < 0.5
    return med("erank", te + 500, te + 1500) / med("erank", te - 1000, te) < 0.8


def rescore(r, alarm):
    te = r["t_e"]
    if te is None:
        return {"fa": alarm is not None, "hit": False, "early": False, "delay": np.nan}
    end = te + Mo.HORIZON
    if r["setup"] == "T8_restart" and r["t_r"] is not None and r["t_r"] > te:
        end = min(end, r["t_r"])
    early = alarm is not None and alarm <= te
    hit = alarm is not None and te < alarm <= end
    return {"fa": False, "hit": hit, "early": early, "delay": alarm - te if hit else np.nan}


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    src = [r for r in runs if r["setup"] == "source"]
    print(len(runs), "runs loaded;", len(src), "source")
    if all(f"{k}" in src[0]["win"].columns for k in SECONDARY):
        Mo.MONITORS.update(SECONDARY)
    rules = {}
    for name, mon in Mo.MONITORS.items():
        rules[name] = Mo.calibrate(src, mon)
    json.dump(rules, open(OUT / "rules.json", "w"), indent=1, default=float)

    rows = []
    for r in runs:
        conf = confirm(r)
        for name, mon in Mo.MONITORS.items():
            p = rules[name]
            a = Mo.evaluate([r], mon, p, p.get("thr"))[0]["alarm"]
            rows.append({"setup": r["setup"], "seed": r["seed"], "event": r["event"] or "none",
                         "t_e": r["t_e"], "t_r": r["t_r"], "confirmed": conf, "monitor": name,
                         "alarm": a, **rescore(r, a)})
    PR = pd.DataFrame(rows)
    PR.to_csv(OUT / "per_run.csv", index=False)

    summ = []
    for (setup, mon), g in PR.groupby(["setup", "monitor"]):
        ev, nu = g[g.event != "none"], g[g.event == "none"]
        evc = ev[ev.confirmed == True]  # noqa: E712
        hr = ev.hit.mean() if len(ev) else np.nan
        far = nu.fa.mean() if len(nu) else np.nan
        summ.append({"setup": setup, "monitor": mon, "n_ev": len(ev), "hits": int(ev.hit.sum()),
                     "hits_conf": f"{int(evc.hit.sum())}/{len(evc)}", "early": int(ev.early.sum()),
                     "n_null": len(nu), "fa": int(nu.fa.sum()),
                     **{f"hit_{e}": int(ev[ev.event == e].hit.sum()) for e in EVENTS},
                     "delay": float(np.nanmedian(ev.delay)) if ev.hit.any() else np.nan,
                     "J": hr - far})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)

    mons = list(Mo.MONITORS)
    order = ["source"] + [t for t in TARGETS if t in set(S.setup)]
    cell = S.assign(c=lambda d: d.hits.astype(str) + "/" + d.n_ev.astype(str) + " " + d.fa.astype(str)
                    + "/" + d.n_null.astype(str) + "+" + d.early.astype(str))
    tab = cell.pivot(index="monitor", columns="setup", values="c").reindex(index=mons, columns=order)
    Jt = S.pivot(index="monitor", columns="setup", values="J").reindex(index=mons, columns=order)
    T19 = [t for t in TARGETS[1:] if t in Jt.columns]
    T09 = [t for t in TARGETS if t in Jt.columns]
    Jt["J_T1..T9"] = Jt[T19].mean(1)
    Jt["J_T0..T9"] = Jt[T09].mean(1)
    tot = S[S.setup.isin(T09)].groupby("monitor").agg(hits=("hits", "sum"), n_ev=("n_ev", "sum"),
                                                     fa=("fa", "sum"), n_null=("n_null", "sum"),
                                                     early=("early", "sum"))
    dl = PR[PR.setup.isin(T09) & PR.hit].groupby("monitor").delay.median()
    tot["delay"] = dl
    tot = tot.reindex(mons)
    tot["J_T1..T9"] = Jt["J_T1..T9"]; tot["J_T0..T9"] = Jt["J_T0..T9"]
    tot = tot.sort_values("J_T1..T9", ascending=False)
    tab.to_csv(OUT / "table_cells.csv"); Jt.to_csv(OUT / "table_J.csv"); tot.to_csv(OUT / "table_overall.csv")

    # descriptive: dynamics per setup (null runs, windows ending in (2000, 4000])
    desc = []
    for r in runs:
        if r["event"] is not None:
            continue
        w = r["win"]; w = w[(w.end > 2000) & (w.end <= 4000)]
        t = r["truth"]["t"]; sel = (t > 2000) & (t <= 4000)
        desc.append({"setup": r["setup"], "P": r["P"], "acc": r["acc"], "wall_s": r["wall"],
                     **{k: w[k].median() for k in ("MG", "MG_t1k20", "self_repeat", "recurrence_rate",
                                                   "roughness", "gn_med", "loss_med")},
                     **{f"truth_{k}": float(np.median(r["truth"][k][sel])) for k in ("upr", "erank", "srank", "dormant")}})
    Dd = pd.DataFrame(desc).groupby("setup").median().reindex(order)
    Dd.to_csv(OUT / "dynamics_by_setup.csv")
    conf = PR[(PR.monitor == "MG") & (PR.event != "none")].groupby(["setup", "event"]).confirmed.agg(["sum", "count"])
    conf.to_csv(OUT / "confirmation.csv")

    sec = None
    if "MG_dt" in rules:
        srows = []
        for k in Mo.SCALAR:
            a, b = rules[k], rules[f"{k}_dt"]
            pick = f"{k}_dt" if (b["cal_hits"], -b["cal_delay"]) > (a["cal_hits"], -a["cal_delay"]) else k
            row = tot.loc[pick].to_dict()
            srows.append({"statistic": k, "log_chosen_on_S": "detrended" if pick != k else "raw",
                          "cal_hits": rules[pick]["cal_hits"], **row})
        sec = pd.DataFrame(srows).sort_values("J_T1..T9", ascending=False)
        sec.to_csv(OUT / "table_secondary.csv", index=False)
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 50)
    with open(OUT / "tables.txt", "w", encoding="utf8") as f:
        print("cells: hits/n_event  nullFA/n_null + early", file=f)
        print(tab.to_string(), file=f)
        print("\nYouden J", file=f); print(Jt.round(2).to_string(), file=f)
        print("\noverall T0..T9", file=f); print(tot.round(3).to_string(), file=f)
        print("\ndynamics (null runs, windows ending in (2000,4000])", file=f); print(Dd.round(4).to_string(), file=f)
        print("\nconfirmation", file=f); print(conf.to_string(), file=f)
        print("\nrules", file=f)
        for k, v in rules.items():
            print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, file=f)
    print(open(OUT / "tables.txt", encoding="utf8").read()[:200])


if __name__ == "__main__":
    main()

"""Prepare the closed-loop confirmation (cl_s1.py): frozen calibrated rules for the main
policies, plus a secondary theory-direction MG rule (MG_drop: REL rule, alarm only on a DROP
of MG, calibrated on calibration runs only; restricted grid). Also reports MG_drop on the
test pools by renewal (secondary, not used for the verdict)."""
import json
import numpy as np
import pandas as pd
import eval_s1 as E

for fam in ("F1", "F2"):
    df = E.load(fam)
    cal_pools, mons = E.runs_of(df, "cal", E.CAL_CONDS)
    test_pools, _ = E.runs_of(df, "test", E.CAL_CONDS + E.UNSEEN)
    best = None
    for m in [c for c in mons if c.startswith("MG|")]:
        for B in (2, 4):
            for M in (1, 2, 4):
                for d in E.DELTAS:
                    rule = ("REL", 1, B, M, d)
                    u, n = E.score(E.evaluate(cal_pools, lambda c, k, run: E.alarm_index(run[1][m], rule)))
                    if best is None or (round(u, 9), -n) > best[0]:
                        best = ((round(u, 9), -n), m, rule, u, n)
    _, m, rule, u, n = best
    ut, nt = E.score(E.evaluate(test_pools, lambda c, k, run: E.alarm_index(run[1][m], rule)))
    print(f"{fam} MG_drop: {m} {rule} cal {u:.4f}/{n:.1f} test {ut:.4f}/{nt:.1f}")
    allm = pd.read_csv(E.RES / f"{fam}_all_monitors.csv")
    bys = pd.read_csv(E.RES / f"{fam}_by_stat.csv")
    bys = bys[bys.scope == "any"].set_index("stat")
    chosen = {r["monitor"]: r["rule"] for r in json.load(open(E.RES / f"{fam}_calibration.json"))["monitors"]}
    pol = {"never": "never", "fixed": "fixed"}
    for s in ("MG", "dormant_0.0", "dormant_0.1", "srank", "pnorm_end", "acc_mean", "self_repeat", "spectral_entropy"):
        mon = bys.loc[s, "monitor"]
        pol[s] = [mon, chosen[mon]]
    pol["MG_drop"] = [m, list(rule)]
    json.dump(pol, open(E.HERE / "closed_loop" / f"{fam}_policies.json", "w"), indent=1)
    json.dump({"monitor": m, "rule": rule, "cal_util": u, "cal_resets": n, "test_util": ut, "test_resets": nt},
              open(E.RES / f"{fam}_MG_drop.json", "w"), indent=1)

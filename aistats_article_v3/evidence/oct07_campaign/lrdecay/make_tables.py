"""Markdown tables for REPORT_ru.md from the saved analysis CSVs."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def f(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def short(p):
    if not isinstance(p, str):
        return ""
    p = json.loads(p)
    keys = ("log", "type", "tmin", "sign", "q", "M", "delta", "eps", "f", "patience", "thr", "mode", "z", "start", "th")
    ab = {"param_norm": "pn", "batch_loss": "bl"}
    return ", ".join(f"{k}={ab.get(p[k], p[k]) if not isinstance(p[k], float) else round(p[k], 3)}" for k in keys if k in p)


def s3(suffix=""):
    t = pd.read_csv(HERE / "results" / f"test_table{suffix}.csv")
    out = ["| правило | test acc | регрет к оракулу | test loss | MG − правило [95% CI] | MG W/T/L | медиана момента спада | параметры (калибровка) |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        name = f"**{r.rule}**" if r.rule == "MG" else r.rule
        ci = "" if r.rule == "MG" else f"{r.MG_minus_rule:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        out.append(f"| {name} | {f(r.test_acc, 4)} | {f(r.test_regret, 4)} | {f(r.test_loss)} | {ci} | "
                   f"{'' if r.rule == 'MG' else r['MG_wins/ties/losses']} | {f(r.median_decay_frac, 2)} | {short(r.get('params'))} |")
    return "\n".join(out)


def s3b(C):
    t = pd.read_csv(HERE / "results_anneal" / "s3b_tables.csv")
    t = t[t.C.astype(str) == str(C)]
    out = [f"oracle allocation (upper bound): acc {t.oracle_alloc_acc.iloc[0]:.4f}", "",
           "| правило | test acc | доля вычислений | выигрыш над фикс. фронтом [95% CI] | MG − правило, acc [95% CI] | MG − правило, вычисл. | параметры |",
           "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        name = f"**{r.rule}**" if r.rule == "MG" else r.rule
        d = "" if r.rule == "MG" else f"{r.MG_minus_rule_acc:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        out.append(f"| {name} | {f(r.test_acc, 4)} | {f(r.test_compute, 3)} | {r.gain_vs_fixed_frontier:+.4f} "
                   f"[{r.gain_ci_lo:+.4f}, {r.gain_ci_hi:+.4f}] | {d} | "
                   f"{'' if r.rule == 'MG' else f'{r.MG_minus_rule_compute:+.3f}'} | {short(r.params)} |")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    which = sys.argv[1]
    if which == "s3":
        print(s3())
    elif which == "s3pure":
        print(s3("_pure"))
    else:
        print(s3b(which))

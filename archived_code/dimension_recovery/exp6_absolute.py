"""Does any scalar metric recover k *numerically*, not just in rank order?

exp1 established that six observers order k = 1..6 correctly under favourable conditions.
Ordering is claim (1) of three. This file tests claim (2): that some metric estimates the
absolute number of active components, stably, across everything that can be varied.

The distinction is not pedantic. A statistic that is a monotone but unknown function of k
supports "something changed"; only a statistic that is numerically close to k, with a
bias that does not move when the nuisance parameters do, supports "there are k active
degrees of freedom". The second is what the paper wants to say.

Design: one base configuration, then one factor varied at a time (OFAT), because a full
factorial over eight factors is not affordable and OFAT is enough to show *whether* a
metric is stable -- any factor that moves it disqualifies it.

Reported per metric, pooled and per factor:

    bias      mean(d_hat - k)                    systematic offset
    MAE       mean|d_hat - k|                    absolute error
    MRE       mean|d_hat - k|/k                  relative error
    sd_seed   sd across seeds at fixed settings  run-to-run spread
    range     max - min of bias across levels    sensitivity to the factor
    rho       Spearman(d_hat, k)                 ordering only
    sync      d_hat(k independent) - d_hat(k synchronised)   can it tell them apart

    python exp6_absolute.py            # ~8 minutes
    python exp6_absolute.py base ofat  # named parts
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import estimators as E
import systems as S

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

BASE = dict(window=2000, cycles=300.0, snr=1e6, n=20000, amp=0.1, D=64,
            max_E=15, kn=5, tau=1, stride=2000)
KS = (1, 2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
METRICS = ("MG", "LB", "PR", "TwoNN", "roughness")


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def measure(cfg, k, seed, mode="quasiperiodic", observer="norm_fro", metrics=METRICS):
    """Median of each metric over sliding windows of one realisation."""
    info = S.make_system(mode, k=k, D=cfg["D"], n=cfg["n"],
                         cycles_per_window=cfg["cycles"], window=cfg["window"],
                         amp=cfg["amp"], seed=seed, band_mode="matched")
    x = S.observers(info, seed=seed, obs_snr=cfg["snr"])[observer]
    res = E.evaluate(x, cfg["window"], cfg["stride"], names=list(metrics),
                     max_E=cfg["max_E"], k=cfg["kn"], tau=cfg["tau"])
    return {m: res[m] for m in metrics}


def sweep(cfg, metrics=METRICS, seeds=SEEDS, mode="quasiperiodic"):
    """One (k x seed) block. Returns a long frame."""
    rows = []
    for k in KS:
        for seed in seeds:
            v = measure(cfg, k, seed, mode=mode, metrics=metrics)
            for m, val in v.items():
                rows.append({"k": k, "seed": seed, "metric": m, "value": val})
    return pd.DataFrame(rows)


def summarise(frame, label=""):
    """bias / MAE / MRE / sd across seeds / Spearman, per metric."""
    out = []
    for m, sub in frame.groupby("metric"):
        piv = sub.pivot_table(index="k", columns="seed", values="value")
        med = piv.median(axis=1)
        err = med - np.asarray(KS, float)[: len(med)]
        ok = np.isfinite(med)
        out.append({"label": label, "metric": m,
                    "bias": float(np.mean(err[ok])),
                    "MAE": float(np.mean(np.abs(err[ok]))),
                    "MRE": float(np.mean(np.abs(err[ok]) / np.asarray(KS)[: len(med)][ok])),
                    "sd_seed": float(piv.std(axis=1).mean()),
                    "rho": float(spearmanr(np.asarray(KS)[: len(med)][ok],
                                           med[ok]).statistic) if ok.sum() > 2 else np.nan})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ base configuration

def part_base():
    rule("BASE  Every metric against known k, at the most favourable settings found")
    print(f"  {BASE}\n")
    frame = sweep(BASE, metrics=METRICS + ("CorrDim",))
    frame.to_csv(OUT / "exp6_base_raw.csv", index=False)
    piv = frame.pivot_table(index="metric", columns="k", values="value", aggfunc="median")
    print(f"  {'metric':<11}" + "".join(f"{f'k={k}':>8}" for k in KS)
          + f"{'bias':>8}{'MAE':>7}{'MRE':>7}{'sd':>7}{'rho':>7}")
    s = summarise(frame, "base")
    for m in ("MG", "LB", "PR", "TwoNN", "CorrDim", "roughness"):
        if m not in piv.index:
            continue
        r = s[s.metric == m].iloc[0]
        print(f"  {m:<11}" + "".join(f"{piv.loc[m, k]:8.2f}" for k in KS)
              + f"{r.bias:>+8.2f}{r.MAE:>7.2f}{r.MRE:>7.0%}{r.sd_seed:>7.2f}{r.rho:>+7.2f}")
    s.to_csv(OUT / "exp6_base_summary.csv", index=False)
    print("\n  MAE is in units of 'active components'. A metric claiming to count them")
    print("  needs MAE well below 1; a metric with MAE above 1 is ordering k, not")
    print("  measuring it.")

    print("\n  Can any metric tell k independent oscillators from k synchronised ones?")
    sync = sweep(BASE, metrics=METRICS, mode="sync_phased")
    sync.to_csv(OUT / "exp6_base_sync.csv", index=False)
    ps = sync.pivot_table(index="metric", columns="k", values="value", aggfunc="median")
    print(f"  {'metric':<11}" + "".join(f"{f'k={k}':>8}" for k in KS)
          + "   (true dimension is 1 in every column)")
    for m in METRICS:
        print(f"  {m:<11}" + "".join(f"{ps.loc[m, k]:8.2f}" for k in KS))
    print(f"\n  {'metric':<11}{'gap at k=4':>12}{'gap at k=6':>12}"
          f"   independent minus synchronised")
    for m in METRICS:
        print(f"  {m:<11}{piv.loc[m, 4] - ps.loc[m, 4]:>12.2f}"
              f"{piv.loc[m, 6] - ps.loc[m, 6]:>12.2f}")


# --------------------------------------------------------------- one factor at a time

FACTORS = {
    "amp": (0.02, 0.1, 0.5),
    "cycles": (30.0, 100.0, 300.0, 1000.0),
    "snr": (1e3, 1e4, 1e6),
    "window": (1000, 2000, 4000),
    "n": (10000, 20000, 40000),
    "max_E": (10, 15, 20),
    "kn": (5, 10, 20),
    "tau": (1, 2, 3),
}


def part_ofat():
    rule("OFAT  One factor at a time: does any metric hold its calibration?")
    print("  Each block varies one factor and holds the rest at the base setting.")
    print("  'range of bias' is the span of the systematic offset across the levels of")
    print("  that factor -- the number that decides whether a metric is calibrated.\n")
    rows = []
    for factor, levels in FACTORS.items():
        for level in levels:
            cfg = {**BASE, factor: level}
            if factor == "window":
                cfg["stride"] = level
                cfg["n"] = max(BASE["n"], 10 * level)
            frame = sweep(cfg, seeds=(0, 1))
            s = summarise(frame, f"{factor}={level}")
            s["factor"], s["level"] = factor, level
            rows.append(s)
    allf = pd.concat(rows, ignore_index=True)
    allf.to_csv(OUT / "exp6_ofat.csv", index=False)

    for factor in FACTORS:
        sub = allf[allf.factor == factor]
        print(f"  -- {factor}")
        print(f"     {'metric':<11}" + "".join(f"{str(l):>10}" for l in FACTORS[factor])
              + f"{'range':>9}")
        for m in METRICS:
            v = [sub[(sub.metric == m) & (sub.level == l)].bias.iloc[0]
                 for l in FACTORS[factor]]
            print(f"     {m:<11}" + "".join(f"{x:>+10.2f}" for x in v)
                  + f"{max(v) - min(v):>9.2f}")
        print()

    print("  Pooled over every configuration tested:")
    print(f"  {'metric':<11}{'mean bias':>11}{'mean MAE':>10}{'mean MRE':>10}"
          f"{'worst MAE':>11}{'bias range':>12}{'mean rho':>10}")
    summary = []
    for m in METRICS:
        sub = allf[allf.metric == m]
        summary.append({"metric": m, "bias": sub.bias.mean(), "MAE": sub.MAE.mean(),
                        "MRE": sub.MRE.mean(), "worst_MAE": sub.MAE.max(),
                        "bias_range": sub.bias.max() - sub.bias.min(),
                        "rho": sub.rho.mean()})
        r = summary[-1]
        print(f"  {m:<11}{r['bias']:>+11.2f}{r['MAE']:>10.2f}{r['MRE']:>10.0%}"
              f"{r['worst_MAE']:>11.2f}{r['bias_range']:>12.2f}{r['rho']:>+10.2f}")
    pd.DataFrame(summary).to_csv(OUT / "exp6_pooled.csv", index=False)
    print("\n  A metric that estimated k would have mean bias near 0, MAE below ~0.5,")
    print("  and a bias range near 0. A metric that only orders k can have any bias so")
    print("  long as rho stays high. Read the last two columns together.")


# ------------------------------------------------------------------ observers at base

def part_obs():
    rule("OBS  Does the answer depend on which scalar function of W is logged?")
    rows = []
    for observer in ("norm_fro", "trace", "logdet", "proj_rand0", "proj_rand1"):
        for k in KS:
            for seed in SEEDS:
                v = measure(BASE, k, seed, observer=observer, metrics=("MG",))
                rows.append({"observer": observer, "k": k, "seed": seed,
                             "metric": "MG", "value": v["MG"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp6_observers.csv", index=False)
    piv = frame.pivot_table(index="observer", columns="k", values="value", aggfunc="median")
    print(f"  MG estimate by observer (median over {len(SEEDS)} seeds)\n")
    print(f"  {'observer':<12}" + "".join(f"{f'k={k}':>8}" for k in KS)
          + f"{'bias':>8}{'MAE':>7}")
    for o in piv.index:
        err = piv.loc[o].to_numpy() - np.asarray(KS, float)
        print(f"  {o:<12}" + "".join(f"{piv.loc[o, k]:8.2f}" for k in KS)
              + f"{err.mean():>+8.2f}{np.abs(err).mean():>7.2f}")
    print("\n  If the bias differs between observers of the same system, the number is a")
    print("  property of the observable and cannot be read as a count of components.")


PARTS = {"base": part_base, "ofat": part_ofat, "obs": part_obs}


def main(argv):
    for name in (argv[1:] or list(PARTS)):
        if name not in PARTS:
            print(f"unknown part {name!r}")
            return 2
        t0 = time.time()
        PARTS[name]()
        print(f"\n  [{name} took {time.time() - t0:.0f}s]")
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

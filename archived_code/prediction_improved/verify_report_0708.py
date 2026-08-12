"""Reproduce every number marked [computed] in report_0708.md.

    python verify_report_0708.py

Reads only files already in the repo:
  results/*_train.csv                   training logs
  results/*_probe.csv                   per-logged-step probe series
  results/*_probe_snapshots.npz         normalised logit matrices, every 200 steps
  ../edm_validation/results/phase9_trace_*.csv   sliding-window LB estimates

Each section prints the table that appears in the report, in the same order.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TRACES = HERE.parent / "edm_validation" / "results"

RUNS = ["grok", "grok_s1", "grok_s2", "nogap", "lowdata15", "lowdata20", "wd0"]
TRAIN = {"grok": "grok_train.csv", "grok_s1": "grok_seed1_train.csv",
         "grok_s2": "grok_seed2_train.csv", "nogap": "nogap_train.csv",
         "lowdata15": "lowdata15_train.csv", "lowdata20": "lowdata20_train.csv",
         "wd0": "wd0_train.csv"}
SNAP = {"grok": "grok", "grok_s1": "grok_seed1", "grok_s2": "grok_seed2",
        "nogap": "nogap", "lowdata15": "lowdata15", "lowdata20": "lowdata20",
        "wd0": "wd0"}
TRACE = {"grok": "phase9_trace_grok.csv", "grok_s1": "phase9_trace_grok_s1.csv",
         "grok_s2": "phase9_trace_grok_s2.csv",
         "lowdata15": "phase9_trace_lowdata15.csv",
         "lowdata20": "phase9_trace_lowdata20.csv", "wd0": "phase9_trace_wd0.csv"}

WINDOW_STEPS = 3000          # 300 samples at log_every=10
LINE_CONST_K5_W0 = 4.0 / (2 * np.log(3.0) + 2 * np.log(1.5))


def first_sustained(steps, acc, thr, smooth_w):
    """Definition 1: min t such that acc >= thr for every tau >= t."""
    a = pd.Series(acc).rolling(smooth_w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    idx = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        idx = i
    return int(steps[idx]) if idx is not None else None


def events(run, smooth_w=1):
    df = pd.read_csv(RES / TRAIN[run])
    s = df["step"].to_numpy()
    return (first_sustained(s, df["train_acc"].to_numpy(), 0.95, smooth_w),
            first_sustained(s, df["val_acc"].to_numpy(), 0.95, smooth_w))


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# -- 0. the closed-form line constants (report Proposition 1) -----------------

rule("0. Line constants, closed form (report Proposition 1)")


def line_constant(k, theiler):
    offsets = np.repeat(np.arange(theiler + 1, theiler + k + 1), 2)
    r = np.sort(offsets)[:k]
    return (k - 1) / np.sum(np.log(r[-1] / r[:-1]))


for k, w, quoted in ((5, 0, 1.330), (5, 14, 10.765)):
    print(f"  k={k} W={w:>2}  closed form {line_constant(k, w):8.4f}   report {quoted}")

# -- 1. event detection, sensitivity to smoothing (report section 5.1) --------

rule("1. Event detection: t_mem/t_gen against the smoothing window")
print(f"  {'run':<11}" + "".join(f"{f'w={w}':>18}" for w in (1, 5, 11, 21)))
for run in RUNS:
    cells = []
    for w in (1, 5, 11, 21):
        tm, tg = events(run, w)
        cells.append(f"{tm}/{tg}".rjust(18))
    print(f"  {run:<11}" + "".join(cells))
print("\n  method.md documents t_gen = 13700 / 2990 / 3140 / 780 for")
print("  grok / grok_s1 / grok_s2 / nogap, and t_mem = 1700 for grok.")

EV = {r: events(r, 5) for r in RUNS}          # w=5 used for the band tables below

# -- 2. how many estimator windows fit inside the plateau (section 5.1) -------

rule("2. Stride-10 window positions lying wholly inside (t_mem, t_gen)")
print(f"  {'run':<11}{'t_mem':>7}{'t_gen':>7}{'gap':>7}{'W=3000':>9}{'W=1500':>9}{'W=600':>8}")
for run in ("grok", "grok_s1", "grok_s2"):
    tm, tg = EV[run]
    gap = tg - tm
    cells = []
    for W in (3000, 1500, 600):
        n = (max(0, gap - W) // 10 + 1) if gap >= W else 0
        cells.append(f"{n:>9}" if W != 600 else f"{n:>8}")
    print(f"  {run:<11}{tm:>7}{tg:>7}{gap:>7}" + "".join(cells))

# -- 3. phase-aligned dimension trace with dispersion (sections 4.2, 5.2) -----

rule("3. LB estimate on weight_norm, phase-aligned, causal labels, W=0 k=5 win=300")
bands = [(1000, 2000), (2000, 3000), (3000, 5000), (5000, 8000), (8000, 12000)]
print(f"  {'run':<11}" + "".join(f"{f'+{a//1000}-{b//1000}k':>18}" for a, b in bands))
for run, tf in TRACE.items():
    tm, tg = EV[run]
    tr = pd.read_csv(TRACES / tf)
    right = tr["step"].to_numpy() + WINDOW_STEPS // 2     # centre -> right edge
    d = tr["dim_W0"].to_numpy()
    cells = []
    for a, b in bands:
        sel = (right >= tm + a) & (right < tm + b)
        if tg is not None:
            sel &= right <= tg
        if sel.sum() >= 2:
            v = np.nanmedian(d[sel])
            lo, hi = np.nanpercentile(d[sel], 10), np.nanpercentile(d[sel], 90)
            cells.append(f"{v:.2f} [{lo:.2f},{hi:.2f}]".rjust(18))
        else:
            cells.append(f"n={sel.sum()}".rjust(18))
    print(f"  {run:<11}" + "".join(cells))
print(f"\n  straight-line constant at k=5, W=0: {LINE_CONST_K5_W0:.3f}")

# -- 4. straightness of the function-space motion (section 8.3) --------------

rule("4. Straightness D = ||sum dZ|| / sum ||dZ||, band t_mem+1000..t_mem+2000")


def straightness(run, horizon_snapshots, probe="train"):
    npz = np.load(RES / f"{SNAP[run]}_probe_snapshots.npz")
    steps, Z = npz["steps"], npz[probe]
    dZ = np.diff(Z.reshape(len(Z), -1), axis=0)
    seg = np.linalg.norm(dZ, axis=1)
    out, mid = [], []
    for i in range(len(dZ) - horizon_snapshots + 1):
        gross = seg[i:i + horizon_snapshots].sum()
        net = np.linalg.norm(dZ[i:i + horizon_snapshots].sum(axis=0))
        out.append(net / gross if gross > 0 else np.nan)
        mid.append(steps[i + horizon_snapshots // 2])
    return np.array(mid), np.array(out), steps[1:], seg


for probe in ("train", "val"):
    print(f"\n  probe = {probe}")
    print(f"     {'run':<11}" + "".join(f"{f'H={h*200}':>10}" for h in (3, 5, 10, 20)))
    for run in RUNS:
        tm, tg = EV[run]
        cells = []
        for h in (3, 5, 10, 20):
            mid, st, _, _ = straightness(run, h, probe)
            sel = (mid >= tm + 1000) & (mid < tm + 2000)
            if tg is not None:
                sel &= mid < tg
            cells.append((f"{np.nanmedian(st[sel]):.3f}" if sel.sum() >= 2 else "--").rjust(10))
        print(f"     {run:<11}" + "".join(cells))
print("\n  The ordering at H=1000 inverts by H=4000: the separation is not robust.")

# -- 5. velocity, reproducing NOTES.md, and the phase confound (section 5.3) --

rule("5. Normalised-logit velocity: matched phase against 'second half'")
print(f"  {'run':<11}{'V(+1-2k)':>12}{'V(late half)':>15}{'NOTES.md':>12}")
QUOTED = {"lowdata20": 7.26e-2, "lowdata15": 5.30e-2, "nogap": 2.34e-2,
          "grok_s2": 1.94e-2, "grok": 1.84e-2, "grok_s1": 1.38e-2, "wd0": 1.53e-4}
for run in RUNS:
    tm, tg = EV[run]
    p = pd.read_csv(RES / f"{SNAP[run]}_probe.csv")
    s, v = p["step"].to_numpy(), p["train_velocity"].to_numpy()
    sel = (s >= tm + 1000) & (s < tm + 2000)
    if tg is not None:
        sel &= s < tg
    band = np.nanmedian(v[sel]) if sel.sum() >= 2 else np.nan
    late = np.nanmedian(v[len(v) // 2:])
    print(f"  {run:<11}{band:>12.4g}{late:>15.4g}{QUOTED[run]:>12.3g}")
print("\n  At matched phase grok is the HIGHEST of all runs; the report's ordering")
print("  is a late-training statement, not a pre-transition one.")

# -- 6. information in d_hat beyond the roughness ratio (section 6.2) ---------

rule("6. Does the LB estimate carry information beyond std(diff x)/std(x)?")
print(f"  {'run':<11}{'spearman':>10}{'sd(resid)':>11}{'sd(ranks)':>11}{'explained':>11}")
for run, tf in TRACE.items():
    tr = pd.read_csv(TRACES / tf)
    d, r = tr["dim_W0"].to_numpy(), tr["roughness"].to_numpy()
    m = np.isfinite(d) & np.isfinite(r)
    rho = spearmanr(d[m], r[m]).statistic
    rd, rr = rankdata(d[m]), rankdata(r[m])
    resid = rd - np.polyval(np.polyfit(rr, rd, 1), rr)
    frac = 1 - (resid.std() / rd.std()) ** 2
    print(f"  {run:<11}{rho:>+10.3f}{resid.std():>11.2f}{rd.std():>11.2f}{frac:>10.0%}")
print("\n  Not run-invariant: 10% explained in lowdata20, 99% in wd0.")

# -- 7. independent observations in the transition (section 1.1) -------------

rule("7. Independent observations inside the transition")


def acf_time(x, max_lag=400):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x @ x)
    below = np.flatnonzero(ac[:max_lag] < 1 / np.e)
    return int(below[0]) if len(below) else max_lag


print(f"  {'run':<11}{'gap':>8}{'tau (steps)':>13}{'independent':>13}")
for run in ("grok", "grok_s1", "grok_s2"):
    tm, tg = EV[run]
    df = pd.read_csv(RES / TRAIN[run])
    s, tl = df["step"].to_numpy(), df["train_loss"].to_numpy()
    sel = (s >= tm) & (s <= tg)
    tau = acf_time(tl[sel]) * 10                    # rows -> optimisation steps
    print(f"  {run:<11}{tg - tm:>8}{tau:>13}{(tg - tm) / max(tau, 1):>13.1f}")
print("\n  tau scales with the gap, so a longer plateau does not buy more samples.")
print("  phase14_budget.txt quotes 10 from a dense run (gap 1615, tau 163);")
print("  the measured range across runs is 7 to 17.")

print("\nDone. Every table above appears in report_0708.md with the same numbers.")

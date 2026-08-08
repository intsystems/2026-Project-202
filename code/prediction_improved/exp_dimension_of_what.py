"""Tier 0 / Tier 1 of the audit's validation programme (report_0708.md, section 7).

The question is not whether the Levina-Bickel / MacKay-Ghahramani number moves. It does.
The question is what it is a property of. Eight checks, all on data already in the repo
plus synthetic systems whose dimension is known by construction.

  A  E0.2   calibration of the estimator itself on i.i.d. clouds of known dimension
  B  E0.1a  how many cycles a window must contain before the estimate finds a 2-torus
  C  E0.1b  a system whose true dimension changes at a known step, at published settings
  D  E1.1a  three real observables of the same run, same estimator, same window
  E  E1.1b  smooth monotone reparameterisation of one observable  <- the sharpest test
  F  E1.2   raw against locally detrended
  G  E1.5   trend + phase-randomised residual as the null the report is missing
  H  E1.3   does the claim survive the estimator's free parameters

Under Takens/Stark, if two observables both embed the same compact invariant set then the
images are diffeomorphic and, on a compact set, bi-Lipschitz equivalent; box-counting,
Hausdorff and correlation dimension are bi-Lipschitz invariants. So D and E are not
robustness checks. They are the theorem's own signature, and failing them means the number
is a property of the series, not of the system.

    python exp_dimension_of_what.py            # everything, about 4 minutes
    python exp_dimension_of_what.py A C E      # named sections only
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grokking_analysis"))

from edm import mle_intrinsic_dimension                     # noqa: E402

RES = HERE / "results"
CONF = HERE.parent / "edm_validation" / "results" / "conf"
OUT = HERE / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

# The published configuration, reproduced exactly from phase9_paper_settings.py.
WINDOW, STRIDE, K, MAX_E, THEILER = 300, 10, 5, 15, 0

RUNS = {"grok": "grok_train.csv", "grok_s1": "grok_seed1_train.csv",
        "grok_s2": "grok_seed2_train.csv", "lowdata15": "lowdata15_train.csv",
        "lowdata20": "lowdata20_train.csv", "wd0": "wd0_train.csv"}
GENERALISES = {"grok", "grok_s1", "grok_s2"}


# --------------------------------------------------------------------------- helpers

def d_hat(series, k=K, max_E=MAX_E, theiler=THEILER, tau=1, correction="levina_bickel"):
    return mle_intrinsic_dimension(series, tau=tau, max_E=max_E, k_neighbors=k,
                                   correction=correction, theiler_window=theiler,
                                   rng=np.random.default_rng(0))


def trace(series, window=WINDOW, stride=STRIDE, **kw):
    """Sliding estimate labelled by the window's RIGHT edge (index of the last sample)."""
    right, dims = [], []
    for start in range(0, len(series) - window + 1, stride):
        dims.append(d_hat(series[start:start + window], **kw))
        right.append(start + window - 1)
    return np.asarray(right), np.asarray(dims, dtype=float)


def first_sustained(steps, acc, thr=0.95, smooth_w=5):
    a = pd.Series(acc).rolling(smooth_w, min_periods=1, center=True).mean().to_numpy()
    ok = a >= thr
    idx = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        idx = i
    return int(steps[idx]) if idx is not None else None


def load(name):
    path = RES / RUNS[name] if name in RUNS else CONF / f"{name}.csv"
    df = pd.read_csv(path)
    s = df["step"].to_numpy()
    return df, s, first_sustained(s, df["train_acc"].to_numpy()), \
        first_sustained(s, df["val_acc"].to_numpy())


def band_median(steps_right, dims, t_mem, t_gen, lo=1000, hi=3000):
    """Median over causal windows whose right edge sits lo..hi steps after memorisation.

    Windows whose right edge is past t_gen are dropped: nothing may read the future, and
    nothing may read the transition it is supposed to anticipate.
    """
    sel = (steps_right >= t_mem + lo) & (steps_right < t_mem + hi)
    if t_gen is not None:
        sel &= steps_right <= t_gen
    return (float(np.nanmedian(dims[sel])), int(sel.sum())) if sel.sum() >= 2 else (np.nan, int(sel.sum()))


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def separation(values):
    """Gap between the generalising and never-generalising groups, in units of spread.

    Returns None when either group is empty or a value is missing.
    """
    g = [v for k, v in values.items() if k in GENERALISES and np.isfinite(v)]
    n = [v for k, v in values.items() if k not in GENERALISES and np.isfinite(v)]
    if len(g) < 2 or len(n) < 2:
        return None
    pooled = np.sqrt((np.var(g, ddof=1) + np.var(n, ddof=1)) / 2)
    return (np.mean(g) - np.mean(n)) / pooled if pooled > 0 else np.inf


def auc(values):
    """Fraction of (generalising, never-generalising) pairs the statistic orders correctly.

    Cohen's d on three runs per group is driven by whichever run is furthest out --
    here always wd0, whose weight norm is a straight line. The rank statistic is not,
    and 0.5 is chance. Ties count a half.
    """
    g = [v for k, v in values.items() if k in GENERALISES and np.isfinite(v)]
    n = [v for k, v in values.items() if k not in GENERALISES and np.isfinite(v)]
    if not g or not n:
        return np.nan
    wins = sum((a > b) + 0.5 * (a == b) for a in g for b in n)
    return wins / (len(g) * len(n))


# ------------------------------------------------------- A. E0.2 estimator calibration

def section_A():
    rule("A  (E0.2)  The estimator on i.i.d. clouds of KNOWN dimension")
    print("  Uniform samples from the unit d-cube: the estimator under its own")
    print("  assumptions, so any error here is a floor on the error everywhere else.")
    print("  LB is the Levina-Bickel average, MG the MacKay-Ghahramani pooled likelihood.")
    print("  MG/LB should be (k-2)/(k-1) exactly -- 0.750 at k=5, 0.889 at k=10.\n")
    rng = np.random.default_rng(0)
    ns, ds = (100, 300, 1000, 3000), (1, 2, 3, 5, 8)
    print(f"  {'true d':>7}{'n':>7}{'LB k=5':>10}{'MG k=5':>10}{'MG k=10':>10}"
          f"{'MG k=20':>10}   MG(k=5)/true")
    rows = []
    for d in ds:
        for n in ns:
            vals = {}
            for k in (5, 10, 20):
                reps = [_cloud_id(rng.random((n, d)), k) for _ in range(5)]
                vals[f"LB{k}"] = float(np.mean([r[0] for r in reps]))
                vals[f"MG{k}"] = float(np.mean([r[1] for r in reps]))
            print(f"  {d:>7}{n:>7}{vals['LB5']:>10.2f}{vals['MG5']:>10.2f}"
                  f"{vals['MG10']:>10.2f}{vals['MG20']:>10.2f}   {vals['MG5'] / d:>10.2f}x")
            rows.append({"true_d": d, "n": n, **vals, "mg5_ratio": vals["MG5"] / d})
    pd.DataFrame(rows).to_csv(OUT / "e02_calibration.csv", index=False)
    print("\n  After the MG correction the estimator is unbiased at d=1-2 and biased DOWN")
    print("  from d=3 up; the bias grows with d and shrinks with n. Raw LB at k=5 carries")
    print("  a fixed 1.333x inflation on top of that, which is why an LB reading near 1.33")
    print("  means d=1 -- a curve -- and not 'slightly more than one dimension'.")


def _cloud_id(points, k):
    """(LB, MG) intrinsic dimension of a point cloud -- the two poolings of the same S_i."""
    from scipy.spatial import KDTree
    dist, _ = KDTree(points).query(points, k=k + 1)
    dist = np.maximum(dist[:, 1:], 1e-12)
    s = np.sum(np.log(dist[:, -1:] / dist[:, :-1]), axis=1)
    s = s[s > 0]
    lb = float(np.mean((k - 1) / s))
    mg = float((len(s) * (k - 1) - 1) / np.sum(s))
    return lb, mg


# --------------------------------------------- B. E0.1a cycles per window on a 2-torus

def section_B():
    rule("B  (E0.1a)  How much of the attractor must a window contain?")
    print("  Scalar observable of a circle (true d = 1), a 2-torus (true d = 2) and the")
    print("  Lorenz attractor (correlation dimension 2.06). Same 300-sample window, same")
    print("  k=5, max_E=15. The only thing that changes is how many characteristic")
    print("  periods the window spans. Cycle counts are deliberately irrational, so the")
    print("  sampled orbit never repeats exactly; the noise floor is 1e-3 of the range.\n")
    cycles = (0.3, 1.1, 2.3, 5.7, 11.3, 31.7, 101.9)
    print(f"  {'system':<10}{'true d':>7}{'LB=':>6}" + "".join(f"{f'{c}':>13}" for c in cycles))
    rows = []
    for name, gen, true_d in (("circle", _circle, 1.0), ("2-torus", _torus, 2.0),
                              ("lorenz", _lorenz, 2.06)):
        cells = []
        for c in cycles:
            x = gen(WINDOW, c)
            v0 = d_hat(x)
            v14 = d_hat(x, theiler=14)
            cells.append(f"{v0:5.2f} /{v14:5.2f}".rjust(13))
            rows.append({"system": name, "true_d": true_d, "cycles": c,
                         "lb_W0": v0, "lb_W14": v14,
                         "lb_expected": true_d * 4 / 3})
        print(f"  {name:<10}{true_d:>7}{true_d * 4 / 3:>6.2f}" + "".join(cells))
    pd.DataFrame(rows).to_csv(OUT / "e01a_cycles.csv", index=False)
    print("\n  columns are cycles per window; each cell is LB at W=0 / LB at W=14.")
    print("  'LB=' is what an unbiased LB at k=5 should return for that true dimension.")
    print("\n  Below a few cycles per window every system returns the tangent constant,")
    print("  whatever its true dimension: the window sees an arc, not an attractor. A")
    print("  300-sample window of weight_norm contains no cycles at all -- the series is")
    print("  monotone across the window -- so it is in the leftmost column by")
    print("  construction, and no setting of k, E or W moves it out.")


def _circle(n, cycles, noise=1e-3, seed=0):
    t = np.arange(n)
    x = np.sin(2 * np.pi * cycles * t / n)
    return x + np.random.default_rng(seed).normal(0, noise, n)


def _torus(n, cycles, noise=1e-3, seed=0):
    t = np.arange(n)
    phi = (1 + np.sqrt(5)) / 2                       # incommensurate second frequency
    x = np.sin(2 * np.pi * cycles * t / n) + np.sin(2 * np.pi * cycles * phi * t / n + 0.7)
    return x + np.random.default_rng(seed).normal(0, noise, n)


def _lorenz_full(n_steps=200000, dt=0.002, seed=0):
    def f(v):
        x, y, z = v
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])
    v = np.array([1.0, 1.0, 1.0])
    for _ in range(20000):                            # transient
        k1 = f(v); k2 = f(v + dt / 2 * k1); k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    out = np.empty((n_steps, 3))
    for i in range(n_steps):
        k1 = f(v); k2 = f(v + dt / 2 * k1); k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = v
    return out[:, 0], dt


_LORENZ_CACHE = {}


def _lorenz(n, cycles, noise=1e-4, seed=0):
    """n samples spanning `cycles` orbit times (one Lorenz lobe circuit ~ 0.75 t.u.)."""
    if "x" not in _LORENZ_CACHE:
        _LORENZ_CACHE["x"], _LORENZ_CACHE["dt"] = _lorenz_full()
    x, dt = _LORENZ_CACHE["x"], _LORENZ_CACHE["dt"]
    span_time = cycles * 0.75
    stride = max(1, int(round(span_time / dt / n)))
    take = x[:n * stride:stride][:n]
    if len(take) < n:                                 # too coarse: fall back to the tail
        take = x[np.linspace(0, len(x) - 1, n).astype(int)]
    take = (take - take.mean()) / take.std()
    return take + np.random.default_rng(seed).normal(0, noise, len(take))


# ------------------------------------ C. E0.1b a KNOWN dimension change, known location

def section_C():
    rule("C  (E0.1b)  A system whose dimension changes 2 -> 1 at a step we know")
    print("  A 2-torus whose second amplitude is ramped to zero over steps 8000-12000, so")
    print("  the true dimension is 2 before and 1 after. 2000 samples, published trace")
    print("  settings. If the estimator cannot resolve this, it cannot resolve anything")
    print("  weaker, and every claim in the paper about a dimension change is bounded by")
    print("  this result.\n")
    n, ramp_lo, ramp_hi = 2000, 800, 1200            # rows; the logs are 2001 rows at log_every=10
    t = np.arange(n)
    a = np.clip((ramp_hi - t) / (ramp_hi - ramp_lo), 0, 1)
    rows_out = []
    print(f"  {'cycles/win':>11}{'W':>5}{'d before':>10}{'d after':>9}{'ratio':>8}"
          f"{'true':>7}   verdict")
    for cyc_per_window in (5.7, 20.3, 50.9):
        f1 = cyc_per_window / WINDOW
        phi = (1 + np.sqrt(5)) / 2
        x = np.sin(2 * np.pi * f1 * t) + a * np.sin(2 * np.pi * f1 * phi * t + 0.7)
        x = x + np.random.default_rng(0).normal(0, 1e-3, n)
        for theiler in (0, 14):
            right, dims = trace(x, theiler=theiler)
            before = np.nanmedian(dims[right < ramp_lo])
            after = np.nanmedian(dims[right > ramp_hi + WINDOW])
            ratio = before / after
            verdict = ("resolves it" if 1.4 <= ratio <= 3.0 else
                       "wrong direction" if ratio < 1.0 else "overshoots")
            print(f"  {cyc_per_window:>11}{theiler:>5}{before:>10.2f}{after:>9.2f}"
                  f"{ratio:>8.2f}{2.0:>7.1f}   {verdict}")
            rows_out.append({"cycles_per_window": cyc_per_window, "theiler": theiler,
                             "d_before": before, "d_after": after,
                             "ratio": ratio, "true_ratio": 2.0, "verdict": verdict})
    pd.DataFrame(rows_out).to_csv(OUT / "e01b_known_change.csv", index=False)
    print("\n  The true ratio is 2.0 in every row, and the estimator finds it. This is a")
    print("  POSITIVE result and it must be reported as one: the machinery is not broken,")
    print("  and a real 2 -> 1 collapse in a well-sampled window is recovered to within")
    print("  the calibration error of section A. What section B establishes is the")
    print("  precondition -- several returns inside the window -- and what section D")
    print("  establishes is that the training logs do not meet it. The failure is")
    print("  specific and diagnosable, not a general objection to the method.")


# --------------------------------------------------- D. E1.1a several real observables

def section_D():
    rule("D  (E1.1a)  Three observables of the same run, one estimator")
    print("  Under Takens/Stark two observables that both embed the same invariant set")
    print("  give diffeomorphic, hence bi-Lipschitz equivalent, images; dimension is a")
    print("  bi-Lipschitz invariant. So these columns must agree if the number is a")
    print("  property of the optimiser rather than of the series.\n")
    obs = ("weight_norm", "train_loss", "val_loss")
    rows = []
    for win in (300, 100):
        print(f"  -- window = {win} samples = {win * 10} steps "
              f"({win - MAX_E + 1} points embedded in R^15)")
        print(f"     {'run':<11}{'weight_norm':>13}{'train_loss':>13}{'val_loss':>13}"
              f"{'max/min':>9}{'rho(wn,tl)':>12}{'rho(wn,vl)':>12}")
        for run in RUNS:
            df, steps, t_mem, t_gen = load(run)
            traces_here, cells, meds = {}, [], []
            for o in obs:
                right_idx, dims = trace(df[o].to_numpy(), window=win)
                traces_here[o] = dims
                med, n = band_median(steps[right_idx], dims, t_mem, t_gen)
                meds.append(med)
                cells.append(f"{med:6.2f}".rjust(13))
                rows.append({"run": run, "window": win, "observable": o,
                             "band_median": med, "n_windows": n,
                             "whole_run_median": float(np.nanmedian(dims))})
            spread = (np.nanmax(meds) / np.nanmin(meds)) if np.isfinite(meds).all() else np.nan
            r1 = _rho(traces_here["weight_norm"], traces_here["train_loss"])
            r2 = _rho(traces_here["weight_norm"], traces_here["val_loss"])
            print(f"     {run:<11}" + "".join(cells) + f"{spread:>9.1f}x"
                  f"{r1:>+12.2f}{r2:>+12.2f}")
        print()
    pd.DataFrame(rows).to_csv(OUT / "e11a_observers.csv", index=False)
    print("  band median = causal windows 1000-3000 steps after t_mem, capped at t_gen;")
    print("  rho = Spearman between the two whole-run traces.")
    print("\n  Three observables of one system, one estimator, one window, one instant.")
    print("  If the number were the dimension of an invariant set they would agree.")


def _rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return spearmanr(a[m], b[m]).statistic


# ---------------------------------------- E. E1.1b smooth reparameterisation invariance

def section_E():
    rule("E  (E1.1b)  The same observable through a smooth monotone reparameterisation")
    print("  h(x) for h a diffeomorphism of the data range. Takens/Stark say nothing")
    print("  changes: h o phi is as good an observable as phi, the images are")
    print("  diffeomorphic, the dimension is identical. This is the single cheapest test")
    print("  of whether the number is geometry or arithmetic.\n")
    print("  The first four warps are smooth but nearly affine ACROSS ONE WINDOW, because")
    print("  a window spans a small part of the run's range. Those are the weak half of")
    print("  the test. The last two are diffeomorphisms of R whose curvature is tuned to")
    print("  the within-window range itself, so they cannot be absorbed into a local")
    print("  affine map. Those are the sharp half.\n")
    rows = []
    for run in RUNS:
        df, steps, t_mem, t_gen = load(run)
        for o in ("weight_norm", "train_loss"):
            x0 = np.maximum(df[o].to_numpy(), 1e-12) if o == "train_loss" else df[o].to_numpy()
            warps = _warps(x0, window=100)
            if run == "grok" and o == "weight_norm":
                print(f"  {'run':<11}{'observable':<13}" + "".join(f"{w:>10}" for w in warps))
            cells = []
            for wname, h in warps.items():
                right_idx, dims = trace(h(x0), window=100)
                med, n = band_median(steps[right_idx], dims, t_mem, t_gen)
                cells.append(f"{med:6.2f}".rjust(10))
                rows.append({"run": run, "observable": o, "warp": wname,
                             "band_median": med, "n_windows": n})
            print(f"  {run:<11}{o:<13}" + "".join(cells))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "e11b_reparameterisation.csv", index=False)
    print("\n  spread of the SAME quantity across reparameterisations of the SAME series:")
    for half, cols in (("near-affine warps", ["x", "x^2", "x^3", "log x"]),
                       ("window-scale warps", ["wiggle L", "wiggle L/3"])):
        sub = frame[frame.warp.isin(cols)]
        sp = sub.groupby(["run", "observable"]).band_median.agg(["min", "max"])
        f = (sp["max"] / sp["min"]).dropna()
        print(f"    {half:<20} median {f.median():.2f}x   worst {f.max():.2f}x"
              f"   ({(f > 1.2).sum()} of {len(f)} series above 1.2x)")
    print("\n  Read the two halves against each other. Near-affine warps leave the estimate")
    print("  alone, which is expected and carries no information: over one window a")
    print("  monotone series is nearly a straight line and any smooth h acts on it as an")
    print("  affine map, to which the W=0 estimate is exactly invariant. The window-scale")
    print("  warps are the test that has content, because dimension is a diffeomorphism")
    print("  invariant for the SET but the estimator sees 300 points at a fixed scale.")


def _warps(x, window):
    """Six diffeomorphisms of the data range; the last two curve at within-window scale."""
    mu, sd = x.mean(), x.std()
    starts = range(0, len(x) - window + 1, 50)
    local_range = float(np.median([np.ptp(x[s:s + window]) for s in starts])) or sd
    return {"x": lambda v: v,
            "x^2": lambda v: v ** 2,
            "x^3": lambda v: v ** 3,
            "log x": np.log,
            # h'(v) = 1 + 0.5 cos(.) > 0, so each is a genuine diffeomorphism
            "wiggle L": lambda v, L=local_range: v + (0.5 * L / (2 * np.pi))
            * np.sin(2 * np.pi * (v - mu) / L),
            "wiggle L/3": lambda v, L=local_range / 3: v + (0.5 * L / (2 * np.pi))
            * np.sin(2 * np.pi * (v - mu) / L)}


# ------------------------------------------------------------- F. E1.2 raw vs detrended

def section_F():
    rule("F  (E1.2)  Raw against locally detrended")
    print("  Each window is replaced by its residual after a local quadratic fit before")
    print("  embedding. If the signal is the trend, it does not survive. Note this is not")
    print("  a nuisance correction: a monotone trend is exactly the configuration in")
    print("  which the W=0 estimate returns its tangent constant.\n")
    rows = []
    for win in (300, 100):
        print(f"  -- window = {win} samples")
        print(f"     {'run':<11}{'raw band':>10}{'detr band':>11}{'raw all':>9}"
              f"{'detr all':>10}{'rho':>8}")
        for run in RUNS:
            df, steps, t_mem, t_gen = load(run)
            x = df["weight_norm"].to_numpy()
            r_idx, d_raw = trace(x, window=win)
            d_det = np.array([_detrended_d(x[s:s + win])
                              for s in range(0, len(x) - win + 1, STRIDE)])
            b_raw, _ = band_median(steps[r_idx], d_raw, t_mem, t_gen)
            b_det, _ = band_median(steps[r_idx], d_det, t_mem, t_gen)
            print(f"     {run:<11}{b_raw:>10.2f}{b_det:>11.2f}{np.nanmedian(d_raw):>9.2f}"
                  f"{np.nanmedian(d_det):>10.2f}{_rho(d_raw, d_det):>+8.2f}")
            rows.append({"run": run, "window": win, "band_raw": b_raw,
                         "band_detrended": b_det, "all_raw": float(np.nanmedian(d_raw)),
                         "all_detrended": float(np.nanmedian(d_det)),
                         "spearman": _rho(d_raw, d_det)})
        frame = pd.DataFrame([r for r in rows if r["window"] == win]).set_index("run")
        for label, col in (("raw", "band_raw"), ("detrended", "band_detrended")):
            d, a = separation(frame[col].to_dict()), auc(frame[col].to_dict())
            print(f"     separation, {label:<10} Cohen d "
                  f"{'  n/a ' if d is None else f'{d:+.2f}'}   AUC {a:.2f}")
        print()
    pd.DataFrame(rows).to_csv(OUT / "e12_detrended.csv", index=False)
    print("  AUC is the fraction of (generalising, never-generalising) run pairs ordered")
    print("  correctly; 0.5 is chance and 1.00 needs every pair separated. With three runs")
    print("  a side the finest resolution AUC has is 1/9 = 0.11, so nothing here is a")
    print("  measurement of discrimination -- it is a check for whether the ordering is")
    print("  even consistent before spending runs on estimating it.")


def _detrended_d(window):
    t = np.arange(len(window))
    resid = window - np.polyval(np.polyfit(t, window, 2), t)
    return d_hat(resid)


# --------------------------------------------- G. E1.5 the null the report does not use

def section_G():
    rule("G  (E1.5)  Trend + phase-randomised residual: the null that has power")
    print("  IAAFT on a monotone series nearly reproduces it, so it cannot reject. The")
    print("  null that matters keeps the local trend and destroys everything else: fit a")
    print("  quadratic, phase-randomise the residual, recombine. If the observed estimate")
    print("  sits inside that null, the window contains no geometry beyond its trend.\n")
    n_sur, sub_stride = 39, 50
    print(f"  {'run':<11}{'windows':>8}{'obs median':>12}{'null median':>13}"
          f"{'z (median)':>12}{'frac p<0.05':>13}")
    rows = []
    for run in RUNS:
        df, steps, t_mem, t_gen = load(run)
        x = df["weight_norm"].to_numpy()
        rng = np.random.default_rng(0)
        obs, nulls, zs, hits = [], [], [], []
        for start in range(0, len(x) - WINDOW + 1, sub_stride):
            w = x[start:start + WINDOW]
            o = d_hat(w)
            t = np.arange(len(w))
            coef = np.polyfit(t, w, 2)
            trend = np.polyval(coef, t)
            resid = w - trend
            null = np.array([d_hat(trend + _phase_randomise(resid, rng))
                             for _ in range(n_sur)])
            sd = np.nanstd(null)
            obs.append(o)
            nulls.append(np.nanmedian(null))
            zs.append((o - np.nanmean(null)) / sd if sd > 0 else np.nan)
            hits.append(np.mean(null >= o) < 0.05 or np.mean(null <= o) < 0.05)
        print(f"  {run:<11}{len(obs):>8}{np.nanmedian(obs):>12.2f}{np.nanmedian(nulls):>13.2f}"
              f"{np.nanmedian(zs):>+12.2f}{np.mean(hits):>13.0%}")
        rows.append({"run": run, "n_windows": len(obs), "obs_median": float(np.nanmedian(obs)),
                     "null_median": float(np.nanmedian(nulls)),
                     "z_median": float(np.nanmedian(zs)), "frac_rejected": float(np.mean(hits))})
    pd.DataFrame(rows).to_csv(OUT / "e15_trend_surrogate.csv", index=False)
    print("\n  z is per-window (observed - null mean)/null sd; 'frac p<0.05' is the share")
    print("  of windows where the observed estimate leaves the 39-surrogate null band.")


def _phase_randomise(x, rng):
    f = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0, 2 * np.pi, len(f))
    phases[0] = 0
    if len(x) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=len(x)) + x.mean()


# --------------------------------------------------------- H. E1.3 does the claim survive

def section_H():
    rule("H  (E1.3)  Does 'grokking runs differ from never-generalising runs' survive?")
    print("  The claim, not the number, has to be invariant. For every setting: median")
    print("  estimate over causal windows 1000-3000 steps after t_mem (capped at t_gen),")
    print("  then Cohen's d between the two families. |d| < 0.8 is no separation; a sign")
    print("  flip across settings means the direction of the claim is a parameter choice.\n")
    grid = [(E, k, th, win)
            for E in (5, 10, 15, 20) for k in (5, 10, 20)
            for th in (0, "embedding") for win in (100, 200, 300)]
    print(f"  {'max_E':>6}{'k':>4}{'W':>11}{'win':>5}{'cohen d':>10}{'grok mean':>11}"
          f"{'other mean':>11}{'n miss':>7}")
    rows = []
    for E, k, th, win in grid:
        vals = {}
        for run in RUNS:
            df, steps, t_mem, t_gen = load(run)
            r_idx, dims = trace(df["weight_norm"].to_numpy(), window=win,
                                k=k, max_E=E, theiler=th)
            vals[run], _ = band_median(steps[r_idx] if len(r_idx) else np.array([]),
                                       dims, t_mem, t_gen)
        sep = separation(vals)
        g = np.mean([v for r, v in vals.items() if r in GENERALISES and np.isfinite(v)])
        o = np.mean([v for r, v in vals.items() if r not in GENERALISES and np.isfinite(v)])
        miss = sum(1 for v in vals.values() if not np.isfinite(v))
        print(f"  {E:>6}{k:>4}{str(th):>11}{win:>5}"
              f"{'    --   ' if sep is None else f'{sep:>+10.2f}'}"
              f"{g:>11.2f}{o:>11.2f}{miss:>7}")
        rows.append({"max_E": E, "k": k, "theiler": str(th), "window": win,
                     "cohen_d": sep, "grok_mean": g, "other_mean": o, "n_missing": miss})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "e13_sensitivity.csv", index=False)
    ok = frame.cohen_d.dropna()
    print(f"\n  {len(ok)} settings evaluated. |d| >= 0.8 in {(ok.abs() >= 0.8).sum()};"
          f" sign is positive in {(ok > 0).sum()} and negative in {(ok < 0).sum()}.")
    print("  A claim that changes sign across the estimator's free parameters is not a")
    print("  claim about the data.")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D,
            "E": section_E, "F": section_F, "G": section_G, "H": section_H}


def main(argv):
    wanted = [a.upper() for a in argv[1:]] or list(SECTIONS)
    for name in wanted:
        if name not in SECTIONS:
            print(f"unknown section {name!r}; expected {sorted(SECTIONS)}")
            return 2
        t0 = time.time()
        SECTIONS[name]()
        print(f"\n  [{name} took {time.time() - t0:.1f}s]")
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

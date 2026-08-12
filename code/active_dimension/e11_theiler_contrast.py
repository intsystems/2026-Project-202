"""E11 -- is the transient's active dimension undefined, or merely unidentifiable?

The paper (\\cref{tab:classes}) says the active dimension is *undefined* on a decaying
transient, "a curve, never revisited", and reports MG ~ 29 there.  A reviewer objects that the
Levina-Bickel MLE requires no recurrence as a mathematical condition: it requires enough points
sampled locally from the set.  A decaying transient traces a perfectly good one-dimensional
curve, so its intrinsic dimension is 1 and it is well defined.  What goes wrong is specific to
this protocol -- there is one time series, and the Theiler exclusion discards every temporally
adjacent point.  On a curve that is never revisited the temporal neighbours *are* the only near
neighbours, so after the exclusion nothing near remains, the log-distance ratios collapse, and
the reciprocal blows up.

The design is a two-by-two on the same points with the same estimator, changing only the
Theiler exclusion, plus a sweep of the exclusion between the two endpoints:

                     W_T = 0                        W_T = the frozen setting
    recurrent arm    the tangent, or r if the       r  (the published result)
                     record is not oversampled
    transient arm    ~1, the curve's true dim       ~29 (the published result)

Both diagonal cells must reproduce the paper's published numbers or the experiment is not
measuring what it claims; ``--report`` prints that check explicitly against
``results/e2_rank_sweep/sweep_raw.csv``.

Everything is measured with the canonical estimator: the frozen ``MGConfig`` of E1, the delay
embedding, dither, distance floor, Theiler resolution and neighbour search of
``grokking_analysis/edm``, and the MG / LB pooling of ``mg.all_estimators``.  For speed the ten
exclusions of one window share a single KD-tree query -- ``edm._neighbor_distances`` queries
``k + 2 W_T + 1`` candidates and keeps the first ``k`` that survive the exclusion, so querying
once at the largest ``W_T`` and filtering gives the identical neighbour set at every smaller
one.  ``--verify`` proves that against ``mg.all_estimators`` cell by cell rather than asserting
it.

Recorded per cell, because "the estimate blew up" is a claim and "and here is the mechanism" is
a measurement: the realised exclusion in samples (``theiler`` is ``"autocorr"`` in the frozen
configuration, so it is data-dependent and capped at ``mg.THEILER_CAP``), the neighbour pairs
that survive, the median nearest-neighbour distance before and after the exclusion, the median
log-ratio sum whose collapse is the failure, the degeneracy flag and PRdelay.

    python e11_theiler_contrast.py --simulate      # ~15 min on 9 cores
    python e11_theiler_contrast.py --score         # ~15 min on 12 cores
    python e11_theiler_contrast.py --control       # ~1 min, the same sweep on known curves
    python e11_theiler_contrast.py --pooled        # ~10 min on 12 cores
    python e11_theiler_contrast.py --verify        # ~3 min, single process
    python e11_theiler_contrast.py --report
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e11_theiler_contrast"
SERIES = OUT / "series"
sys.path.insert(0, str(HERE))

import mg as MG                                                          # noqa: E402
from e1_calibration import load_frozen                                   # noqa: E402
from e2_rank_sweep import make_spec                                      # noqa: E402
from dynamics import F_FAST, F_SLOW, equalise_gains, simulate            # noqa: E402
from runner import get_system                                            # noqa: E402
from system import frequencies                                           # noqa: E402
# mg.py puts ``grokking_analysis`` on the path, so this is the same machinery mg.py uses
from edm.dimension import _dither, resolve_theiler_window                # noqa: E402
from edm.embedding import autocorrelation_time, delay_embedding          # noqa: E402

#: label -> ``e2_rank_sweep`` mode.  "fast" and "slow" are the two recurrent drive rates of
#: sec:digits; "transient" is the decaying-transient arm the paper reports at ~29.
ARMS = (("fast", "qp"), ("slow", "qp_slow"), ("transient", "gd"))
R_VALUES = (2, 4, 6, 8)
SEEDS = (0, 1, 2)
#: four of the ten scored observers, one per family.  ``w_fro`` is the parameter norm a run
#: ordinarily logs; ``c_norm`` is a monotone function of it and returns the identical MG, so
#: it is not run twice.
OBSERVERS = ("w_fro", "c_proj1", "g_fro", "loss_probe")
#: the sweep.  ``"frozen"`` is the configuration's own ``theiler="autocorr"``, whose realised
#: value in samples is data-dependent and recorded per window.
THEILER_INTS = (0, 1, 2, 5, 10, 20, 50, 100, 150)
THEILER_GRID = THEILER_INTS + ("frozen",)
#: E2's simulation length, and its rule for keeping ~7 windows whatever the frozen window is
T, BURN = 30_000, 4_000
DITHER_SEED = 0          # ``runner.run_one`` calls ``MG.summarise(z, cfg)``, i.e. seed 0

# ---- the pooled-runs bonus test ------------------------------------------------------------
POOL_R = (1, 4)
POOL_M = (1, 2, 4, 8, 16)
POOL_T = 8_000           # one frozen window per run, with burn = 0: the transient's first 8000
POOL_OBS = "w_fro"
POOL_THEILER = 150       # the realised frozen exclusion on the transient arm


# ============================================================================== simulation
def stride_for(n, cfg):
    """E2's rule: keep ~7 windows whatever window length the calibration froze."""
    return max(500, (n - cfg.window) // 6)


def sim_job(arm, mode, seed, cfg):
    A, c_star = get_system(seed, 10)
    out = []
    for r in R_VALUES:
        spec = make_spec(mode, r, seed)
        mix, cond = equalise_gains(A, spec, c_star)
        logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)
        np.savez_compressed(
            SERIES / f"{arm}_r{r}_s{seed}.npz",
            traj_PR=info["traj_PR"], drive_cond=cond, f0=spec.f0,
            **{f"log__{o}": logs[o] for o in OBSERVERS})
        out.append((arm, r, seed, float(info["traj_PR"])))
    return out


def run_simulate(n_jobs):
    cfg, _ = load_frozen()
    SERIES.mkdir(parents=True, exist_ok=True)
    jobs = [(a, m, s, cfg) for a, m in ARMS for s in SEEDS]
    print(f"{len(jobs)} (arm, seed) jobs x {len(R_VALUES)} ranks", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(delayed(sim_job)(*j) for j in jobs)
    gt = pd.DataFrame([x for rr in res for x in rr],
                      columns=["arm", "r", "seed", "traj_PR"])
    gt.to_csv(OUT / "ground_truth_PR.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")
    print(gt.pivot_table(index="arm", columns="r", values="traj_PR").round(3).to_string())


# ============================================================================== the estimator
def window_query(w, cfg, dither_seed=DITHER_SEED, extra_theiler=()):
    """One KD-tree query per window, deep enough to serve every exclusion in the sweep.

    Returns ``(dist, dt, base, th_frozen)``.  ``dist`` and ``dt`` are the distances and the
    temporal separations of the ``k + 2 W_max + 1`` nearest candidates of every delay vector,
    in ascending distance -- exactly what ``edm._neighbor_distances`` queries at ``W_max``, so
    filtering them at any smaller ``W_T`` reproduces its neighbour set exactly.
    """
    w = np.asarray(w, float)
    xd = _dither(w, cfg.dither, np.random.default_rng(dither_seed))
    tau = MG.resolve_tau(cfg, w)
    th_frozen = int(min(resolve_theiler_window(cfg.theiler, xd, tau, cfg.max_E),
                        MG.THEILER_CAP))
    emb = delay_embedding(xd, cfg.max_E, tau)
    n, k = len(emb), cfg.k_neighbors
    w_max = max(THEILER_INTS + (th_frozen,) + tuple(extra_theiler))
    kq = min(n, k + 2 * w_max + 1)
    dist, idx = KDTree(emb).query(emb, k=kq)
    dt = np.abs(idx - np.arange(n)[:, None])

    c = emb - emb.mean(0, keepdims=True)
    s2 = np.linalg.svd(c, compute_uv=False) ** 2
    sd = float(w.std())
    base = dict(n_points=int(n), tau_used=int(tau), n_query=int(kq),
                PRdelay=float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else np.nan,
                acorr=float(autocorrelation_time(xd)),
                roughness=float(np.diff(w).std() / sd) if sd > 0 else np.nan)
    return dist, dt, base, th_frozen


def _mg_lb(d, k):
    """``mg.all_estimators``'s pooling, on an (n, k) block of Theiler-filtered distances."""
    d = np.maximum(d, MG.FLOOR_DIST)
    hit_floor = float((d <= MG.FLOOR_DIST * 1.000001).mean())
    S = np.sum(np.log(d[:, -1:] / d[:, :-1]), axis=1)
    hit_sum = float((S <= MG.FLOOR_SUM).mean())
    S = np.maximum(S, MG.FLOOR_SUM)
    n, total = len(S), float(S.sum())
    mg = lb = np.nan
    if np.isfinite(total) and total > 0:
        mg = (n * (k - 1) - 1) / total
        loc = (k - 1) / S
        lb = float(np.mean(loc[np.isfinite(loc)]))
    return dict(MG=mg, LB=lb, S_med=float(np.median(S)),
                degenerate=bool(hit_floor > 0.01 or hit_sum > 0.01),
                frac_floor=hit_floor, frac_sumfloor=hit_sum)


def filter_at(dist, dt, theiler, k):
    """The first ``k`` candidates with ``|i - j| > theiler``, in ascending distance."""
    valid = dt > theiler
    if valid.sum(1).min() < k:
        return None, None, float(valid.mean())
    order = np.argsort(~valid, axis=1, kind="stable")
    d = np.take_along_axis(dist, order, axis=1)[:, :k]
    t = np.take_along_axis(dt, order, axis=1)[:, :k]
    return d, t, float(valid.mean())


def window_cells(w, cfg, dither_seed=DITHER_SEED):
    """One record per exclusion in ``THEILER_GRID`` for a single window."""
    dist, dt, base, th_frozen = window_query(w, cfg, dither_seed)
    k = cfg.k_neighbors

    # the neighbour set the estimator would use with no exclusion: the reference the mechanism
    # columns are measured against
    d0, t0, _ = filter_at(dist, dt, 0, k)
    if d0 is None:
        return []
    d0f = np.maximum(d0, MG.FLOOR_DIST)
    d_ref = float(np.median(d0f[:, -1]))          # "as near as the temporal neighbours were"
    r1_w0 = float(np.median(d0f[:, 0]))
    rk_w0 = d_ref
    dt_w0 = float(np.median(t0))

    recs = []
    for label in THEILER_GRID:
        th = th_frozen if label == "frozen" else int(label)
        d, t, frac_valid = filter_at(dist, dt, th, k)
        rec = dict(theiler_label=str(label), theiler_used=int(th), **base,
                   r1_med_W0=r1_w0, rk_med_W0=rk_w0, dt_med_W0=dt_w0, d_ref=d_ref,
                   frac_query_valid=frac_valid)
        if d is None:
            recs.append({**rec, "MG": np.nan, "LB": np.nan, "degenerate": True})
            continue
        df = np.maximum(d, MG.FLOOR_DIST)
        rec.update(_mg_lb(d, k))
        rec.update(
            n_pairs=int(df.size),
            r1_med=float(np.median(df[:, 0])),
            rk_med=float(np.median(df[:, -1])),
            spread_med=float(np.median(df[:, -1] / df[:, 0])),
            dt_med=float(np.median(t)),
            # how much of the no-exclusion neighbour set survives, and how many of the
            # neighbours that DO survive are still as near as those were
            frac_kept_from_W0=float((t0 > th).mean()),
            n_pairs_near_ref=int((df <= d_ref).sum()),
            frac_near_ref=float((df <= d_ref).mean()),
            dist_inflation=float(np.median(df[:, 0]) / max(r1_w0, MG.FLOOR_DIST)))
        recs.append(rec)
    return recs


def score_job(path, cfg):
    d = np.load(path, allow_pickle=True)
    arm, rtag, stag = path.stem.split("_")
    r, seed = int(rtag[1:]), int(stag[1:])
    n_expect = None
    rows = []
    for obs in OBSERVERS:
        x = np.asarray(d[f"log__{obs}"], float)
        if not np.isfinite(x).all() or x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()                 # runner.run_one's normalisation
        n = len(z)
        c2 = dataclasses.replace(cfg, stride=stride_for(n, cfg))
        starts = MG.window_starts(n, c2)
        n_expect = len(starts)
        for wi, a in enumerate(starts):
            for rec in window_cells(z[a:a + c2.window], c2):
                rows.append(dict(arm=arm, r=r, seed=seed, observer=obs, window=wi,
                                 start=int(a), n_windows=n_expect,
                                 traj_PR=float(d["traj_PR"]), **rec))
    return rows


def run_score(n_jobs):
    cfg, _ = load_frozen()
    print(f"frozen config: {cfg}", flush=True)
    paths = sorted(SERIES.glob("*.npz"))
    if not paths:
        raise SystemExit("run --simulate first")
    print(f"{len(paths)} series x {len(OBSERVERS)} observers x "
          f"{len(THEILER_GRID)} exclusions", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(score_job)(p, cfg) for p in paths)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "sweep_windows.csv", index=False)
    print(f"\n{len(df)} window records in {time.time() - t0:.0f}s")

    agg = aggregate(df)
    agg.to_csv(OUT / "sweep_cells.csv", index=False)
    print(f"{len(agg)} cells -> {OUT / 'sweep_cells.csv'}")


def aggregate(df):
    """``mg.summarise``'s rule: median over the windows that are not flagged degenerate."""
    keys = ["arm", "r", "seed", "observer", "theiler_label"]
    num = ["MG", "LB", "S_med", "theiler_used", "n_points", "n_pairs", "n_query",
           "n_pairs_near_ref", "frac_near_ref", "frac_query_valid", "frac_kept_from_W0",
           "r1_med", "rk_med", "spread_med", "dt_med", "dist_inflation",
           "r1_med_W0", "rk_med_W0", "dt_med_W0", "PRdelay", "acorr", "roughness",
           "tau_used", "traj_PR"]
    out = []
    for key, g in df.groupby(keys, sort=False):
        ok = g[~g.degenerate.astype(bool)]
        src = ok if len(ok) else g
        rec = dict(zip(keys, key), n_windows=int(len(g)),
                   frac_degenerate=float(g.degenerate.astype(bool).mean()),
                   all_degenerate=bool(len(ok) == 0))
        for c in num:
            v = pd.to_numeric(src[c], errors="coerce").dropna()
            rec[c] = float(np.median(v)) if len(v) else np.nan
        rec["MG_sd"] = float(pd.to_numeric(src["MG"], errors="coerce").std())
        out.append(rec)
    return pd.DataFrame(out)


# ============================================================================== verification
def run_verify():
    """Prove the shared-query path equals ``mg.all_estimators`` instead of asserting it."""
    cfg, _ = load_frozen()
    rows = []
    for arm, _ in ARMS:
        p = SERIES / f"{arm}_r4_s0.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        x = np.asarray(d[f"log__{OBSERVERS[0]}"], float)
        z = (x - x.mean()) / x.std()
        c2 = dataclasses.replace(cfg, stride=stride_for(len(z), cfg))
        w = z[:c2.window]
        fast = {rec["theiler_label"]: rec for rec in window_cells(w, c2)}
        for label in THEILER_GRID:
            cc = dataclasses.replace(c2, theiler=cfg.theiler if label == "frozen"
                                     else int(label))
            can = MG.all_estimators(w, cc, DITHER_SEED)
            f = fast[str(label)]
            rows.append(dict(arm=arm, theiler_label=str(label),
                             theiler_canonical=can["theiler_used"],
                             theiler_fast=f["theiler_used"],
                             MG_canonical=can["MG"], MG_fast=f["MG"],
                             LB_canonical=can["LB"], LB_fast=f["LB"],
                             dMG=abs(can["MG"] - f["MG"]), dLB=abs(can["LB"] - f["LB"]),
                             deg_canonical=bool(can["degenerate"]),
                             deg_fast=bool(f["degenerate"])))
            print(f"  {arm:>10} {str(label):>6}  canonical MG {can['MG']:8.4f}  "
                  f"shared-tree MG {f['MG']:8.4f}  |diff| {rows[-1]['dMG']:.2e}", flush=True)
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "verify_against_canonical.csv", index=False)
    print(f"\nmax |MG - MG_canonical| = {v.dMG.max():.3e}")
    print(f"max |LB - LB_canonical| = {v.dLB.max():.3e}")
    print(f"realised exclusion identical: {bool((v.theiler_canonical == v.theiler_fast).all())}")


# ============================================================================== the cap
def run_uncapped():
    """What the frozen setting would do if ``mg.THEILER_CAP`` did not clip it.

    ``theiler="autocorr"`` resolves to ``max((max_E - 1) tau, acf_time)``, and the acf of a
    monotone transient does not decay: it resolves to ~1600 samples on the transient arm and
    is then clipped to 150 by the performance cap.  The published ~29 is therefore the value
    at the cap, not the value the configuration asks for.
    """
    cfg, _ = load_frozen()
    old = MG.THEILER_CAP
    MG.THEILER_CAP = 10 ** 9          # as ``e7b_theiler_quick.job`` lifts it
    rows = []
    try:
        for arm, _m in ARMS:
            for r in R_VALUES[:2]:
                p = SERIES / f"{arm}_r{r}_s0.npz"
                if not p.exists():
                    continue
                d = np.load(p, allow_pickle=True)
                x = np.asarray(d[f"log__{OBSERVERS[0]}"], float)
                z = (x - x.mean()) / x.std()
                w = z[:cfg.window]
                for cap in (150, 10 ** 9):
                    MG.THEILER_CAP = cap
                    rec = MG.all_estimators(w, cfg, DITHER_SEED)
                    rows.append(dict(arm=arm, r=r, cap=cap, theiler_used=rec["theiler_used"],
                                     MG=rec["MG"], LB=rec["LB"],
                                     degenerate=bool(rec["degenerate"])))
                    print(f"  {arm:>10} r={r} cap={cap:<10} realised "
                          f"{rec['theiler_used']:5d}  MG={rec['MG']:9.3f}", flush=True)
    finally:
        MG.THEILER_CAP = old
    pd.DataFrame(rows).to_csv(OUT / "cap_effect.csv", index=False)


# ============================================================================== analytic control
#: The same three regimes with no optimiser, no data and no simulation, where the intrinsic
#: dimension of the delay cloud is known by construction rather than measured.  A decaying
#: exponential traces a curve, so its estimand is exactly 1; a sum of two incommensurate
#: sinusoids fills a 2-torus, so its estimand is exactly 2.  ``fast`` and ``slow`` differ only
#: in samples per cycle, i.e. in whether the record is oversampled.
def control_series(name, seed, n=8_000):
    """``system.frequencies`` is used rather than hand-picked ratios: an exactly rational
    frequency makes the delay coordinates of that mode take finitely many values, and a
    "2-torus" built from 1/16 and a multiple of it measures 1.13, not 2 -- checked, and the
    reason this helper does not write its own frequencies."""
    t = np.arange(float(n))
    g = np.random.default_rng(seed)
    if name == "curve_decay":              # a transient: a 1-D curve, never revisited
        return np.exp(-t / 3000.0), 1.0
    if name == "curve_decay_osc":          # a transient that is not monotone: still a curve
        f = frequencies(1, F_FAST, 2.0)[0]
        return np.exp(-t / 3000.0) * np.sin(2 * np.pi * f * t + g.uniform(0, 6.28)), 1.0
    if name == "torus2_fast":              # 2-torus, ~16 samples per cycle: not oversampled
        f = frequencies(2, F_FAST, 2.0)
    elif name == "torus2_slow":            # 2-torus, ~400 samples per cycle: oversampled
        f = frequencies(2, F_SLOW, 2.0)
    else:
        raise ValueError(name)
    ph = g.uniform(0, 2 * np.pi, 2)
    return np.sin(2 * np.pi * np.outer(t, f) + ph).sum(1), 2.0


CONTROLS = ("curve_decay", "curve_decay_osc", "torus2_fast", "torus2_slow")


def run_control():
    cfg, _ = load_frozen()
    rows = []
    for name in CONTROLS:
        for seed in SEEDS:
            x, truth = control_series(name, seed)
            z = (x - x.mean()) / x.std()
            for rec in window_cells(z, cfg):
                rows.append(dict(series=name, seed=seed, truth=truth, **rec))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "analytic_control.csv", index=False)
    p = (df.pivot_table(index="theiler_label", columns="series", values="MG")
         .reindex([str(x) for x in THEILER_GRID]))
    print("\nMG on analytic series of known dimension (median over 3 phase seeds)")
    print("truth: curve_decay 1, curve_decay_osc 1, torus2_fast 2, torus2_slow 2")
    print(p.round(3).to_string())
    print("\nrealised exclusion at theiler='autocorr':")
    print(df[df.theiler_label == "frozen"].groupby("series").theiler_used.median().to_string())


# ============================================================================== pooled runs
def pool_sim_job(r, m, cfg):
    """One independent transient: the same system and the same r-dimensional subspace of
    initial displacements, a different draw within it.

    ``noise_amp`` is inert in mode ``gd`` (``dynamics.simulate`` gates the injection on
    ``mode in ("noise", "mixed", "batch_proj")``) but does enter the run's rng key, so this
    reseeds the initial displacement and changes nothing else -- in particular ``_drive_setup``
    depends only on ``spec.seed``, so ``Ud`` and hence the subspace is common to all runs.
    """
    A, c_star = get_system(0, 10)
    spec = make_spec("gd", r, 0, T=POOL_T, noise_amp=1e-4 * (m + 1))
    mix, _ = equalise_gains(A, spec, c_star)
    logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)
    return r, m, np.asarray(logs[POOL_OBS], float), float(info["traj_PR"])


def pooled_cloud(series, cfg):
    """Delay vectors of every run in one cloud, on a common scale, tagged by run and time."""
    allx = np.concatenate(series)
    mu, sd = allx.mean(), allx.std()
    E, tau = cfg.max_E, int(cfg.tau)
    blocks, run_id, t_id = [], [], []
    for m, x in enumerate(series):
        xd = _dither((x - mu) / sd, cfg.dither, np.random.default_rng(1000 + m))
        e = delay_embedding(xd, E, tau)
        blocks.append(e)
        run_id.append(np.full(len(e), m))
        t_id.append(np.arange(len(e)))
    return (np.vstack(blocks), np.concatenate(run_id), np.concatenate(t_id))


RULES = ("none", "within", "crossonly")


def pooled_estimates(emb, run_id, t_id, k, theiler=POOL_THEILER, chunk=8_000):
    """Three neighbour rules on one cloud, from one query, in row chunks so that a
    16-run cloud does not need a gigabyte of candidate indices.

    ``none``      no exclusion at all -- the reviewer's route to local density.
    ``within``    the protocol's Theiler exclusion applied *within* a run, cross-run
                  neighbours always allowed.  The paper's estimator, handed the density it
                  is assumed not to have.
    ``crossonly`` every neighbour must come from a different run: no time structure at all.
    """
    n = len(emb)
    kq = min(n, k + 2 * theiler + 1)
    tree = KDTree(emb)
    keep = {r: [] for r in RULES}
    same_frac = {r: [] for r in RULES}
    n_rows = {r: 0 for r in RULES}
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dist, idx = tree.query(emb[a:b], k=kq)
        same = run_id[idx] == run_id[a:b, None]
        dts = np.abs(t_id[idx] - t_id[a:b, None])
        rules = dict(none=~(same & (dts == 0)),
                     within=(~same) | (dts > theiler),
                     crossonly=~same)
        for name, valid in rules.items():
            enough = valid.sum(1) >= k
            n_rows[name] += int(enough.sum())
            if not enough.any():
                continue
            v, dd, ss = valid[enough], dist[enough], same[enough]
            order = np.argsort(~v, axis=1, kind="stable")
            keep[name].append(np.take_along_axis(dd, order, axis=1)[:, :k])
            same_frac[name].append(np.take_along_axis(ss, order, axis=1)[:, :k])
    out = {}
    for name in RULES:
        frac_rows = n_rows[name] / n
        if n_rows[name] < 50:
            out[name] = dict(MG=np.nan, LB=np.nan, rows_used=float(frac_rows))
            continue
        d = np.vstack(keep[name])
        s = np.vstack(same_frac[name])
        rec = _mg_lb(d, k)
        d = np.maximum(d, MG.FLOOR_DIST)
        rec.update(rows_used=float(frac_rows), frac_same_run=float(s.mean()),
                   r1_med=float(np.median(d[:, 0])), rk_med=float(np.median(d[:, -1])),
                   spread_med=float(np.median(d[:, -1] / d[:, 0])))
        out[name] = rec
    return out


def run_pooled(n_jobs):
    cfg, _ = load_frozen()
    jobs = [(r, m, cfg) for r in POOL_R for m in range(max(POOL_M))]
    print(f"{len(jobs)} independent transients (T={POOL_T})", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(pool_sim_job)(*j) for j in jobs)
    by_r = {r: [None] * max(POOL_M) for r in POOL_R}
    pr = {}
    for r, m, x, p in res:
        by_r[r][m] = x
        pr[r] = p
    print(f"simulated in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for r in POOL_R:
        for M in POOL_M:
            emb, run_id, t_id = pooled_cloud(by_r[r][:M], cfg)
            est = pooled_estimates(emb, run_id, t_id, cfg.k_neighbors)
            for rule, rec in est.items():
                rows.append(dict(r=r, M=M, rule=rule, n_points=len(emb),
                                 traj_PR=pr[r], theiler=POOL_THEILER, **rec))
                print(f"  r={r} M={M:2d} {rule:>9}  MG={rec.get('MG', np.nan):8.3f}  "
                      f"LB={rec.get('LB', np.nan):8.3f}  "
                      f"same-run frac={rec.get('frac_same_run', np.nan):.3f}  "
                      f"rows={rec.get('rows_used', np.nan):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pooled_runs.csv", index=False)
    print(f"\n-> {OUT / 'pooled_runs.csv'}  ({time.time() - t0:.0f}s)")


# ============================================================================== report
def published():
    """The E2 numbers this experiment must reproduce on its two diagonal cells."""
    p = HERE / "results" / "e2_rank_sweep" / "sweep_raw.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.arm.isin(["qp", "qp_slow", "gd"]) & d.r.isin(R_VALUES)
          & d.observer.isin(OBSERVERS) & d.seed.isin(SEEDS)]
    d["arm"] = d.arm.map({"qp": "fast", "qp_slow": "slow", "gd": "transient"})
    return (d.groupby(["arm", "r"]).agg(MG_published=("MG", "median"),
                                        LB_published=("LB", "median"),
                                        truth=("traj_PR", "median")).reset_index())


def run_report():
    cells = pd.read_csv(OUT / "sweep_cells.csv")
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    med = (cells.groupby(["arm", "theiler_label", "r"])
           .agg(MG=("MG", "median"), LB=("LB", "median"),
                theiler_used=("theiler_used", "median"),
                truth=("traj_PR", "median"), S_med=("S_med", "median"),
                spread=("spread_med", "median"), r1=("r1_med", "median"),
                r1_W0=("r1_med_W0", "median"), near_ref=("frac_near_ref", "median"),
                kept=("frac_kept_from_W0", "median"), dt_W0=("dt_med_W0", "median"),
                deg=("frac_degenerate", "mean"), PRdelay=("PRdelay", "median"),
                n=("MG", "size")).reset_index())
    med.to_csv(OUT / "summary_by_arm_theiler_r.csv", index=False)

    say("=" * 96)
    say("THE TWO-BY-TWO  (median over 4 observers x 3 seeds; truth = measured traj_PR)")
    say("=" * 96)
    pub = published()
    for arm in ("fast", "slow", "transient"):
        for label in ("0", "frozen"):
            b = med[(med.arm == arm) & (med.theiler_label == label)].sort_values("r")
            if not len(b):
                continue
            txt = "  ".join(f"r={int(x.r)}: {x.MG:6.2f}" for x in b.itertuples())
            say(f"{arm:>10}  W_T={label:>6} (realised {b.theiler_used.min():.0f}"
                f"-{b.theiler_used.max():.0f})   {txt}")
    say("")
    say("truth (measured trajectory participation ratio):")
    for arm in ("fast", "slow", "transient"):
        b = med[(med.arm == arm) & (med.theiler_label == "0")].sort_values("r")
        say(f"{arm:>10}  " + "  ".join(f"r={int(x.r)}: {x.truth:5.2f}" for x in b.itertuples()))

    if pub is not None:
        say("")
        say("=" * 96)
        say("CORRECTNESS CHECK -- do the two diagonal cells reproduce E2's published values?")
        say("=" * 96)
        chk = (med[med.theiler_label == "frozen"][["arm", "r", "MG", "LB"]]
               .merge(pub, on=["arm", "r"]))
        chk["dMG"] = chk.MG - chk.MG_published
        chk["rel"] = chk.dMG / chk.MG_published
        say(chk.round(3).to_string(index=False))
        say(f"max |relative difference| = {chk.rel.abs().max():.4f}")
        chk.to_csv(OUT / "diagonal_check.csv", index=False)

    say("")
    say("=" * 96)
    say("THE W_T SWEEP  (MG, median over observers, seeds and r within each arm)")
    say("=" * 96)
    sw = med.pivot_table(index="theiler_label", columns="arm", values="MG")
    order = [str(x) for x in THEILER_GRID]
    say(sw.reindex(order).round(3).to_string())
    say("")
    say("per rank:")
    say(med.pivot_table(index="theiler_label", columns=["arm", "r"], values="MG")
        .reindex(order).round(2).to_string())

    say("")
    say("=" * 96)
    say("THE MECHANISM  (median over observers, seeds and r)")
    say("=" * 96)
    mech = (med.groupby(["arm", "theiler_label"])
            .agg(MG=("MG", "median"), S_med=("S_med", "median"),
                 spread_rk_r1=("spread", "median"), r1=("r1", "median"),
                 r1_at_W0=("r1_W0", "median"), frac_near_ref=("near_ref", "median"),
                 frac_kept=("kept", "median"), dt_med_W0=("dt_W0", "median"),
                 deg=("deg", "mean")).reset_index())
    mech["r1_inflation"] = mech.r1 / mech.r1_at_W0
    mech.to_csv(OUT / "mechanism.csv", index=False)
    for arm in ("fast", "slow", "transient"):
        say(f"\n--- {arm} ---")
        b = mech[mech.arm == arm].set_index("theiler_label").reindex(order)
        say(b[["MG", "S_med", "spread_rk_r1", "r1_inflation", "frac_near_ref",
               "frac_kept", "dt_med_W0", "deg"]].round(4).to_string())

    p = OUT / "cap_effect.csv"
    if p.exists():
        say("")
        say("=" * 96)
        say("THE CAP -- the frozen setting with and without mg.THEILER_CAP = 150")
        say("=" * 96)
        say(pd.read_csv(p).round(3).to_string(index=False))

    p = OUT / "analytic_control.csv"
    if p.exists():
        say("")
        say("=" * 96)
        say("ANALYTIC CONTROL -- the same sweep on series whose dimension is known exactly")
        say("(curve_decay 1, curve_decay_osc 1, torus2_fast 2, torus2_slow 2)")
        say("=" * 96)
        ct = pd.read_csv(p)
        say(ct.pivot_table(index="theiler_label", columns="series", values="MG")
            .reindex(order).round(3).to_string())

    p = OUT / "pooled_runs.csv"
    if p.exists():
        say("")
        say("=" * 96)
        say("POOLED INDEPENDENT TRANSIENTS")
        say("=" * 96)
        pl = pd.read_csv(p)
        say(pl.pivot_table(index=["r", "M"], columns="rule",
                           values=["MG", "frac_same_run"]).round(3).to_string())

    (OUT / "report_tables.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true", help="run the trajectories -> NPZ")
    ap.add_argument("--score", action="store_true", help="the Theiler sweep -> CSV")
    ap.add_argument("--verify", action="store_true",
                    help="shared-tree path vs mg.all_estimators")
    ap.add_argument("--pooled", action="store_true", help="the pooled-runs bonus test")
    ap.add_argument("--control", action="store_true",
                    help="the same sweep on analytic series of known dimension")
    ap.add_argument("--uncapped", action="store_true",
                    help="what the frozen setting does without mg.THEILER_CAP")
    ap.add_argument("--report", action="store_true", help="tables from the CSVs")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    ran = False
    for flag, fn in (("simulate", lambda: run_simulate(a.jobs)),
                     ("score", lambda: run_score(a.jobs)),
                     ("verify", run_verify),
                     ("uncapped", run_uncapped),
                     ("control", run_control),
                     ("pooled", lambda: run_pooled(a.jobs)),
                     ("report", run_report)):
        if getattr(a, flag):
            fn()
            ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()

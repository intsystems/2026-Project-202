"""E10 -- what sets the ceiling: the embedding dimension, or the record length?

The paper recovers the constructed active dimension up to about eight directions and then
saturates.  Section 5.2 leaves two explanations open and they are not distinguishable from
any number already published, because at the frozen configuration the two coincide:

  H1  TAKENS.  The embedding condition is E > 2d, and the frozen configuration uses
      E_max = 20, so d = 10 is the largest recoverable dimension by construction (and the
      twenty-direction configuration, E_max = 40 at r = 20, sits at E = 2d exactly, one
      short of the condition).  Prediction: the ceiling scales with E_max / 2 and is
      insensitive to the record length.

  H2  ECKMANN-RUELLE.  A finite record bounds the measurable dimension, D <~ 2 log10 N
      (Eckmann & Ruelle 1992, Physica D 56:185).  The frozen window is N = 8000, giving
      7.8 -- also about eight.  Prediction: the ceiling scales with 2 log10 N and is
      insensitive to E_max.

Each hypothesis is insensitive to the other's knob, so two one-dimensional sweeps separate
them.  Sweep E: E_max in {10, 14, 20, 28, 40, 56} at N = 8000.  Sweep N: N in {1000, ...,
64000} at E_max = 20.  Everything else is the eight-direction frozen configuration of
appendix C (tau = 4, m = 20, the autocorrelation Theiler rule, dither 1e-9), and no clamping
is applied anywhere -- the published numbers are unclamped, which is why the report contains
estimates above E_max.

Three things this file is careful about.

*The record is lengthened, not resampled.*  One trajectory of ``BURN + 64000`` steps is
simulated per (r, seed) and every N in the sweep is a **prefix** of it.  The sampling rate,
the drive frequencies and the delay lag are therefore identical at every N: a longer record
is a longer record.  Resampling a fixed time span to more points would change the lag in
periods and confound the sweep with the tau-sensitivity of section 6.2.  Because ``qp`` mode
draws no per-step randomness, the prefix is bit-identical to the shorter run, which
``--check`` verifies against the trajectories already in ``results/k20_calibration``.

*E_max cannot be moved on its own.*  Raising it at fixed tau also raises the delay span
``(E-1) tau`` and, through the autocorrelation rule, the Theiler exclusion (36 samples at
E_max = 10, 150 at E_max = 40 and above, where ``mg.THEILER_CAP`` binds).  Sweep E is
therefore run in three arms -- ``frozen`` (the rule as published), ``theiler76`` (exclusion
held at its frozen value), ``fixedspan`` (span held at its frozen value, so tau falls) -- and
the realised exclusion is recorded on every row.

*A null result for one knob is only informative where the other is not binding.*  At the
frozen E_max = 20 the Takens bound is 10 and the Eckmann-Ruelle bound at the frozen window is
7.8, so a flat record-length sweep there could be masked by the embedding.  The record sweep
is therefore repeated at E_max = 56, where the embedding is nowhere near limiting over this
rank grid (``--n-sweep-E 56``).

*The estimator is ``mg.all_estimators``*, not ``dimension_recovery/estimators.py``, which has
different Theiler and clamping defaults.

    python e10_ceiling_sweep.py --check                     # reproduce the published ceiling
    python e10_ceiling_sweep.py --simulate                  # ~10 min on 8 cores
    python e10_ceiling_sweep.py --score --resume            # ~3 h on 8 cores
    python e10_ceiling_sweep.py --score --arms "" \
        --n-sweep-E 56 --n-grid 1000,2000,4000,8000,16000 \
        --out ceiling_raw_e56.csv                           # the control arm, ~15 min
    python e10_ceiling_sweep.py --analyse --figure          # seconds
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")           # one BLAS thread per worker, not per core

import argparse                                                          # noqa: E402
import dataclasses                                                       # noqa: E402
import json                                                              # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402
from pathlib import Path                                                 # noqa: E402

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from joblib import Parallel, delayed                                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mg as MG                                                          # noqa: E402

OUT = HERE / "results" / "e10_ceiling"
TRAJ = OUT / "trajectories"
K20 = HERE / "results" / "k20_calibration" / "trajectories"
OUT.mkdir(parents=True, exist_ok=True)
TRAJ.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ the grid
R_VALUES = (2, 4, 6, 8, 10, 12, 14, 16, 20)
SEEDS = (0, 1, 2)
E_GRID = (10, 14, 20, 28, 40, 56)
N_GRID = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
N_REF = 8000                       # the frozen window; sweep E is run here
E_REF = 20                         # the frozen E_max; sweep N is run here
BURN = 2000
T_STEPS = BURN + max(N_GRID)       # 66 000: every N is a prefix of one trajectory

#: The eight-direction frozen configuration of appendix C, table 2.  ``window``/``stride``
#: are set per cell (window = N, one window per record) and ``max_E`` is the swept knob.
FROZEN = MG.MGConfig(max_E=E_REF, tau=4, k_neighbors=20, theiler="autocorr",
                     window=N_REF, stride=N_REF, dither=1e-9)

#: The published twenty-direction configuration, used only by ``--check``.
PUBLISHED_K20 = MG.MGConfig(max_E=40, tau=16, k_neighbors=20, theiler="autocorr",
                            window=8000, stride=4000, dither=1e-9)

#: Four observers, one per family (norm / gradient / projection / function).  ``w_fro`` is
#: dropped because it is a monotone function of ``c_norm`` -- the two return bit-identical MG
#: at every rank in ``results/k20_calibration/frozen_per_r.csv`` -- and the two loss observers
#: because ``loss_step`` fails the silence control of section 5.3.  All eight of the published
#: panel are stored in the NPZ so ``--check`` can reproduce the published median.
PANEL = ("c_norm", "g_fro", "c_proj1", "fn_proj1")
STORED = ("w_fro", "c_norm", "g_fro", "g_proj", "c_proj1", "fn_fro", "fn_proj1", "loss_full")

#: rho_ident needs a second embedding at 2 E_max.  A 321-neighbour query in 80 dimensions on
#: 64 000 points costs more than the entire rest of the sweep, so it is computed on seed 0
#: only, and at the two longest records on two ranks only -- the same subsetting
#: ``e2_rank_sweep.py`` applies for the same reason.
IDENT_SEED = 0
IDENT_BIG_R = (20,)
IDENT_BIG_N = (32000, 64000)


def traj_path(r, seed):
    return TRAJ / f"qp_r{r:02d}_s{seed:02d}.npz"


# ------------------------------------------------------------------ simulation
def simulate_one(r, seed, force=False):
    """One long ``qp`` trajectory of the k = 20 digits adapter, and its per-N truth."""
    path = traj_path(r, seed)
    if path.exists() and not force:
        return path, "cached"
    from calibration_k20 import build_system, make_spec                  # noqa: E402
    from dynamics import equalise_gains, simulate                        # noqa: E402
    from system import rank_pr                                           # noqa: E402

    A = build_system(seed, fast=True)
    c_star = A.solve()
    spec = dataclasses.replace(make_spec("qp", r, seed, fast=True), T=T_STEPS, burn=BURN)
    mix, cond = equalise_gains(A, spec, c_star)
    logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)

    # the construction's own ground-truth check, measured on exactly the prefix each cell
    # scores rather than on the whole record
    truth = {}
    for n in N_GRID:
        rk, pr = rank_pr(C[:n])
        truth[str(n)] = [int(rk), float(pr)]

    arrays = {f"log__{o}": np.asarray(logs[o], float) for o in STORED}
    arrays["truth_json"] = np.asarray(json.dumps(truth))
    arrays["info_json"] = np.asarray(json.dumps({**info, "drive_cond": cond}))
    arrays["spec_json"] = np.asarray(json.dumps(
        {**dataclasses.asdict(spec), "mode": "qp", "tag": "qp"}, default=str))
    np.savez_compressed(path, **arrays)
    return path, "simulated"


def run_simulations(n_jobs, force=False):
    jobs = [(r, s) for r in R_VALUES for s in SEEDS]
    print(f"simulating {len(jobs)} trajectories of {T_STEPS} steps "
          f"({BURN} burn-in, {max(N_GRID)} kept)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(simulate_one)(r, s, force) for r, s in jobs)
    print(f"{sum(1 for _, w in res if w == 'simulated')} simulated, "
          f"{sum(1 for _, w in res if w == 'cached')} cached, "
          f"{time.time() - t0:.0f}s", flush=True)


_CACHE = {}


def load_traj(r, seed):
    """One-entry cache: jobs are dispatched in (r, seed) order, so a worker reloads once."""
    if (r, seed) in _CACHE:
        return _CACHE[(r, seed)]
    z = np.load(traj_path(r, seed), allow_pickle=False)
    val = ({k[5:]: z[k] for k in z.files if k.startswith("log__")},
           json.loads(str(z["truth_json"])), json.loads(str(z["info_json"])))
    _CACHE.clear()
    _CACHE[(r, seed)] = val
    return val


# ------------------------------------------------------------------ diagnostics
def trend_crossings(x):
    """Sign changes of the window's residual about its least-squares line.

    The paper's non-monotonicity diagnostic (section 4.3); identical to ``oscillations`` in
    ``e5_real_logs.py``.  Near zero on a transient, large on a recurrent orbit.
    """
    t = np.arange(len(x), dtype=float)
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    return int(np.count_nonzero(np.diff(np.signbit(d))))


# ------------------------------------------------------------------ scoring
def cell(sweep, arm, E, N, r, seed, obs, ident, theiler, tau):
    """One (E_max, N, r, seed, observer) cell: one window, optionally a second at 2 E_max."""
    MG.THEILER_CAP = 150                      # the frozen implementation's cap, per worker
    logs, truth, info = load_traj(r, seed)
    tr_rank, tr_pr = truth[str(N)]
    cfg = dataclasses.replace(FROZEN, max_E=E, window=N, stride=N, theiler=theiler, tau=tau)
    x = np.asarray(logs[obs], float)[:N]
    if not np.isfinite(x).all() or x.std() <= 1e-12:
        return None
    z = (x - x.mean()) / x.std()
    rec = MG.all_estimators(z, cfg, seed=seed)
    row = dict(sweep=sweep, arm=arm, max_E=E, N=N, r=r, seed=seed, observer=obs, tau=tau,
               truth_r=r, traj_rank=tr_rank, traj_pr=tr_pr,
               MG=rec["MG"], LB=rec["LB"], TwoNN=rec["TwoNN"],
               PRdelay=rec["PRdelay"], specPR256=rec["specPR256"],
               roughness=rec["roughness"], acorr=rec["acorr"],
               degenerate=bool(rec["degenerate"]),
               frac_floor=rec.get("frac_floor", np.nan),
               frac_sumfloor=rec.get("frac_sumfloor", np.nan),
               tau_used=rec.get("tau_used", np.nan),
               theiler_used=rec.get("theiler_used", np.nan),
               theiler_requested=((E - 1) * int(tau) if theiler == "autocorr"
                                  else int(theiler)),
               embed_span=(E - 1) * int(tau),
               n_delay_vectors=max(0, N - (E - 1) * int(tau)),
               crossings=trend_crossings(z),
               margin_res=info.get("margin_res"), drive_cond=info.get("drive_cond"),
               MG_2E=np.nan, rho_ident=np.nan, theiler_used_2E=np.nan)
    if ident:
        r2 = MG.all_estimators(z, dataclasses.replace(cfg, max_E=2 * E), seed=seed)
        row["MG_2E"] = r2["MG"]
        row["theiler_used_2E"] = r2.get("theiler_used", np.nan)
        row["rho_ident"] = (r2["MG"] / rec["MG"]) if rec["MG"] else np.nan
    return row


#: The three arms of sweep E.  ``E_max`` cannot be raised at fixed ``tau`` without also
#: raising the delay span ``(E-1) tau`` and, through the autocorrelation rule, the Theiler
#: exclusion; each arm holds one of those two fixed so the three together say which is doing
#: the work.  Each has a defect and they are stated rather than hidden: ``theiler76`` uses an
#: exclusion smaller than its own embedding span above E_max = 20, which is the flaw appendix
#: F criticises in the twenty-direction arm, and ``fixedspan`` changes tau, which section 6.2
#: shows the estimate is sensitive to.
SPAN_REF = (E_REF - 1) * 4                                     # 76 samples, the frozen span


def arm_params(arm, E):
    """(theiler, tau) for one arm at one E_max."""
    if arm == "frozen":                       # the frozen rule: exclusion grows with E_max
        return "autocorr", 4
    if arm == "theiler76":                    # exclusion held at its frozen value
        return SPAN_REF, 4
    if arm == "fixedspan":                    # delay span held at its frozen value
        return "autocorr", max(1, int(round(SPAN_REF / (E - 1))))
    raise ValueError(arm)


def build_jobs(arms, n_sweep_E=E_REF, n_grid=N_GRID):
    """(sweep, arm, E, N, r, seed, observer, ident, theiler, tau), in trajectory order.

    ``n_sweep_E`` repeats the record-length sweep at a different embedding dimension.  The
    sweep as specified runs at the frozen E_max = 20, where the Takens bound is 10 and the
    Eckmann-Ruelle bound at the frozen window is 7.8; if the first binds, a null result for
    the second is uninformative, because the record-length effect would be masked.  Re-running
    the same record lengths at E_max = 56, where the embedding is nowhere near limiting over
    this rank grid, removes that objection.
    """
    jobs = []
    for arm in [a for a in arms if a]:
        for E in E_GRID:
            th, tau = arm_params(arm, E)
            for r in R_VALUES:
                for s in SEEDS:
                    for o in PANEL:
                        jobs.append(("E", arm, E, N_REF, r, s, o,
                                     arm == "frozen" and s == IDENT_SEED, th, tau))
    label = "N" if n_sweep_E == E_REF else f"N_E{n_sweep_E}"
    for N in n_grid:
        for r in R_VALUES:
            for s in SEEDS:
                big = N in IDENT_BIG_N
                for o in PANEL:
                    jobs.append((label, "frozen", n_sweep_E, N, r, s, o,
                                 label == "N" and s == IDENT_SEED
                                 and (not big or r in IDENT_BIG_R),
                                 "autocorr", 4))
    # group by trajectory so the one-entry cache in load_traj actually hits, and take the
    # short records first: the cost per cell grows faster than linearly in N, so this order
    # leaves a usable sweep on disk at every checkpoint instead of only at the end
    jobs.sort(key=lambda j: (j[3], j[4], j[5], -j[2]))
    return jobs


JOB_KEY = ("sweep", "arm", "max_E", "N", "r", "seed", "observer")


def chunked(jobs, heavy_from):
    """Group the sorted job list into batches small enough to checkpoint between.

    The longest records cost minutes per cell, so the sweep is dispatched in pieces and the
    partial CSV is rewritten after each: a long run should not be all-or-nothing.
    """
    by_n = {}
    for j in jobs:
        by_n.setdefault(j[3], []).append(j)
    for N in sorted(by_n):
        g = by_n[N]
        size = 12 if N >= heavy_from else (60 if N >= N_REF else 300)
        for i in range(0, len(g), size):
            yield N, g[i:i + size]


def score(n_jobs, heavy_jobs, arms, heavy_from=32000, out_name="ceiling_raw.csv",
          resume=False, n_sweep_E=E_REF, n_grid=N_GRID):
    missing = [(r, s) for r in R_VALUES for s in SEEDS if not traj_path(r, s).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} trajectories missing; run --simulate first")
    jobs = build_jobs(arms, n_sweep_E, n_grid)
    out = []
    if resume and (OUT / out_name).exists():
        prev = pd.read_csv(OUT / out_name)
        out = prev.to_dict("records")
        have = {tuple(t) for t in prev[list(JOB_KEY)].itertuples(index=False, name=None)}
        n0 = len(jobs)
        jobs = [j for j in jobs
                if (j[0], j[1], j[2], j[3], j[4], j[5], j[6]) not in have]
        print(f"resuming: {n0 - len(jobs)} cells already on disk", flush=True)
    batches = list(chunked(jobs, heavy_from))
    print(f"{len(jobs)} cells in {len(batches)} batches, shortest records first", flush=True)
    t0 = time.time()
    done = 0
    for i, (N, batch) in enumerate(batches, 1):
        # the 2 E_max embeddings on the longest records are the only memory-bound cells
        nj = heavy_jobs if (N >= heavy_from and any(j[7] for j in batch)) else n_jobs
        res = Parallel(n_jobs=nj, verbose=0, batch_size=1)(delayed(cell)(*j) for j in batch)
        out += [row for row in res if row is not None]
        done += len(batch)
        pd.DataFrame(out).to_csv(OUT / out_name, index=False)
        el = time.time() - t0
        print(f"[{i}/{len(batches)}] N={N:6d} {len(batch):3d} cells at {nj} workers "
              f"| {done}/{len(jobs)} done | {el/60:.1f} min elapsed, "
              f"~{el/done*(len(jobs)-done)/60:.0f} min left", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(OUT / out_name, index=False)
    print(f"{len(df)} rows -> {OUT / out_name}  ({time.time() - t0:.0f}s)", flush=True)
    return df


# ------------------------------------------------------------------ reproduction check
def check(n_jobs):
    """Reproduce a published ceiling number before sweeping anything.

    Two checks.  (a) The twenty-direction row of table 3: the frozen E_max = 40 / tau = 16
    configuration on the k = 20 trajectories already in ``results/k20_calibration`` must give
    a median MG of 15.1 at r = 20 against a measured effective rank of 19.99.  (b) The long
    trajectories written by ``--simulate`` must agree bit-for-bit with those trajectories over
    their common prefix, which is what licenses treating every N as a prefix of one record.
    """
    lines = []
    paths = [(r, s, K20 / f"qp_r{r:02d}_s{s:02d}.npz") for r in R_VALUES for s in SEEDS]
    have = [(r, s, p) for r, s, p in paths if p.exists()]
    if not have:
        raise RuntimeError(f"no published trajectories under {K20}")

    def one(r, s, p):
        z = np.load(p, allow_pickle=False)
        info = json.loads(str(z["info_json"]))
        rows = []
        for obs in STORED:
            x = np.asarray(z[f"log__{obs}"], float)
            # standardise on the whole log and take the first window, exactly as
            # ``calibration_k20.score`` -> ``mg.summarise`` does: the 10 000-sample record
            # and a stride of 4 000 leave one window at offset 0
            zz = ((x - x.mean()) / x.std())[:PUBLISHED_K20.window]
            rec = MG.all_estimators(zz, PUBLISHED_K20, seed=s)
            rows.append(dict(r=r, seed=s, observer=obs, MG=rec["MG"],
                             traj_pr=info["traj_PR"],
                             degenerate=bool(rec["degenerate"])))
        return rows

    print(f"(a) rescoring {len(have)} published trajectories at {PUBLISHED_K20}", flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(one)(r, s, p) for r, s, p in have)
    rep = pd.DataFrame([x for rr in res for x in rr])
    rep.to_csv(OUT / "check_published.csv", index=False)
    med = rep.groupby("r").agg(MG=("MG", "median"), traj_pr=("traj_pr", "median"))
    lines.append("(a) twenty-direction configuration on the published trajectories, "
                 "median over 8 observers x 3 seeds")
    lines.append(med.round(2).to_string())
    if 20 in med.index:
        lines.append(f"    r=20: MG = {med.MG.loc[20]:.2f} (paper: 15.1), "
                     f"measured effective rank = {med.traj_pr.loc[20]:.2f} (paper: 19.99)")

    # (b) prefix identity
    lines.append("")
    lines.append("(b) long trajectories vs the published ones over their common prefix")
    n_ok = n_bad = 0
    for r, s, p in have:
        if not traj_path(r, s).exists():
            continue
        z = np.load(p, allow_pickle=False)
        logs, _, _ = load_traj(r, s)
        n = min(len(z["log__c_norm"]), len(logs["c_norm"]))
        worst = max(float(np.max(np.abs(np.asarray(z[f"log__{o}"], float)[:n]
                                        - logs[o][:n]))) for o in STORED)
        if worst == 0.0:
            n_ok += 1
        else:
            n_bad += 1
            lines.append(f"    r={r} s={s}: max |difference| = {worst:.3e} over {n} samples")
    lines.append(f"    {n_ok} trajectories bit-identical, {n_bad} not")
    txt = "\n".join(lines)
    (OUT / "check.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)


# ------------------------------------------------------------------ analysis
def crossing_point(rs, vals, tol):
    """First r at which ``r - MG(r)`` exceeds ``tol``, linearly interpolated in r.

    ``inf`` when the estimate never falls that far behind the truth on the grid, which is a
    ceiling above the top of the grid rather than a missing value.
    """
    rs = np.asarray(rs, float)
    g = rs - np.asarray(vals, float)
    ok = np.isfinite(g)
    rs, g = rs[ok], g[ok]
    if len(rs) < 2:
        return np.nan
    for i in range(len(rs)):
        if g[i] > tol:
            if i == 0:
                return float(rs[0])
            x0, x1, y0, y1 = rs[i - 1], rs[i], g[i - 1], g[i]
            return float(x0 + (tol - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    return np.inf


def ceilings(block, tols=(0.5, 1.0, 2.0)):
    """Both operational ceilings for one (arm, E_max, N) block of the grid.

    Cells flagged degenerate are dropped before the median, as everywhere else in this
    package: a window whose neighbour distances hit the 1e-8 floor returns a number, and it
    is not an estimate.  ``n_r_dropped`` counts ranks lost that way, because a ceiling
    computed on a curve with holes in it is not the same measurement.
    """
    good = block[~block.degenerate.astype(bool)]
    med = (good.groupby("r").agg(MG=("MG", "median"), truth=("traj_pr", "median"),
                                 PRdelay=("PRdelay", "median")).sort_index())
    out = {f"r_track_{t:g}": crossing_point(med.index.values, med.MG.values, t)
           for t in tols}
    top = med.loc[med.index >= 16, "MG"]
    out["MG_plateau"] = float(top.median()) if len(top) else np.nan
    out["MG_at_20"] = float(med.MG.loc[20]) if 20 in med.index else np.nan
    out["MG_at_8"] = float(med.MG.loc[8]) if 8 in med.index else np.nan
    # the same two ceilings for the linear null.  PRdelay is the participation ratio of the
    # delay covariance: it knows nothing about manifolds, neighbours or Takens, and its
    # only hard limit is E itself.  If it ceilings where MG does, the ceiling is a statement
    # about how many components the delay window resolves, not about geometry.
    out["PR_track_1"] = crossing_point(med.index.values, med.PRdelay.values, 1.0)
    topl = med.loc[med.index >= 16, "PRdelay"]
    out["PR_plateau"] = float(topl.median()) if len(topl) else np.nan
    # the third and least arbitrary measure: how fast the estimate still responds to r over
    # the top of the grid.  A true ceiling is a slope of zero; a slope of one is recovery.
    # It needs no threshold and no extrapolation beyond the grid.
    hi = med.loc[med.index >= 8]
    for name, col in (("slope_top", "MG"), ("slope_top_PR", "PRdelay")):
        y = hi[col].values
        ok = np.isfinite(y)
        out[name] = (float(np.polyfit(hi.index.values[ok], y[ok], 1)[0])
                     if ok.sum() >= 3 else np.nan)
    out["n_r_dropped"] = int(block.r.nunique() - med.index.nunique())
    return out, med


def fit_models(pts, ys):
    """Which of the two bounds -- or their minimum -- explains the observed ceilings.

    ``pts`` are (E_max, N) pairs and ``ys`` the ceiling measured there.  Three models are
    scored: the Takens bound alone, the Eckmann-Ruelle bound alone, and their pointwise
    minimum, each with one free scale per bound so the *shape* is tested rather than the
    constant.  The unscaled (a = b = 1) versions are reported beside them, since a model that
    only fits after its coefficient moves a long way is not the same claim.
    """
    E = np.array([p[0] for p in pts], float)
    N = np.array([p[1] for p in pts], float)
    y = np.asarray(ys, float)
    ok = np.isfinite(y) & np.isfinite(E) & np.isfinite(N)
    E, N, y = E[ok], N[ok], y[ok]
    tak, er = E / 2.0, 2 * np.log10(N)
    if len(y) < 4:
        return {}

    def rmse(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    a = float((tak @ y) / (tak @ tak))
    b = float((er @ y) / (er @ er))
    grid = np.linspace(0.2, 3.0, 141)
    best = min(((rmse(np.minimum(u * tak, v * er)), u, v) for u in grid for v in grid))

    # a fourth model, suggested by the data rather than by either hypothesis: the ceiling
    # is logarithmic in *both* knobs.  Eckmann-Ruelle already has that form in N (its slope
    # is 2.0 per decade); the Takens bound does not have it in E_max, so a good fit here is
    # a statement that the embedding condition is not what the ceiling is obeying.
    X = np.column_stack([np.log10(E), np.log10(N), np.ones_like(E)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    out = dict(n=int(len(y)),
               rmse_takens=rmse(tak), rmse_er=rmse(er),
               rmse_min=rmse(np.minimum(tak, er)),
               a_takens=a, rmse_takens_fit=rmse(a * tak),
               b_er=b, rmse_er_fit=rmse(b * er),
               a_min=best[1], b_min=best[2], rmse_min_fit=best[0],
               loglog_dE=float(coef[0]), loglog_dN=float(coef[1]),
               loglog_const=float(coef[2]),
               rmse_loglog=float(np.sqrt(np.mean(resid ** 2))),
               sd_y=float(np.std(y)))
    return out


def analyse():
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    parts = sorted(OUT.glob("ceiling_raw*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    print(f"read {len(df)} rows from {', '.join(p.name for p in parts)}")
    keys = ["sweep", "arm", "max_E", "N", "r"]
    per_cell = (df.groupby(keys)
                  .agg(MG=("MG", "median"), MG_sd=("MG", "std"),
                       LB=("LB", "median"), PRdelay=("PRdelay", "median"),
                       specPR256=("specPR256", "median"),
                       traj_pr=("traj_pr", "median"), traj_rank=("traj_rank", "median"),
                       rho_ident=("rho_ident", "median"), MG_2E=("MG_2E", "median"),
                       crossings=("crossings", "median"),
                       degen_rate=("degenerate", "mean"),
                       frac_floor=("frac_floor", "median"),
                       theiler_used=("theiler_used", "median"),
                       theiler_requested=("theiler_requested", "median"),
                       n_delay_vectors=("n_delay_vectors", "median"),
                       roughness=("roughness", "median"),
                       margin_res=("margin_res", "median"),
                       n=("MG", "size"))
                  .reset_index())
    per_cell.to_csv(OUT / "ceiling_cells.csv", index=False)

    rows = []
    for (sweep, arm, E, N), blk in df.groupby(["sweep", "arm", "max_E", "N"]):
        c, _ = ceilings(blk)
        rows.append(dict(sweep=sweep, arm=arm, max_E=E, N=N,
                         takens=E / 2.0, eckmann_ruelle=2 * np.log10(N),
                         theiler_used=float(blk.theiler_used.median()),
                         degen_rate=float(blk.degenerate.mean()),
                         rho_ident=float(blk.rho_ident.median(skipna=True)),
                         crossings=float(blk.crossings.median()), **c))
    summ = pd.DataFrame(rows).sort_values(["sweep", "arm", "max_E", "N"])
    summ.to_csv(OUT / "ceiling_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== sweep E (N = 8000) ===")
    print(summ[summ.sweep == "E"].round(2).to_string(index=False))
    print("\n=== sweep N (E_max = 20) ===")
    print(summ[summ.sweep == "N"].round(2).to_string(index=False))
    print("\n=== MG against r ===")
    for (sweep, arm), blk in df.groupby(["sweep", "arm"]):
        idx = "max_E" if sweep == "E" else "N"
        blk = blk[~blk.degenerate.astype(bool)]
        for v in ("MG", "PRdelay"):
            print(f"\n-- sweep {sweep}, arm {arm}: median {v} by r")
            print(blk.pivot_table(index=idx, columns="r", values=v,
                                  aggfunc="median").round(2).to_string())

    # one point per distinct (E_max, N) on the frozen arm; the E_max=20 / N=8000 cell is
    # common to both sweeps and is counted once
    f = summ[summ.arm == "frozen"].drop_duplicates(subset=["max_E", "N"])
    pts = list(zip(f.max_E, f.N))
    fits = {k: fit_models(pts, pd.to_numeric(f[k], errors="coerce")
                          .replace(np.inf, np.nan).values)
            for k in ("r_track_1", "MG_plateau", "PR_plateau")}
    pd.DataFrame(fits).T.to_csv(OUT / "ceiling_fits.csv")
    print("\n=== which bound explains the ceilings (frozen arm, both sweeps pooled) ===")
    print(pd.DataFrame(fits).T.round(3).to_string())

    # the sharpest form of the two predictions is a slope, not a level: Takens says the
    # ceiling rises 0.5 components per unit of E_max, Eckmann-Ruelle 2.0 per decade of N
    sl = []
    for (sweep, arm), g in summ.groupby(["sweep", "arm"]):
        xcol, pred, unit = (("max_E", 0.5, "per unit E_max") if sweep == "E"
                            else (None, 2.0, "per decade of N"))
        x = g.max_E.values.astype(float) if sweep == "E" else np.log10(g.N.values.astype(float))
        for ycol in ("r_track_1", "MG_plateau", "MG_at_20", "slope_top", "slope_top_PR"):
            y = pd.to_numeric(g[ycol], errors="coerce").replace(np.inf, np.nan).values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            a, b = np.polyfit(x[ok], y[ok], 1)
            sl.append(dict(sweep=sweep, arm=arm, quantity=ycol, slope=float(a),
                           intercept=float(b), predicted_slope=pred, units=unit,
                           n=int(ok.sum())))
    slopes = pd.DataFrame(sl)
    slopes.to_csv(OUT / "ceiling_slopes.csv", index=False)
    print("\n=== observed slope against the slope each hypothesis predicts ===")
    print(slopes.round(3).to_string(index=False))
    return df, per_cell, summ


def figure():
    """Ceiling against each knob, beside the two predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = pd.read_csv(OUT / "ceiling_summary.csv")
    cells = pd.read_csv(OUT / "ceiling_cells.csv")
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    # (a) and (b): the raw evidence -- the estimate against the truth, one curve per knob
    ce = cells[(cells.sweep == "E") & (cells.arm == "frozen")]
    for E, g in ce.groupby("max_E"):
        g = g.sort_values("r")
        ax[0, 0].plot(g.r, g.MG, "o-", ms=3, label=f"$E_{{max}}$={E}")
    cn = cells[cells.sweep == "N"]
    for N, g in cn.groupby("N"):
        g = g.sort_values("r").dropna(subset=["MG"])
        if len(g) >= len(R_VALUES):
            ax[0, 1].plot(g.r, g.MG, "o-", ms=3, label=f"N={N}")
    for a, ttl in ((ax[0, 0], f"sweep $E_{{max}}$ at N = {N_REF}"),
                   (ax[0, 1], f"sweep N at $E_{{max}}$ = {E_REF}")):
        a.plot([0, 20], [0, 20], "k--", lw=1, label="truth")
        a.set_xlabel("constructed active dimension r"); a.set_ylabel("MG")
        a.set_title(ttl); a.legend(fontsize=7)

    # (c) and (d): the ceiling measures against each knob, beside the two predictions
    e = summ[(summ.sweep == "E") & (summ.arm == "frozen")].sort_values("max_E")
    ax[1, 0].plot(e.max_E, e.r_track_1, "o-", label="tracking ceiling (|MG-r| < 1)")
    for arm, mk in (("theiler76", "s--"), ("fixedspan", "d--")):
        a = summ[(summ.sweep == "E") & (summ.arm == arm)].sort_values("max_E")
        if len(a):
            ax[1, 0].plot(a.max_E, a.r_track_1, mk, ms=4, label=f"tracking, {arm}")
    ax[1, 0].plot(e.max_E, e.MG_plateau, "^-", label="MG at r >= 16")
    ax[1, 0].plot(e.max_E, e.PR_plateau, "v:", color="grey", label="linear null PRdelay")
    ax[1, 0].plot(e.max_E, e.takens, "k:", label="Takens: $E_{max}/2$")
    ax[1, 0].axhline(2 * np.log10(N_REF), color="r", ls="-.",
                     label=f"Eckmann-Ruelle at N={N_REF}: {2*np.log10(N_REF):.1f}")
    ax[1, 0].set_xlabel("$E_{max}$"); ax[1, 0].set_ylabel("components")
    ax[1, 0].set_title("ceiling against the embedding"); ax[1, 0].legend(fontsize=7)

    for sw, lab, mk in (("N", f"$E_{{max}}$={E_REF}", "o-"), ("N_E56", "$E_{max}$=56", "s-")):
        n = summ[summ.sweep == sw].sort_values("N")
        n = n[np.isfinite(pd.to_numeric(n.MG_plateau, errors="coerce"))]
        if not len(n):
            continue
        ax[1, 1].semilogx(n.N, pd.to_numeric(n.r_track_1, errors="coerce"), mk,
                          label=f"tracking ceiling, {lab}")
        ax[1, 1].semilogx(n.N, n.MG_plateau, mk[0] + "--", label=f"MG at r >= 16, {lab}")
    grid = np.array(N_GRID, float)
    ax[1, 1].semilogx(grid, 2 * np.log10(grid), "r-.", label="Eckmann-Ruelle: $2\\log_{10}N$")
    ax[1, 1].axhline(E_REF / 2, color="k", ls=":", label=f"Takens at $E_{{max}}$={E_REF}")
    ax[1, 1].set_xlabel("record length N"); ax[1, 1].set_ylabel("components")
    ax[1, 1].set_title("ceiling against the record"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ceiling.png", dpi=150)
    print(OUT / "ceiling.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="reproduce the published ceiling")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in the output CSV")
    ap.add_argument("--heavy-jobs", type=int, default=3,
                    help="workers for cells with N >= --heavy-from (memory, not CPU, bound)")
    ap.add_argument("--heavy-from", type=int, default=32000)
    ap.add_argument("--arms", default="frozen,theiler76,fixedspan",
                    help="sweep-E arms; pass an empty string to run the record sweep alone")
    ap.add_argument("--n-sweep-E", type=int, default=E_REF,
                    help="embedding dimension for the record-length sweep")
    ap.add_argument("--n-grid", default=",".join(str(n) for n in N_GRID))
    ap.add_argument("--out", default="ceiling_raw.csv")
    ap.add_argument("--force", action="store_true", help="re-simulate cached trajectories")
    args = ap.parse_args()
    if args.simulate:
        run_simulations(args.jobs, args.force)
    if args.check:
        check(args.jobs)
    if args.score:
        score(args.jobs, args.heavy_jobs, tuple(args.arms.split(",")), args.heavy_from,
              args.out, args.resume, args.n_sweep_E,
              tuple(int(x) for x in args.n_grid.split(",") if x))
    if args.analyse:
        analyse()
    if args.figure:
        figure()
    if not any((args.check, args.simulate, args.score, args.analyse, args.figure)):
        ap.print_help()


if __name__ == "__main__":
    main()

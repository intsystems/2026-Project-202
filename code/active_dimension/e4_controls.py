"""E4 -- things that change without the number of active directions changing.

At fixed r the following must leave MG alone.  Each is a way the earlier reports' "dimension
drop" could be produced by something that is not a change of dimension:

``baseline``    nothing changes.  Its own within-run variation sets the decision threshold,
                so the false-alarm rate is measured against a null estimated from data
                rather than from a guess.
``obs_scale``   the observer is multiplied by a ramp of 10x.  A pure output gain.
``amp_ramp``    the drive amplitude ramps 4x, so the state excursion grows while r does not.
                This is the closest analogue of a weight norm that grows during training.
``lr_step``     the learning rate halves at the midpoint: the same trajectory traversed at a
                different speed, i.e. a different sampling density along the same manifold.
``noise_step``  the injected noise amplitude triples at fixed rank.
``rotate``      a fixed orthogonal rotation of the k coordinates.  Exactly invariant for a
                rotation-invariant observer, not for a fixed projection -- which is a
                property of the observer, and worth separating from a property of MG.
``freq_half`` / ``freq_double``
                the drive frequency band moves by an octave at fixed r.  This changes the
                autocorrelation time and the roughness of every observer while leaving the
                number of active directions untouched, and is the control that matters most:
                ``report_0808.md`` records Spearman 0.934 between the roughness null and the
                LB estimate on this project's real logs.

Decision rule: a control "fires" when the calibrated estimate moves by more than ``delta``,
where ``delta`` is the 95th percentile of the same statistic on the ``baseline`` runs.  The
false-alarm rate is the fraction of control runs that fire, reported per observer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mg as MG
from dynamics import F_FAST, Spec, equalise_gains, simulate
from e2_rank_sweep import SWEEP_OBS as OBSERVERS
from e1_calibration import load_frozen
from runner import get_system

OUT = Path(__file__).resolve().parent / "results" / "e4_controls"
OUT.mkdir(parents=True, exist_ok=True)

T, BURN, R_FIX = 26_000, 4_000, 4
SEEDS = (0, 1, 2, 3, 4)
NW = T - BURN


def build_spec(control, seed, mode):
    kw = dict(seed=seed, k=10, r=R_FIX, T=T, burn=BURN, mode=mode, precondition=True,
              eta=0.15, f0=F_FAST)
    kw.update(drive_amp=0.8, noise_amp=0.0) if mode == "qp" else kw.update(
        drive_amp=0.0, noise_amp=0.08)
    t = np.arange(T)
    if control == "obs_scale":
        kw["obs_scale"] = np.exp(np.log(10.0) * t / T)
    elif control == "amp_ramp":
        kw["amp_scale"] = 1.0 + 3.0 * t / T
    elif control == "lr_step":
        s = np.ones(T); s[T // 2:] = 0.5
        kw["lr_scale"] = s
    elif control == "noise_step":
        s = np.ones(T); s[T // 2:] = 3.0
        kw["noise_scale"] = s
    elif control == "rotate":
        kw["rotate"] = True
    elif control == "freq_half":
        kw["f0"] = F_FAST / 2
    elif control == "freq_double":
        kw["f0"] = F_FAST * 2
    elif control != "baseline":
        raise ValueError(control)
    return Spec(**kw)


def job(control, seed, mode, cfg):
    A, c_star = get_system(seed, 10)
    spec = build_spec(control, seed, mode)
    mix, _ = equalise_gains(A, spec, c_star)
    logs, C, Dm, info = simulate(A, spec, c_star=c_star, mix=mix)

    rows = []
    for o in OBSERVERS:
        x = logs[o]
        if x.std() <= 1e-12:
            continue
        z = (x - x.mean()) / x.std()
        right, tr = MG.sliding(z, cfg)
        v, ok = tr["MG"], ~(tr["degenerate"] > 0.5) & np.isfinite(tr["MG"])
        n = len(v)
        first, second = ok & (np.arange(n) < n // 2), ok & (np.arange(n) >= n // 2)
        rows.append(dict(control=control, mode=mode, seed=seed, observer=o,
                         mg_all=float(np.median(v[ok])) if ok.any() else np.nan,
                         mg_first=float(np.median(v[first])) if first.any() else np.nan,
                         mg_second=float(np.median(v[second])) if second.any() else np.nan,
                         rough_all=float(np.median(tr["roughness"][ok])) if ok.any() else np.nan,
                         acorr_all=float(np.median(tr["acorr"][ok])) if ok.any() else np.nan,
                         prd_all=float(np.median(tr["PRdelay"][ok])) if ok.any() else np.nan,
                         spec_all=float(np.median(tr["specPR0"][ok])) if ok.any() else np.nan,
                         traj_PR=info["traj_PR"], n_windows=int(ok.sum())))
    return rows


#: ``noise_step`` is dropped from the ``qp`` arm: there ``noise_amp = 0``, so the per-step
#: gain multiplies zero and the run is bit-identical to ``baseline`` -- a guaranteed 0% row.
#: ``rotate`` is scored on the projection observers only: it acts through ``R @ c``, so the
#: other thirteen observers are untouched by construction and would dilute the rate to zero.
CONTROLS = ("baseline", "obs_scale", "amp_ramp", "lr_step", "noise_step",
            "rotate", "freq_half", "freq_double")
ROTATE_OBS = ("c_proj1", "c_proj2", "c_proj3")


def main():
    cfg, _ = load_frozen()
    t0 = time.time()
    jobs = [(c, s, m, cfg) for m in ("qp", "noise") for c in CONTROLS for s in SEEDS
            if not (m == "noise" and c in ("amp_ramp", "freq_half", "freq_double"))
            and not (m == "qp" and c == "noise_step")]
    print(f"{len(jobs)} control runs, config {cfg}", flush=True)
    res = Parallel(n_jobs=12, verbose=5, batch_size=1)(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "controls_raw.csv", index=False)

    df = df[~((df.control == "rotate") & (~df.observer.isin(ROTATE_OBS)))]
    df["within"] = (df.mg_second - df.mg_first).abs()

    # Two statistics, two nulls.  `within` is a paired half-to-half difference inside one
    # run; `between` compares a run's level with the baseline level and additionally carries
    # seed-to-seed variance.  Sharing one threshold gives the two arms of the OR different
    # and unmeasured sizes.  The baseline's own `between` null is computed leave-one-out, so
    # it is not shrunk toward zero by letting each run help define its own reference.
    # Bonferroni: the 97.5th percentile of each, for a nominal 5% overall.
    base = df[df.control == "baseline"]
    lev = base.groupby(["mode", "observer"]).mg_all.median().rename("base_level")
    d = df.merge(lev, on=["mode", "observer"], how="left")
    d["between"] = (d.mg_all - d.base_level).abs()

    loo = []
    for (mo, ob), g in base.groupby(["mode", "observer"]):
        v = g.mg_all.values
        for i in range(len(v)):
            other = np.delete(v, i)
            if len(other):
                loo.append(dict(mode=mo, observer=ob,
                                loo=abs(v[i] - float(np.median(other)))))
    loo = pd.DataFrame(loo)
    tw = base.groupby(["mode", "observer"]).within.quantile(0.975).rename("delta_within")
    tb = loo.groupby(["mode", "observer"]).loo.quantile(0.975).rename("delta_between")
    d = d.merge(tw, on=["mode", "observer"], how="left").merge(
        tb, on=["mode", "observer"], how="left")
    d["fires_within"] = d.within > d.delta_within
    d["fires_between"] = d.between > d.delta_between
    d["fires"] = d.fires_within | d.fires_between
    d.to_csv(OUT / "controls_scored.csv", index=False)

    print(f"\ndone in {time.time()-t0:.0f}s")
    print("\n=== false alarms per control.  The 16 observers are functions of the same c, "
          "so this is a cell fraction, not a rate over independent trials ===")
    for col in ("fires_within", "fires_between", "fires"):
        print(f"\n-- {col} --")
        print(d.pivot_table(index="control", columns="mode", values=col,
                            aggfunc="mean").round(3).to_string())
    print("\n=== median MG level by control (qp arm), with the nulls ===")
    q = d[d["mode"] == "qp"]
    print(q.pivot_table(index="observer", columns="control",
                        values="mg_all", aggfunc="median").round(2).to_string())
    print("\n=== roughness null by control (qp arm) ===")
    print(q.pivot_table(index="observer", columns="control",
                        values="rough_all", aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    main()

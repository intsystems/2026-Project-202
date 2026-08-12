"""Known-active-dimension calibration, k=1..20.

This is the direct experiment proposed in the discussion:

* real sklearn digits data and a frozen nonlinear backbone;
* a trainable adapter with 20 available orthonormal directions;
* r independently excited directions, r=1,...,20;
* the active dimension is measured from the full trajectory covariance and
  update covariance (PR/PCA), never assumed from the label r;
* MG, LB, TwoNN, delay-PR and spectral PR are evaluated on several scalar
  observers;
* full-batch recurrent/transient and mini-batch/noise arms are kept separate;
* a held-out calibration split freezes one estimator configuration before the
  test ranks are scored.

The script is deliberately self-contained but reuses the audited implementations
in this directory.  It writes raw trajectories as NPZ and all summaries as CSV,
so the expensive simulation and the estimator sweep can be repeated separately.

Run from this directory:

    python calibration_k20.py --simulate
    python calibration_k20.py --score
    python calibration_k20.py --report

The default ``--fast`` settings take minutes on a laptop.  Remove ``--fast`` for
the larger digits backbone and longer trajectories.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "k20_calibration"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from dynamics import F_FAST, F_SLOW, Spec, equalise_gains, simulate  # noqa: E402
from mg import MGConfig, all_estimators, summarise  # noqa: E402
from system import build  # noqa: E402


R_VALUES = tuple(range(1, 21))
CAL_R = (2, 6, 10, 14, 18)
TEST_R = tuple(r for r in R_VALUES if r not in CAL_R)
SEEDS = (0, 1, 2)
MODES = ("qp", "qp_slow", "noise", "batch_proj", "gd")
OBSERVERS = ("w_fro", "c_norm", "g_fro", "g_proj", "c_proj1",
             "fn_fro", "fn_proj1", "loss_full")
CAL_OBSERVERS = ("w_fro", "c_norm", "g_fro", "c_proj1")


def configs(fast: bool = True):
    """Small predeclared grid; no data-generating parameter is tuned here."""
    # Four configurations are enough to expose the two important axes without
    # turning this validation into a hyperparameter hunt: embedding capacity E
    # and temporal span tau.  Window length is fixed before seeing the answer.
    if fast:
        windows = (8000,)
        max_es = (20, 40)
    else:
        windows = (10000,)
        max_es = (20, 40)
    out = []
    for E in max_es:
        for tau in (4, 16):
            for W in windows:
                out.append(MGConfig(max_E=E, tau=tau, k_neighbors=20,
                                    theiler="autocorr", window=W,
                                    stride=max(1000, W // 2)))
    return out


def make_spec(mode, r, seed, fast=True, **kw):
    if fast:
        T, burn = 12000, 2000
    else:
        T, burn = 20000, 3000
    base = dict(seed=seed, k=20, r=r, T=T, burn=burn, mode=mode,
                precondition=True, eta=0.15, f0=F_FAST,
                drive_amp=0.8, noise_amp=0.0, n_groups=24)
    if mode == "qp_slow":
        base.update(mode="qp", f0=F_SLOW)
    elif mode == "noise":
        base.update(drive_amp=0.0, noise_amp=0.08)
    elif mode == "batch_proj":
        base.update(drive_amp=0.0, noise_amp=3.0, batch=64)
    elif mode == "gd":
        base.update(drive_amp=0.0, noise_amp=0.0, eta=0.006,
                    precondition=False, burn=0, gd_disp=1.0)
    elif mode != "qp":
        raise ValueError(mode)
    base.update(kw)
    return Spec(**base)


def build_system(seed: int, fast: bool):
    if fast:
        return build(seed=seed, k=20, n_train=512, n_probe=256,
                     hidden=(64, 64), backbone_steps=1000)
    return build(seed=seed, k=20, n_train=1024, n_probe=384,
                 hidden=(96, 96), backbone_steps=2000)


def simulate_one(mode, r, seed, fast=True, tag=None, **kw):
    tag = tag or mode
    path = OUT / "trajectories" / f"{tag}_r{r:02d}_s{seed:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    A = build_system(seed, fast)
    c_star = A.solve()
    spec = make_spec(mode, r, seed, fast=fast, **kw)
    mix, cond = equalise_gains(A, spec, c_star)
    logs, C, D, info = simulate(A, spec, c_star=c_star, mix=mix)
    arrays = {f"log__{k}": v for k, v in logs.items()}
    arrays["C"] = C.astype(np.float32)
    arrays["D"] = D.astype(np.float32)
    arrays["info_json"] = np.asarray(json.dumps({**info, "drive_cond": cond}))
    arrays["spec_json"] = np.asarray(json.dumps({**dataclasses.asdict(spec),
                                                  "mode": mode, "tag": tag}, default=str))
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    info = json.loads(str(z["info_json"]))
    spec = json.loads(str(z["spec_json"]))
    logs = {k[5:]: z[k] for k in z.files if k.startswith("log__")}
    return logs, z["C"], z["D"], info, spec


def run_simulations(fast=True):
    jobs = []
    # Primary test: every r, three independent seeds.
    for r in R_VALUES:
        for seed in SEEDS:
            jobs.append(("qp", r, seed, {}))

    # Robustness arms.  The complete r=1..20 grid is not repeated because these
    # arms answer a different question: whether changing excitation class while
    # keeping measured active PR fixed changes the scalar estimate.  Five anchor
    # ranks cover the full range without obscuring the main test.
    anchors = (1, 5, 10, 15, 20)
    for mode in ("qp_slow", "noise", "batch_proj", "gd"):
        for r in anchors:
            jobs.append((mode, r, 0, {}))
    # Fixed-r nuisance controls.  They test scale and observer-basis dependence,
    # not active-r recovery, so they are intentionally small.
    r0 = 10
    for name, kw in (
        # ``Spec.obs_scale`` is indexed on the complete trajectory and is cut
        # after burn-in inside ``simulate``.
        ("qp_scale2", {"obs_scale": np.full(12000 if fast else 20000, 2.0)}),
        ("qp_rotate", {"rotate": True}),
    ):
        jobs.append(("qp", r0, 0, {"tag": name, **kw}))

    print(f"simulating {len(jobs)} trajectories", flush=True)
    for i, (mode, r, seed, kw) in enumerate(jobs, 1):
        t0 = time.time()
        simulate_one(mode, r, seed, fast=fast, **kw)
        print(f"[{i}/{len(jobs)}] {mode:10s} r={r:2d} seed={seed} "
              f"({time.time()-t0:.1f}s)", flush=True)


def rank_configs(cal, cfgs):
    """Stage-1 selection: rank (config, observer) pairs and freeze the winning config.

    Extracted from :func:`score` so that ``--rank-only`` can regenerate
    ``config_observer_ranking.csv`` and ``frozen_k20.json`` from the committed
    ``calibration_configs.csv`` in seconds, without re-simulating or re-scoring.  That
    matters: the version of this file that shipped before Installment 5 scored an *isotonic*
    fit of MG onto the truth, fitted on the same five calibration ranks it then scored.  An
    isotonic fit interpolates five monotone points exactly, so every cal_mae came out 0.000
    and the four configurations tied -- see the comment below, which is why the fit was
    removed.  The code was fixed and the CSV was not regenerated, so the committed ranking
    file disagreed with the committed script until this path existed.  The selection outcome
    was unaffected (``frozen_k20.json`` already recorded cfg_id 3, which is what the raw
    criterion picks; the stale tie would have picked cfg_id 0).
    """
    ranking = []
    for (cid, obs), g in cal.groupby(["cfg_id", "observer"]):
        med = g.groupby("r").MG.median()
        truth = g.groupby("r").traj_pr.median()
        if len(med) < 3 or med.isna().any():
            continue
        # Select the estimator by its *raw* numerical agreement with the known
        # trajectory dimension.  Isotonic regression is deliberately not used
        # here: with only five calibration ranks it can interpolate any
        # monotone curve and make all configurations look equally good.
        pred = med.values
        ranking.append(dict(cfg_id=cid, observer=obs,
                            cal_mae=float(np.mean(np.abs(pred-truth.values))),
                            cal_rho=float(pd.Series(med).corr(pd.Series(truth), method="spearman")),
                            cal_degenerate=float(g.frac_degenerate.mean())))
    rank = pd.DataFrame(ranking)
    rank["score"] = rank.cal_mae + 0.25 * (1-rank.cal_rho.fillna(0)) + 2*rank.cal_degenerate
    rank = rank.sort_values("score")
    rank.to_csv(OUT / "config_observer_ranking.csv", index=False)

    # Freeze the best config globally by median calibration score across observers.
    best_cfg = int(rank.groupby("cfg_id").score.median().idxmin())
    with open(OUT / "frozen_k20.json", "w") as fh:
        json.dump(dict(cfg_id=best_cfg, config=cfgs[best_cfg].as_dict(),
                       calibration_r=list(CAL_R), calibration_seeds=list(SEEDS),
                       selection="minimum median raw absolute error across "
                                 "non-degenerate calibration observers",
                       observers=OBSERVERS), fh, indent=2)
    return rank, best_cfg


def rank_only(fast=True):
    """Regenerate the stage-1 ranking from the committed calibration_configs.csv."""
    p = OUT / "calibration_configs.csv"
    if not p.exists():
        raise RuntimeError("run --score first")
    rank, best = rank_configs(pd.read_csv(p), configs(fast))
    print(rank.to_string(index=False))
    print(f"\nfrozen cfg_id = {best}: {configs(fast)[best]}")
    return rank, best


def score(fast=True):
    cfgs = configs(fast)
    paths = sorted((OUT / "trajectories").glob("*.npz"))
    paths = [p for p in paths if not p.stem.startswith("smoke")]

    # Stage 1: estimator selection only on calibration ranks and four observer
    # families.  Test ranks and robustness arms are not touched during selection.
    cal_rows = []
    for p in paths:
        logs, C, D, info, spec = load_npz(p)
        mode, r, seed = spec["mode"], int(spec["r"]), int(spec["seed"])
        tag = spec.get("tag", mode)
        # Older partially completed trajectories were written before the
        # explicit tag field was added.  Recover the semantic arm from the
        # filename rather than silently merging qp_slow/controls into qp.
        stem = p.stem
        if stem.startswith("qp_slow_"):
            tag = "qp_slow"
        elif stem.startswith("qp_scale2_"):
            tag = "qp_scale2"
        elif stem.startswith("qp_rotate_"):
            tag = "qp_rotate"
        if tag != "qp" or r not in CAL_R:
            continue
        for cid, cfg in enumerate(cfgs):
            for obs in CAL_OBSERVERS:
                x = np.asarray(logs[obs], float)
                z = (x - x.mean()) / x.std()
                rec = summarise(z, cfg, seed=seed)
                cal_rows.append(dict(file=p.name, tag=tag, mode=mode, r=r,
                                     seed=seed, cfg_id=cid, observer=obs,
                                     traj_pr=info["traj_PR"], **rec))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration_configs.csv", index=False)
    if cal.empty:
        raise RuntimeError("No calibration trajectories; run --simulate first")

    rank, best_cfg = rank_configs(cal, cfgs)

    # Stage 2: one frozen configuration on every trajectory and observer.
    cfg = cfgs[best_cfg]
    rows = []
    for p in paths:
        logs, C, D, info, spec = load_npz(p)
        mode, r, seed = spec["mode"], int(spec["r"]), int(spec["seed"])
        tag = spec.get("tag", mode)
        stem = p.stem
        if stem.startswith("qp_slow_"):
            tag = "qp_slow"
        elif stem.startswith("qp_scale2_"):
            tag = "qp_scale2"
        elif stem.startswith("qp_rotate_"):
            tag = "qp_rotate"
        for obs in OBSERVERS:
            x = np.asarray(logs[obs], float)
            if x.size < cfg.window or x.std() <= 1e-12:
                continue
            z = (x - x.mean()) / x.std()
            rec = summarise(z, cfg, seed=seed)
            rows.append(dict(file=p.name, tag=tag, mode=mode, r=r, seed=seed,
                             cfg_id=best_cfg, observer=obs,
                             available=info.get("available", 20),
                             functional_rank=info.get("func_rank"),
                             functional_pr=info.get("func_PR"),
                             traj_rank=info.get("traj_rank"),
                             traj_pr=info.get("traj_PR"),
                             update_pr=info.get("upd_PR"), **rec))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores_frozen.csv", index=False)

    test = df[df.r.isin(TEST_R)]
    # Use only fast qp for clean held-out recovery, then report other dynamics separately.
    clean = test[test.tag == "qp"]
    # Observer-specific affine calibration, fitted on CAL_R only and then frozen.
    cal_best = df[(df.tag == "qp") & df.r.isin(CAL_R)]
    maps = {}
    for obs, g in cal_best.groupby("observer"):
        a, b = np.polyfit(g.MG.values, g.traj_pr.values, 1)
        maps[obs] = (float(a), float(b))
    clean = clean.copy()
    clean["MG_cal"] = [maps[o][0] * x + maps[o][1]
                       for o, x in zip(clean.observer, clean.MG)]
    summary = (clean.groupby("observer").agg(
        rho=("MG_cal", lambda s: float(pd.Series(s).corr(pd.Series(clean.loc[s.index,"traj_pr"]),method="spearman"))),
        mae=("MG_cal", lambda s: float(np.mean(np.abs(s-clean.loc[s.index,"traj_pr"])))),
        mg_median=("MG", "median"), truth_median=("traj_pr", "median"),
        degenerate=("frac_degenerate", "mean"))
        .reset_index())
    summary.to_csv(OUT / "heldout_qp_summary.csv", index=False)
    # Per-mode and per-r medians at the frozen config.
    all_best = df
    med = (all_best.groupby(["tag","mode","observer","r"])
           .agg(MG=("MG","median"), LB=("LB","median"), TwoNN=("TwoNN","median"),
                PRdelay=("PRdelay","median"), traj_pr=("traj_pr","median"),
                update_pr=("update_pr","median"), degenerate=("frac_degenerate","mean"))
           .reset_index())
    med.to_csv(OUT / "frozen_per_r.csv", index=False)
    print("frozen config:", cfgs[best_cfg])
    print(summary.round(3).to_string(index=False))
    return df, rank, summary, med


def report():
    p = OUT
    files = [p / "frozen_k20.json", p / "heldout_qp_summary.csv", p / "frozen_per_r.csv"]
    if not all(x.exists() for x in files):
        raise RuntimeError("run --score first")
    frozen = json.loads((p / "frozen_k20.json").read_text())
    s = pd.read_csv(p / "heldout_qp_summary.csv")
    m = pd.read_csv(p / "frozen_per_r.csv")
    lines = ["# Known-active-dimension calibration (k=1..20)", "",
             "## Design", "",
             "A frozen nonlinear MLP backbone is followed by a trainable adapter with 20 fixed orthonormal parameter directions. The dynamics excite r directions, r=1,...,20. The true active dimension is measured from the trajectory covariance (participation ratio), not equated to r. Functional rank is measured from the held-out-logit Jacobian.", "",
             "The primary arm is deterministic and recurrent (incommensurate sinusoidal forcing). Additional arms are a slow recurrent torus, rank-r stochastic forcing, projected mini-batch noise, and full-batch transient descent. Fixed-r controls test coordinate rotation and constant observer scaling.", "",
             "## Frozen estimator", "", f"The estimator configuration was selected only on r={frozen['calibration_r']} and seeds={frozen['calibration_seeds']}; test ranks are the complementary values. Configuration: `{frozen['config']}`.", "",
             # to_string, not to_markdown: to_markdown needs `tabulate`, which is not
             # installed here and is in no requirements file, so --report raised ImportError
             # on every invocation.  That is why this directory had no report.md at all.
             "## Held-out recurrent test", "",
             "```", s.round(3).to_string(index=False), "```", "",
             "## Interpretation", "",
             "For the recurrent arm, a successful estimator should increase monotonically with measured trajectory PR and remain stable across observers and seeds. The stochastic and transient arms are deliberately not expected to have an r-dimensional deterministic attractor; they test whether MG spuriously reports the injected rank.", "",
             "The decisive comparison is MG versus the directly measured `traj_pr`/`update_pr` and versus linear `PRdelay`. A good result supports equality only for the identifiable recurrent regime, not for arbitrary training trajectories.", ""]
    # NOT report.md.  ``results/k20_calibration/report.md`` is the hand-written write-up of
    # this experiment -- the one the audit found missing -- and it records two defects in this
    # directory that no generator can restate.  This function only assembles the machine
    # tables, so it writes beside it rather than over it.
    (p / "report_auto.md").write_text("\n".join(lines), encoding="utf-8")
    print(p / "report_auto.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--rank-only", action="store_true",
                    help="redo stage-1 selection from the committed calibration_configs.csv")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()
    if args.simulate:
        run_simulations(args.fast)
    if args.score:
        score(args.fast)
    if args.rank_only:
        rank_only(args.fast)
    if args.report:
        report()
    if not (args.simulate or args.score or args.rank_only or args.report):
        ap.print_help()


if __name__ == "__main__":
    main()

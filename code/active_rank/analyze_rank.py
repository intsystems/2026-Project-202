"""Does the parameter trajectory's rank fall at grokking?

Reads the sketches written by ``run_rank.py`` and the matching training log, and computes,
over sliding windows aligned to the run's own memorisation and generalisation steps:

``PR_pos``       participation ratio of the window's *position* covariance.  Dominated by
                 whatever direction the trajectory is drifting along, so it answers "how
                 many directions has it moved in", including a slow monotone drift.
``PR_pos_det``   the same after removing a linear trend per coordinate.  A steady drift is
                 rank 1 and would otherwise mask everything else; this is the honest
                 version of the question.
``PR_step``      participation ratio of the *increment* covariance.  "How many directions
                 is the optimiser exploring right now", trend-free by construction.
``PR_step_m``    the same on increments block-averaged over m logged steps, which
                 suppresses the mini-batch noise floor and leaves the drift.
``PR_fn``        the same statistics on the normalised probe logits: how many directions
                 the computed *function* moves in.

Nulls reported beside every one of them, because a participation ratio can move for
reasons that have nothing to do with rank: ``move`` (total displacement in the window) and
``pnorm`` (the weight norm, which weight decay drives mechanically).

    python analyze_rank.py --indir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def pr(X, tol=1e-12):
    """Participation ratio (Sum lambda)^2 / Sum lambda^2 of the row cloud of X."""
    X = np.asarray(X, float)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return np.nan
    s = np.linalg.svd(X, compute_uv=False) ** 2
    t = s.sum()
    return float(t * t / (s * s).sum()) if t > tol else np.nan


def detrend(X):
    t = np.arange(len(X), dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    beta = (t[:, None] * X).sum(0) / (t @ t)
    return X - np.outer(t, beta) - X.mean(0, keepdims=True)


def block_mean(X, m):
    n = (len(X) // m) * m
    return X[:n].reshape(-1, m, X.shape[1]).mean(1) if n >= m else X[:0]


def milestones(log, thresh=0.95):
    """(t_mem, t_gen) from the run's own log: first step at which each accuracy crosses."""
    out = []
    for col in ("train_acc", "val_acc"):
        hit = log.index[log[col] >= thresh]
        out.append(int(log.step.iloc[hit[0]]) if len(hit) else None)
    return tuple(out)


def sliding(npz, log, window=200, stride=25, smooth=(1, 5, 20)):
    step = npz["step"]
    z, zf = npz["z"], npz["zf"]                      # (T, n_sketch, dim)
    move = npz["param_step"]
    pnorm = npz["param_norm"]
    rows = []
    for a in range(0, len(step) - window + 1, stride):
        b = a + window
        rec = {"right_step": int(step[b - 1]), "left_step": int(step[a])}
        for name, arr in (("", z), ("fn_", zf)):
            per_sketch = {k: [] for k in ["pos", "pos_det", "step"] +
                          [f"step{m}" for m in smooth]}
            for s in range(arr.shape[1]):
                W = arr[a:b, s, :]
                per_sketch["pos"].append(pr(W))
                per_sketch["pos_det"].append(pr(detrend(W)))
                D = np.diff(W, axis=0)
                per_sketch["step"].append(pr(D))
                for m in smooth:
                    per_sketch[f"step{m}"].append(pr(block_mean(D, m)) if m > 1 else np.nan)
            for k, v in per_sketch.items():
                rec[f"{name}PR_{k}"] = float(np.nanmean(v))
                rec[f"{name}PR_{k}_sketchsd"] = float(np.nanstd(v))
        rec["move"] = float(np.nansum(move[a:b]))
        rec["pnorm"] = float(np.nanmean(pnorm[a:b]))
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default=str(HERE / "results"))
    p.add_argument("--outdir", default=str(HERE / "results"))
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--stride", type=int, default=25)
    a = p.parse_args(argv)

    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames, summary = [], []
    for npz_path in sorted(indir.glob("*_rank.npz")):
        tag = npz_path.stem.replace("_rank", "")
        csv = indir / f"{tag}_train.csv"
        if not csv.exists():
            cands = sorted(indir.glob("*.csv"))
            print(f"[{tag}] no {csv.name}; candidates: {[c.name for c in cands]}")
            continue
        npz, log = np.load(npz_path, allow_pickle=True), pd.read_csv(csv)
        t_mem, t_gen = milestones(log)
        df = sliding(npz, log, a.window, a.stride)
        df.insert(0, "run", tag)
        df["t_mem"], df["t_gen"] = t_mem, t_gen
        frames.append(df)
        summary.append(dict(run=tag, n_rows=len(log), n_params=int(npz["n_params"]),
                            dim=int(npz["dim"]), t_mem=t_mem, t_gen=t_gen,
                            final_val_acc=float(log.val_acc.iloc[-1])))
        print(f"[{tag}] t_mem={t_mem} t_gen={t_gen} windows={len(df)}")
    if not frames:
        raise SystemExit(f"no *_rank.npz under {indir}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(outdir / "rank_windows.csv", index=False)
    pd.DataFrame(summary).to_csv(outdir / "rank_summary.csv", index=False)
    (outdir / "rank_milestones.json").write_text(json.dumps(summary, indent=1))
    print(f"\n-> {outdir/'rank_windows.csv'}  ({len(out)} windows)")

    for run, g in out.groupby("run"):
        tg = g.t_gen.iloc[0]
        print(f"\n=== {run}  (t_gen={tg}) ===")
        cols = ["PR_pos_det", "PR_step", "PR_step5", "PR_step20",
                "fn_PR_pos_det", "fn_PR_step", "move"]
        if tg is None or (isinstance(tg, float) and np.isnan(tg)):
            print(g[["right_step"] + cols].iloc[::max(1, len(g) // 8)].round(2)
                  .to_string(index=False))
        else:
            g = g.assign(phase=np.where(g.right_step < tg, "before", "after"))
            print(g.groupby("phase")[cols].median().round(2).to_string())


if __name__ == "__main__":
    raise SystemExit(main())

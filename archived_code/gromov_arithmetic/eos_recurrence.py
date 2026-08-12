"""Does an edge-of-stability log RECUR, or does it only oscillate?

The article's two admissibility diagnostics do not answer this.  The identifiability
ratio catches an estimate that is a property of the embedding space; the trend-crossing
count tests non-monotonicity of the series, and the article says in as many words that
it does not test recurrence in the reconstructed space.  An edge-of-stability run is
non-monotone by construction -- that is what the edge of stability is -- so it can clear
the trend-crossing condition without being in the regime the estimator needs.  This
module measures the missing quantity directly.

THE STATISTIC.  For each reconstructed point, find its nearest neighbour surviving the
Theiler exclusion, and ask what that distance means.  Dividing by the radius of the
cloud is the obvious normalisation and it is the wrong one: a trajectory that merely
moves slowly also has close surviving neighbours, without ever coming back.  The
discriminating comparison is against how far the orbit itself travels during the
exclusion,

    ratio = || y_t - y_nn ||  /  || y_{t+W_T} - y_t || ,

with ``y_nn`` the nearest neighbour at least ``W_T`` samples away in time.

    ratio << 1   the nearest surviving neighbour is far closer than the orbit's own
                 displacement over the exclusion window, so the orbit came back;
    ratio ~ 1    it is simply the same curve continuing past the exclusion, which is
                 what a transient does.

Two controls fix the scale in every run of this script, because the statistic is only
interpretable against them: a two-frequency torus of the kind the article's ladder
builds, and a plain exponential decay.

    python eos_recurrence.py --results ./results/eos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def embed(x, E, tau):
    n = len(x) - (E - 1) * tau
    if n <= 0:
        raise ValueError(f"series of {len(x)} too short for E={E}, tau={tau}")
    return np.stack([x[i:i + n] for i in range(0, E * tau, tau)], axis=1)


def recurrence(x, E=20, tau=1, wt=150, probes=2000):
    """Return (nn/travel, nn/cloud-radius, points used).  See the module docstring."""
    z = (x - x.mean()) / x.std()
    Y = embed(z, E, tau)
    n = len(Y)
    tree = cKDTree(Y)
    # Enough neighbours that at least one is guaranteed to survive the exclusion: the
    # excluded band holds at most 2*wt+1 points, so 2*wt+5 always leaves a survivor.
    k = min(n - 1, 2 * wt + 5)
    dist, idx = tree.query(Y, k=k + 1)
    t = np.arange(n)
    step = max(1, (n - wt) // probes)
    nn, travel = [], []
    for i in range(0, n - wt, step):
        ok = np.abs(idx[i] - t[i]) > wt
        ok[0] = False                       # the point itself
        if ok.any():
            nn.append(dist[i][ok][0])
            travel.append(float(np.linalg.norm(Y[i + wt] - Y[i])))
    nn, travel = np.asarray(nn), np.asarray(travel)
    if nn.size == 0:
        return float("nan"), float("nan"), 0
    good = travel > 0
    scale = float(np.sqrt(np.sum(Y.var(axis=0))))
    return (float(np.median(nn[good] / travel[good])),
            float(np.median(nn) / scale) if scale > 0 else float("nan"),
            int(nn.size))


def controls(n, wt, E, tau, period=400.0):
    """The two reference series, on the same length and the same statistic."""
    t = np.arange(n)
    phi = 0.5 * (1 + 5 ** 0.5)              # incommensurate second frequency
    torus = np.sin(2 * np.pi * t / period) + 0.8 * np.sin(2 * np.pi * t / (period * phi))
    ramp = np.exp(-t / (n / 2.667))
    out = []
    for name, s in (("control: 2-torus", torus), ("control: monotone decay", ramp)):
        r, o, m = recurrence(s, E=E, tau=tau, wt=wt)
        out.append(dict(run=name, lr=np.nan, seed=np.nan, window_start=0,
                        nn_over_travel=r, nn_over_scale=o, points=m,
                        rises=float((np.diff(s) > 0).mean()),
                        eta_lam_over_2=np.nan, at_eos=False, diverged=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results/eos")
    ap.add_argument("--column", default="train_loss")
    ap.add_argument("--window", type=int, default=8000, help="samples per window")
    ap.add_argument("--stride", type=int, default=4000)
    ap.add_argument("--max-E", type=int, default=20)
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--theiler", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    paths = sorted(root.glob("*_train.csv"))
    if not paths:
        raise SystemExit(f"no *_train.csv under {root}")

    rows = controls(args.window, args.theiler, args.max_E, args.tau)
    for path in paths:
        key = path.stem.replace("_train", "")
        meta_path = root / f"{key}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        x = pd.read_csv(path)[args.column].to_numpy(float)
        t_grok = meta.get("t_grok")
        # Read only after the transition: before it every run is a monotone approach
        # and the question does not arise.
        start0 = int(t_grok) if t_grok is not None else 0
        ratio = meta.get("eta_lam_over_2_median_tail")
        at_eos = bool(ratio is not None and ratio > 0.9
                      and meta.get("diverged_at") is None)
        for s in range(start0, len(x) - args.window + 1, args.stride):
            seg = x[s:s + args.window]
            if not np.isfinite(seg).all() or seg.std() <= 0:
                continue
            r, o, m = recurrence(seg, E=args.max_E, tau=args.tau, wt=args.theiler)
            rows.append(dict(run=key, lr=meta.get("lr"), seed=meta.get("seed"),
                             window_start=s, nn_over_travel=r, nn_over_scale=o,
                             points=m, rises=float((np.diff(seg) > 0).mean()),
                             eta_lam_over_2=ratio, at_eos=at_eos,
                             diverged=meta.get("diverged_at") is not None))
        print(f"  {key}: {len([r for r in rows if r['run'] == key])} window(s)",
              flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else root / "eos_recurrence.csv"
    df.to_csv(out, index=False)

    g = (df.groupby(["run", "lr", "at_eos", "diverged"], dropna=False)
         .agg(nn_over_travel=("nn_over_travel", "median"),
              nn_over_scale=("nn_over_scale", "median"),
              rises=("rises", "median"), n=("nn_over_travel", "size"))
         .reset_index().sort_values(["lr", "run"]))
    g.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n" + g.round(5).to_string(index=False))
    print("\nnn_over_travel << 1 is recurrence; ~1 is a curve continuing past the "
          "exclusion.\nRead every row against the two control rows.")


if __name__ == "__main__":
    main()

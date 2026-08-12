"""Apply the article's admissibility diagnostics to the edge-of-stability logs.

``eos.py`` produces full-batch runs logged at every optimiser step, some of which sit at
the edge of stability -- the sharpness self-limits at ``2/eta`` and the loss stops being
monotone.  That is the one candidate the literature offers for a *deterministic,
recurrent* training run, which is the only regime in which the article's estimate may be
read as a count.  This module asks the article's own question of those logs: do the two
diagnostics of Sec. 3.4 admit them?

Three things are varied, and each exists because it could otherwise fake an answer.

LOGGING STRIDE.  Edge-of-stability oscillation is the two-cycle of the unstable mode, so
its period is a few optimiser steps.  Every full-batch log previously committed here is
written at a stride of 10 or more, which does not blur that oscillation but aliases it
away.  The published protocol would therefore have been blind to it whatever the run did,
and re-reading each log at strides 1, 2, 5, 10 and 50 shows how much of the answer is a
property of the sampling rather than of the optimiser.

DELAY LAG.  The frozen configuration uses ``tau = 4``, calibrated on systems whose period
is about 400 samples.  Against a two-step cycle an even lag is close to the worst possible
choice: sampling a period-2 signal every 4 steps returns a constant.  The frozen value is
still reported first, because it is the protocol as published and changing it per dataset
is exactly the per-system tuning the article forbids; but ``tau = 1`` and ``tau = 2`` are
reported beside it, and the gap between them is the honest size of that mismatch.

SEGMENT.  Sharpness only reaches ``2/eta`` after the transition, so the whole record mixes
a monotone approach with whatever follows it.  Each run is therefore read twice: over the
whole record, and over the post-transition segment alone, where the stability ratio is
actually near one.  ``--sharp`` joins the measured ratio onto every window.

    python eos_probe.py --results ./results/eos --out ./results/eos/eos_diagnostics.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_AD = Path(__file__).resolve().parent.parent / "active_dimension"
if str(_AD) not in sys.path:
    sys.path.insert(0, str(_AD))

try:
    import mg as MG
except ImportError as exc:  # pragma: no cover - depends on a sibling folder
    raise SystemExit(f"cannot import ../active_dimension/mg.py ({exc}); "
                     f"this script is a bridge to that analysis and needs it present")


def frozen_config():
    """The E1 frozen config, or the module default, reported either way."""
    try:
        from e1_calibration import load_frozen
        cfg, _ = load_frozen()
        return cfg, "frozen (e1_calibration)"
    except Exception:
        return MG.DEFAULT, "module default (e1 config not found)"


def crossings(x):
    """Sign changes of the linearly detrended window -- the article's trend-crossing count.

    Same behaviour as ``dimension_probe.oscillations`` and ``e5_real_logs``, so the values
    here sit on the scale the article's admissible rectangle is drawn on.
    """
    t = np.arange(len(x), dtype=float)
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    return int(np.count_nonzero(np.diff(np.signbit(d))))


def rises(x):
    """Fraction of consecutive samples on which the series increases.

    The cheapest possible non-monotonicity statistic, and the one that needs no
    embedding: at the edge of stability it should be near a half, and on a monotone
    descent exactly zero.
    """
    return float((np.diff(x) > 0).mean()) if len(x) > 1 else float("nan")


def probe_series(x, cfg, window_steps, stride_steps, sub):
    """Slide the estimator over one already-subsampled series.

    ``window_steps`` and ``stride_steps`` are in OPTIMISER STEPS, so that changing the
    logging stride changes the number of samples in a window and not the span of wall
    time it covers.  That is the comparison the docstring is about; holding the sample
    count fixed instead would silently lengthen the window as the stride grew.
    """
    w = max(4, window_steps // sub)
    s = max(1, stride_steps // sub)
    cfg1 = dataclasses.replace(cfg, window=w, stride=s)
    cfg2 = dataclasses.replace(cfg1, max_E=2 * cfg1.max_E)
    out = []
    for start in MG.window_starts(len(x), cfg1):
        seg = x[start:start + w]
        if len(seg) < w or not np.isfinite(seg).all() or seg.std() <= 1e-12:
            continue
        z = (seg - seg.mean()) / seg.std()
        e1 = MG.all_estimators(z, cfg1)
        e2 = MG.all_estimators(z, cfg2)
        out.append(dict(
            start_sample=int(start), n_samples=int(w),
            MG=e1["MG"], MG_2E=e2["MG"],
            ident_ratio=(e2["MG"] / e1["MG"]) if e1["MG"] else np.nan,
            PRdelay=e1["PRdelay"], roughness=e1["roughness"], acorr=e1["acorr"],
            crossings=crossings(seg), rises=rises(seg),
            degenerate=bool(e1["degenerate"])))
    return out


def _task(job):
    """One (run, observer, segment, stride, lag) cell.  Top level so it can be pickled."""
    (key, lr, seed, column, seg_name, series, sub, tau, t_grok, eta,
     base, wsteps, ssteps) = job
    xs = series[::sub]
    if len(xs) < 8:
        return []
    cfg = dataclasses.replace(base, tau=tau)
    return [dict(run=key, lr=lr, seed=seed, column=column, segment=seg_name,
                 subsample=sub, tau=tau, t_grok=t_grok, eta_lam_over_2=eta, **r)
            for r in probe_series(xs, cfg, wsteps, ssteps, sub)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results/eos")
    ap.add_argument("--columns", nargs="+", default=["train_loss", "weight_norm"])
    ap.add_argument("--subsample", type=int, nargs="+", default=[1, 2, 5, 10, 50])
    ap.add_argument("--taus", type=int, nargs="+", default=[4, 1, 2])
    ap.add_argument("--window-steps", type=int, default=8000)
    ap.add_argument("--stride-steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    paths = sorted(root.glob("*_train.csv"))
    if not paths:
        raise SystemExit(f"no *_train.csv under {root}")
    base, provenance = frozen_config()
    print(f"{len(paths)} run(s); MG config: {provenance}\n  {base}")
    print(f"window {args.window_steps} steps, stride {args.stride_steps} steps, "
          f"subsampling {args.subsample}, taus {args.taus}\n", flush=True)

    jobs = []
    for path in paths:
        key = path.stem.replace("_train", "")
        df = pd.read_csv(path)
        meta_path = root / f"{key}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        t_grok = meta.get("t_grok")
        eta = meta.get("eta_lam_over_2_median_tail")

        for column in args.columns:
            if column not in df.columns:
                continue
            full = df[column].to_numpy(float)
            segments = {"all": full}
            # The post-transition segment is where the stability ratio is actually near
            # one; over the whole record it is dominated by the monotone approach.
            if t_grok is not None and int(t_grok) + 2000 < len(full):
                segments["post"] = full[int(t_grok):]
            for seg_name, series in segments.items():
                for sub in args.subsample:
                    for tau in args.taus:
                        jobs.append((key, meta.get("lr"), meta.get("seed"), column,
                                     seg_name, series, sub, tau, t_grok, eta,
                                     base, args.window_steps, args.stride_steps))

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(jobs)} cells on {workers} worker(s)", flush=True)
    rows = []
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            rows.extend(_task(job))
            print(f"  [{i}/{len(jobs)}] {job[0]}:{job[3]}:{job[4]}:"
                  f"sub{job[6]}:tau{job[7]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, got in enumerate(pool.map(_task, jobs), 1):
                rows.extend(got)
                if i % 10 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] cells done", flush=True)

    out = Path(args.out) if args.out else root / "eos_diagnostics.csv"
    full_df = pd.DataFrame(rows)
    if full_df.empty:
        raise SystemExit("no usable windows")
    full_df.to_csv(out, index=False)

    g = (full_df.groupby(["run", "lr", "column", "segment", "subsample", "tau"])
         .agg(MG=("MG", "median"), ident=("ident_ratio", "median"),
              PRdelay=("PRdelay", "median"), crossings=("crossings", "median"),
              rises=("rises", "median"), degen=("degenerate", "mean"),
              eta_lam=("eta_lam_over_2", "first"), n=("MG", "size"))
         .reset_index())
    # The article's admissible rectangle: every accurate case of its validation fell at
    # rho_ident <= 1.10 with more than eight trend crossings.  It describes the
    # calibration and is not a validated decision rule, which is why it is applied here
    # rather than trusted.
    g["admissible"] = (g["ident"] <= 1.10) & (g["crossings"] > 8)
    g.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    print(f"\nwrote {out} and its summary")
    show = g[(g["column"] == "train_loss")].sort_values(
        ["segment", "tau", "lr", "subsample"])
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(show.round(4).to_string(index=False))
    n_adm = int(g["admissible"].sum())
    print(f"\nadmissible cells: {n_adm} of {len(g)}")
    if n_adm:
        print(g[g["admissible"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

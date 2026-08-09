"""Run the project's dimension estimators on these logs, against a known truth.

``../active_dimension/e5_real_logs.py`` places the existing transformer logs on the E0
atlas but cannot say whether any estimate is *correct*, because nothing about those
runs fixes the dimension of what is being learned.  These runs do fix it: Sec. 3 of
arXiv:2301.02679 gives the grokked solution in closed form, and it uses ``(p+1)/2``
Fourier modes -- 49 at p=97 -- so the representation the network converges to has a
dimension that is known by construction rather than estimated.

That makes two comparisons available on the same time axis:

*the scalar traces*  MG, PRdelay and friends applied to ``train_loss`` / ``val_loss`` /
                     ``weight_norm``, exactly as ``e5_real_logs.py`` does;
*the representation*  the Fourier IPR and the effective rank recorded in
                     ``<key>_obs.csv``, which say directly how many modes are active.

Nothing here asserts the two should agree.  The point is that on these logs the
question is answerable, and on the older ones it was not.

    python dimension_probe.py --results ./results/arith
    python dimension_probe.py --results ./results/arith --columns train_loss weight_norm
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
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


def oscillations(x):
    """Mean-crossings of the linearly detrended window -- how often it comes back.

    Copied in behaviour from ``e5_real_logs.py`` so the numbers here sit on the same
    scale as the ones already reported for the transformer logs.
    """
    t = np.arange(len(x), dtype=float)
    d = x - np.polyval(np.polyfit(t, x, 1), t)
    return int(np.count_nonzero(np.diff(np.signbit(d))))


def frozen_config():
    """The E1 frozen config if it exists, else the module default, reported either way."""
    try:
        from e1_calibration import load_frozen
        cfg, _ = load_frozen()
        return cfg, "frozen (e1_calibration)"
    except Exception:
        return MG.DEFAULT, "module default (e1 config not found)"


def probe(path: Path, column, cfg):
    df = pd.read_csv(path)
    if column not in df.columns:
        return []
    x = df[column].to_numpy(float)
    step = df["step"].to_numpy()
    cfg = dataclasses.replace(cfg, window=min(cfg.window, max(2000, len(x) // 3)),
                              stride=1000)
    cfg2 = dataclasses.replace(cfg, max_E=2 * cfg.max_E)
    out = []
    for s in MG.window_starts(len(x), cfg):
        w = x[s:s + cfg.window]
        if not np.isfinite(w).all() or w.std() <= 1e-12:
            continue
        z = (w - w.mean()) / w.std()
        e1, e2 = MG.all_estimators(z, cfg), MG.all_estimators(z, cfg2)
        out.append(dict(run=path.stem.replace("_train", ""), column=column,
                        right_step=int(step[s + cfg.window - 1]),
                        MG=e1["MG"], MG_2E=e2["MG"],
                        ident_ratio=(e2["MG"] / e1["MG"]) if e1["MG"] else np.nan,
                        PRdelay=e1["PRdelay"], roughness=e1["roughness"],
                        acorr=e1["acorr"], oscillations=oscillations(w),
                        degenerate=bool(e1["degenerate"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results/arith")
    ap.add_argument("--columns", nargs="+",
                    default=["train_loss", "val_loss", "weight_norm"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results)
    paths = sorted(root.glob("*_train.csv"))
    if not paths:
        raise SystemExit(f"no *_train.csv under {root}")
    cfg, provenance = frozen_config()
    print(f"{len(paths)} run(s) x {len(args.columns)} column(s); MG config: {provenance}")
    print(f"  {cfg}\n", flush=True)

    rows = []
    for p in paths:
        for c in args.columns:
            got = probe(p, c, cfg)
            if not got:
                # Silence here once hid that a run had been dropped: a log shorter than
                # the 2000-sample floor yields no windows at all, and the run simply
                # vanished from the summary table as if it had never been asked for.
                print(f"  [skip] {p.name}:{c} -- no usable window "
                      f"(short log, missing column, or a flat trace)", flush=True)
            rows.extend(got)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable windows -- are the logs long enough?")

    g = (df.groupby(["run", "column"])
           .agg(MG=("MG", "median"), MG_2E=("MG_2E", "median"),
                ident=("ident_ratio", "median"), PRdelay=("PRdelay", "median"),
                roughness=("roughness", "median"), acorr=("acorr", "median"),
                osc=("oscillations", "median"), degen=("degenerate", "mean"),
                n=("MG", "size")).reset_index())
    print(g.round(3).to_string(index=False))

    out = Path(args.out) if args.out else root / "dimension_probe.csv"
    df.to_csv(out, index=False)
    g.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"\nwrote {out} and its summary")
    print("\nknown truth for the grokked runs: the analytic solution of Sec. 3.1 uses")
    print("(p+1)/2 = 49 Fourier modes at p=97; <key>_obs.csv carries the measured")
    print("Fourier IPR and effective rank on the same step axis.")


if __name__ == "__main__":
    main()

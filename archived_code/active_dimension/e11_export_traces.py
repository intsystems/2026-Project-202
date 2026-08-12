"""Export one raw scalar log per synthetic regime, small enough to commit.

``e11_theiler_contrast.py --simulate`` writes ~24 MB of float64 traces under
``results/e11_theiler_contrast/series/``, which is gitignored: it is an intermediate the
scorer reads, and it regenerates exactly. But the article wants to *show* a raw series
per regime, and a figure must be buildable from committed files alone -- the whole point
of ``icomp_v2/make_figures.py`` is that ``python make_figures.py`` works in a fresh
clone. So this writes a small extract, one column per arm, that is committed.

    python e11_export_traces.py            # after e11_theiler_contrast.py --simulate

The observer is ``w_fro``, the Frobenius norm of the weights: it is one of the two
parameter norms, which are what a training run ordinarily logs, and it is the observer
the article's own real-log analysis uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "results" / "e11_theiler_contrast" / "series"
DEFAULT_OUT = HERE / "results" / "e11_theiler_contrast" / "example_traces.csv"

# One arm per dynamical regime of the article's table 1 that this experiment simulates.
# The stochastic regime has no synthetic arm here; the article's figure takes it from a
# real mini-batch training log, which is committed in its own right.
ARMS = [("fast", "recurrent"), ("transient", "transient")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--observer", default="w_fro")
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8000)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"{src} not found -- run "
                         f"`python e11_theiler_contrast.py --simulate` first")

    out = {}
    for arm, label in ARMS:
        path = src / f"{arm}_r{args.r}_s{args.seed}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        with np.load(path) as z:
            key = f"log__{args.observer}"
            if key not in z:
                raise SystemExit(f"{path} has no {key}; has {list(z.keys())}")
            out[label] = np.asarray(z[key], dtype=float)[:args.samples]

    n = min(len(v) for v in out.values())
    df = pd.DataFrame({"sample": np.arange(n), **{k: v[:n] for k, v in out.items()}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Full precision: the figure plots these directly and a rounded copy would make the
    # committed file disagree with the series it was taken from.
    df.to_csv(args.out, index=False, float_format="%.17g")
    print(f"wrote {args.out}  ({n} samples x {len(out)} arms, "
          f"observer {args.observer}, r={args.r}, seed={args.seed})")
    for k, v in out.items():
        print(f"  {k:<12} mean {v.mean():.6g}  sd {v.std():.6g}  "
              f"rises {float((np.diff(v[:n]) > 0).mean()):.3f}")


if __name__ == "__main__":
    main()

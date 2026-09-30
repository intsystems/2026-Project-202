"""E7b, Colab side: pull small scalar observers out of the saved E7 logs.

No retraining. The E7 logs already hold (i) norms of the stem, layer1-4 and fc and
(ii) a 1 024-dim CountSketch of every parameter update. The running sum of one
sketch coordinate is a fixed sparse random projection of theta_t - theta_0 -- the
"fixed parameter projection" observer, the most accurate one in the article's
section 5.2. The full-network norm turned out to be a smooth monotone curve on
ResNet-18 (two trend crossings per window), so these are the observers to test.

Writes one file per scenario, a few tens of MB, to download and analyse locally.
"""
import argparse
from pathlib import Path

import numpy as np

N_PROJ = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the e7_results folder")
    args = ap.parse_args()
    for sc in ("scratch", "finetune"):
        d = Path(args.root) / sc
        out = {}
        for f in sorted(d.glob("logs_*_s*.npz")):
            run = f.stem[5:]
            z = np.load(f)
            for k in z.files:
                if k != "update_sketch":
                    out[f"{run}|{k}"] = z[k].astype(np.float64)
            proj = np.cumsum(z["update_sketch"][:, :N_PROJ].astype(np.float64), axis=0)
            for j in range(N_PROJ):
                out[f"{run}|proj{j}"] = proj[:, j]
        np.savez_compressed(Path(args.root) / f"observers_{sc}.npz", **out)
        print(sc, len({k.split('|')[0] for k in out}), "runs", len(out), "series")


if __name__ == "__main__":
    main()

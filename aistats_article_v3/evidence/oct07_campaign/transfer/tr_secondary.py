"""Secondary analysis windows (ADDENDUM 05:30 in tr_main.py): scalar statistics on the
per-window linearly detrended parameter-norm log. Run after training (2 workers)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs"


def detrend(seg):
    t = np.arange(len(seg), dtype=float); t -= t.mean()
    return seg - seg.mean() - t * (t @ (seg - seg.mean())) / (t @ t)


def job(npz):
    import tr_monitors as Mo
    out = npz.with_name(npz.stem + "_wind.csv")
    if out.exists():
        return
    x = np.load(npz)["log_pn"].astype(float)
    rows = []
    for a in range(0, len(x) - Mo.W + 1, Mo.S):
        seg = detrend(x[a:a + Mo.W])
        r = {"start": a}
        for k, f in Mo.SCALAR.items():
            try:
                r[f"{k}_dt"] = float(f(seg))
            except Exception:
                r[f"{k}_dt"] = np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    files = sorted(f for f in RUNS.glob("*.npz") if f.with_name(f.stem + "_win.csv").exists())
    with Pool(2) as p:
        p.map(job, files, chunksize=4)
    print(len(files), "done")

"""Pilot (seed 100 only): which LR / beta2 settings of the tiny transformer give loss spikes
or divergence? Looks only at the training behaviour (loss and internals), never at MG."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, itertools, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "pilot"

def job(a):
    import spk_common as C
    lr, b2, tag = a
    f = OUT / f"p_{tag}_lr{lr}_b{b2}.npz"
    if f.exists():
        return
    r = C.train(100, lr, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 4000, warm=200, beta2=b2, sched="const")
    np.savez(f, **r)
    print(tag, lr, b2, "div", r["diverged_at"], "sec", round(r["seconds"]), flush=True)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grid = [(lr, b2, "a") for lr, b2 in itertools.product((3e-3, 1e-2, 3e-2, 1e-1), (0.95, 0.99))]
    with Pool(3) as p:
        p.map(job, grid, chunksize=1)

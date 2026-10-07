"""Pilot on seed 999 (setup only): look at ground-truth measures and training behaviour of
the candidate events; never at MG or competitors."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tr_common as C

JOBS = [("source", None, 100.0), ("source", "freeze", 100.0), ("source", "prune", 100.0),
        ("source", "wd", 100.0), ("source", "wd", 30.0), ("source", "wd", 300.0),
        ("T8_restart", None, 100.0), ("T1_width4", "wd", 100.0)]


def job(a):
    setup, ev, f = a
    out = HERE / "pilot" / f"{setup}_{ev}_{int(f)}_s999.npz"
    if not out.exists():
        C.run(setup, 999, ev, wd_factor=f, out=out)
    return str(out)


if __name__ == "__main__":
    with Pool(2) as p:
        for r in p.imap_unordered(job, JOBS):
            print(r, flush=True)

"""Pilot 2 (seed 100 only): settings with weight decay 0 and/or beta2 0.999 to provoke spikes.
Usage: python pilot2.py tag lr beta2 wd steps [batch] [eps] [warm]. Looks only at loss/internals."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spk_common as C
tag, lr, b2, wd, steps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
batch = int(sys.argv[6]) if len(sys.argv) > 6 else 32
eps = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-8
warm = int(sys.argv[8]) if len(sys.argv) > 8 else 200
out = HERE / "pilot2"; out.mkdir(exist_ok=True)
r = C.train(100, lr, steps=steps, warm=warm, beta2=b2, wd=wd, batch=batch, eps=eps, sched="const")
np.savez(out / f"p_{tag}_lr{lr}_b{b2}_wd{wd}_bs{batch}_eps{eps}_w{warm}.npz", **r)
print(tag, lr, b2, wd, batch, eps, "div", r["diverged_at"], "sec", round(r["seconds"]), flush=True)

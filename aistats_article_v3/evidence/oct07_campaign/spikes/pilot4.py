"""Pilot 4 (seed 100 only): modular addition (slingshot regime, Thilak et al. 2022), Adam, no wd.
Usage: python pilot3.py tag lr beta2 steps [eps]. Looks only at loss/internals, never at MG."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spk_common as C
tag, lr, b2, steps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
eps = float(sys.argv[5]); tf = float(sys.argv[6])
out = HERE / "pilot4"; out.mkdir(exist_ok=True)
f = out / f"p_{tag}_lr{lr}_b{b2}_eps{eps}_tf{tf}"
r = C.train(100, lr, steps=steps, warm=100, beta2=b2, wd=0.0, eps=eps, d=64, layers=2, batch=256,
            task="mod", train_frac=tf, log_prefix=f, diverge_loss=20.0)
np.savez(str(f) + ".npz", **r)
print(tag, lr, b2, eps, "div", r["diverged_at"], "sec", round(r["seconds"]), flush=True)

"""Pilot (seed 99 only): which configurations lose plasticity on permuted MNIST, and does a
full reset / shrink-and-perturb restore it? Looks only at ground truth (online accuracy per
task) and internal probes, never at MG or competitors."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from pathlib import Path

import numpy as np
import pl_common as P

OUT = Path(__file__).resolve().parent / "pilot"
OUT.mkdir(exist_ok=True)
CFGS = {
    "adam1e-3_w256": dict(opt="adam", lr=1e-3, width=256),
    "adam3e-3_w256": dict(opt="adam", lr=3e-3, width=256),
    "adam1e-2_w256": dict(opt="adam", lr=1e-2, width=256),
    "sgd0.1_w256": dict(opt="sgd", lr=0.1, width=256),
    "sgd0.3_w256": dict(opt="sgd", lr=0.3, width=256),
    "adam3e-3_w64": dict(opt="adam", lr=3e-3, width=64),
    "adam3e-3_w256_label": dict(opt="adam", lr=3e-3, width=256, task="label"),
    "adam3e-3_w256_every_sp": dict(opt="adam", lr=3e-3, width=256, every="sp"),
    "adam3e-3_w256_every_reset": dict(opt="adam", lr=3e-3, width=256, every="reset"),
}

if __name__ == "__main__":
    names = sys.argv[1:]
    for name in names:
        c = dict(CFGS[name]); every = c.pop("every", None)
        cfg = dict(depth=2, wd=0.0, steps=400, batch=32, **c)
        pol = (lambda logs, tasks, k: True) if every else None
        print(name, flush=True)
        r = P.run_stream(cfg, 99, 60, policy=pol, intervention=every or "reset", verbose=True)
        np.savez_compressed(OUT / f"{name}.npz", **r["logs"])
        json.dump(r["tasks"], open(OUT / f"{name}.json", "w"))

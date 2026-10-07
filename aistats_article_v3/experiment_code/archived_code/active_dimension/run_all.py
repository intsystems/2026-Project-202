"""Run E0, E2, E3, E4, E5 in sequence, then the tables and the figures.

E1 must already have been run: everything downstream reads the frozen estimator
configuration from ``results/e1_calibration/frozen_config.json``.

    python e1_calibration.py && python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: e0_atlas.py and e6_tau_sensitivity.py are listed first because everything downstream is
#: interpreted against them; they are commented out here only because this particular chain
#: was restarted after they had already completed.
STEPS = [  # "e0_atlas.py", "e6_tau_sensitivity.py",
    "e2_rank_sweep.py", "e3_transitions.py",
    "e4_controls.py", "e5_real_logs.py", "figures.py"]

if __name__ == "__main__":
    t0 = time.time()
    for s in STEPS:
        print(f"\n{'=' * 70}\n=== {s}   [t+{time.time() - t0:.0f}s]\n{'=' * 70}", flush=True)
        log = HERE / (Path(s).stem.split('_')[0] + ".log")
        with open(log, "w") as fh:
            rc = subprocess.call([sys.executable, str(HERE / s)], cwd=HERE,
                                 stdout=fh, stderr=subprocess.STDOUT)
        print(f"--- exit {rc}, log -> {log}", flush=True)
        if rc != 0:
            print(log.read_text()[-3000:], flush=True)
    with open(HERE / "results" / "tables.txt", "w") as fh:
        subprocess.call([sys.executable, str(HERE / "analyze.py")], cwd=HERE,
                        stdout=fh, stderr=subprocess.STDOUT)
    print(f"\nALL DONE in {time.time() - t0:.0f}s")

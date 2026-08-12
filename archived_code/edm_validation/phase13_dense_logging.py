"""Log every optimisation step, so that a locally stationary analysis becomes possible.

Section 3 of the report measures *global* recurrence and finds it absent: the training
trajectory does not return near its past states. That rules out treating a run as a single
invariant set, but it does not rule out the weaker and more useful hypothesis that the
dynamics are *locally* stationary, drifting slowly compared with the analysis window.

Whether that hypothesis is even testable depends on the logging interval, and at the
interval used so far it is not. With one row per ten optimisation steps and a median
memorisation-to-generalisation gap of 2875 steps:

    analysis window                    span at log_every=10     fraction of the gap
    recurrence probe (E=5, tau=5)       200 steps                 7.0%
    delay vector (E_max=15, tau=1)      150 steps                 5.2%
    dimension window (300 samples)     3000 steps               104.3%

The dimension window is the fatal one. A 300-sample window straddles the entire transition,
so it was never a local measurement of anything. Shrinking it to be genuinely local leaves
roughly fifteen points inside it, which is far too few to estimate a dimension. Logging
every step resolves both horns: the same 300-sample window becomes 300 steps, about a tenth
of the gap, while still containing 300 points.

This script produces those dense logs. It writes no probe, since the questions here are
about recurrence and stationarity of the scalar series, and the probe would dominate the
cost at this logging rate.

Only the cheap series need the dense rate. ``train_loss`` is already computed by the
training step and ``weight_norm`` is a reduction over the parameters, whereas a validation
pass is a forward pass over the held-out split and would dominate the cost if it ran every
step. ``val_every`` decouples them: the reconstruction series are logged every step and the
validation split is evaluated at the old rate of once per ten steps, which is ample for
locating the transition.

    python phase13_dense_logging.py --outdir /content/out/dense
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "grokking_train" / "train.py"

# Seed 1 is used because its gap of 1570 steps is typical of the configuration, unlike the
# single-stream run whose 11790 is an outlier (report section 6). max_steps is cut to 6000
# because generalisation occurs near step 2920 and a validation pass now runs on every
# step, which makes the full 20000-step horizon needlessly expensive.
RUNS = {
    "grok_dense": ("mod_wd1", {"seed": "1", "csv": "grok_dense.csv"}),
    "lowdata15_dense": ("mod_wd1", {"seed": "1", "fraction": "0.15",
                                    "csv": "lowdata15_dense.csv"}),
    "wd0_dense": ("mod_wd0", {"seed": "1", "csv": "wd0_dense.csv"}),
}
MAX_STEPS = "6000"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="dense_logs")
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--max-steps", default=MAX_STEPS)
    parser.add_argument("--val-every", default="10",
                        help="validation interval in steps; the dense series are logged "
                             "every step regardless")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name in args.runs:
        base, overrides = RUNS[name]
        if (outdir / overrides["csv"]).exists():
            print(f"[{name}] already present, skipping", flush=True)
            continue
        cmd = [sys.executable, "-u", str(TRAIN), base, "--outdir", str(outdir), "--force",
               "--quiet", "--set", "log_every=1", "--set", f"val_every={args.val_every}",
               "--set", f"max_steps={args.max_steps}"]
        for key, value in overrides.items():
            cmd += ["--set", f"{key}={value}"]
        print(f"\n{'=' * 60}\n[{name}] {' '.join(cmd[3:])}\n{'=' * 60}", flush=True)
        result = subprocess.run(cmd)
        print(f"[{name}] exit {result.returncode}", flush=True)

    print("\nfiles:", sorted(p.name for p in outdir.glob("*.csv")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replicate the function-space velocity comparison across seeds.

The falsification of the velocity signal (report section 3.2) originally rested on one run
per condition. A claim that a statistic fails its control should not depend on a single
draw, particularly given that the delay itself varies by a factor of four across seeds. This
repeats every condition at three seeds so the comparison can be shown with dispersion.

Conditions, all on modular addition with the Omnigrok transformer:

    grok        weight decay 1.0, fraction 0.30   generalises after a delay
    wd0         weight decay 0.0, fraction 0.30   memorises, never generalises
    lowdata15   weight decay 1.0, fraction 0.15   memorises, never generalises
    lowdata20   weight decay 1.0, fraction 0.20   memorises, never generalises

The contrast that matters is `grok` against `lowdata*`: weight decay is identical and only
the quantity of data differs, so a statistic that reports impending generalisation must
separate them.

    python velocity_replicates.py --outdir /content/out/velocity
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "prediction_improved"))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

CONDITIONS = {
    "grok": ("mod_wd1", {}),
    "wd0": ("mod_wd0", {}),
    "lowdata15": ("mod_wd1", {"fraction": "0.15"}),
    "lowdata20": ("mod_wd1", {"fraction": "0.20"}),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="velocity_logs")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        help="subset to run; the grokking condition is already "
                             "replicated at three seeds by controls.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--n-probe", type=int, default=256)
    args = parser.parse_args(argv)

    import run_probe

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chosen = {k: v for k, v in CONDITIONS.items() if k in args.conditions}
    total = len(chosen) * len(args.seeds)
    n = 0
    for name, (base, overrides) in chosen.items():
        for seed in args.seeds:
            n += 1
            tag = f"{name}_s{seed}"
            print(f"\n{'=' * 60}\n[{n}/{total}] {tag}\n{'=' * 60}", flush=True)
            argv_run = [base, "--outdir", str(outdir), "--tag", tag, "--force",
                        "--progress-every", "200", "--n-probe", str(args.n_probe),
                        # One snapshot only: the per-step series is what is analysed and
                        # the full logit matrices would be ~20 MB per run.
                        "--snapshot-every", "100000",
                        "--set", f"seed={seed}"]
            for key, value in overrides.items():
                argv_run += ["--set", f"{key}={value}"]
            try:
                run_probe.main(argv_run)
            except Exception as exc:                        # noqa: BLE001
                print(f"[{tag}] FAILED: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

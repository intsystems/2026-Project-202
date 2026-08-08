"""Phase 5: repeat the Phase 3 test on our own architecture and task.

Phase 3's result -- cross mapping recovers an injected driver from a 1-D training-loss
log, 5/5 with no false positives -- came from ResNet-18 / CIFAR-10 logs produced by
somebody else's script. A claim that rests on one dataset and one training script is one
coincidence away from being wrong.

This repeats the design on a 1-layer transformer solving modular addition, with drivers
we control and a ghost that is logged but never applied (`inject.py`), and asks the same
question with the same nulls. Agreement across two settings that share nothing but the
method is what makes it a property of the method.

    python phase5_inject_ccm.py [logdir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccm import ccm_test                                    # noqa: E402

TRUTH = {"logistic": True, "sinusoid": True, "iid": True, "ghost": False}
LIBRARY_SIZES = (20, 40, 80, 160, 320, 640, 1200, 1900)
E, TAU, N_SURROGATES = 3, 1, 39


def main(logdir="results/inject"):
    root = Path(logdir)
    if not root.is_dir():
        print(f"no such directory: {root}  (run phase5_inject_run.py first)")
        return 1

    rows = []
    for name, expected in TRUTH.items():
        path = root / f"inject_{name}.csv"
        if not path.exists():
            print(f"  {name:<10} missing {path.name}")
            continue
        frame = pd.read_csv(path)
        driver = frame["driver"].to_numpy(dtype=float)
        response = frame["train_loss"].to_numpy(dtype=float)
        if driver.std() == 0:
            print(f"  {name:<10} constant driver, skipped")
            continue

        # Raw response, per the Phase 3 finding that differencing erases slow drivers;
        # both nulls and the ghost run control for the shared training trend.
        iaaft = ccm_test(response, driver, E=E, tau=TAU, library_sizes=LIBRARY_SIZES,
                         surrogate_kind="iaaft", n_surrogates=N_SURROGATES, seed=0)
        shift = ccm_test(response, driver, E=E, tau=TAU, library_sizes=LIBRARY_SIZES,
                         surrogate_kind="shift", n_surrogates=N_SURROGATES, seed=0)
        detected = iaaft["detected"] or shift["detected"]
        ok = detected == expected

        rows.append({"driver": name, "expected": expected, "detected": detected,
                     "rho": iaaft["rho_max"], "gain": iaaft["gain"],
                     "p_iaaft": iaaft["p"], "p_shift": shift["p"],
                     "realised_mean": float(frame["realised_fraction"].mean())})
        curve = " ".join(f"{v:.2f}" for v in iaaft["curve"].values())
        print(f"  {name:<10} rho={iaaft['rho_max']:+.3f} gain={iaaft['gain']:+.3f} "
              f"p_iaaft={iaaft['p']:.3f} p_shift={shift['p']:.3f} "
              f"applied={frame['realised_fraction'].mean():.3f} "
              f"-> {'DETECT' if detected else 'none  '} {'OK' if ok else 'MISMATCH'}"
              f"   [{curve}]", flush=True)

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "phase5_inject_ccm.csv", index=False)

    if not table.empty:
        tp = int((table.detected & table.expected).sum())
        fp = int((table.detected & ~table.expected.astype(bool)).sum())
        fn = int((~table.detected & table.expected).sum())
        tn = int((~table.detected & ~table.expected.astype(bool)).sum())
        print(f"\n  true positives {tp}  false negatives {fn}")
        print(f"  false positives {fp}  true negatives {tn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/inject"))

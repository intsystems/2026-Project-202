"""Train grokking runs with an injected driver, logging the driver alongside the loss.

Produces the data for the second, independent test of the Phase 3 claim: a different
architecture (1-layer transformer), a different task (modular addition), a different
optimizer setup, and drivers we control. Analysis is `phase5_inject_ccm.py`.

    python phase5_inject_run.py --outdir /content/out/inject
    python phase5_inject_run.py --drivers logistic ghost --steps 8000
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from inject import DRIVERS, LabelPoisoner                   # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drivers", nargs="+", default=list(DRIVERS))
    parser.add_argument("--outdir", default="inject_logs")
    parser.add_argument("--base", default="mod_wd1")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import runs                                             # deferred: needs torch
    from grok import train, tasks

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = runs.get(args.base).with_overrides({"max_steps": str(args.steps)})
    # num_classes is needed to draw corrupted labels; build the task once to read it.
    num_classes = tasks.from_config(config, device="cpu").num_classes

    for name in args.drivers:
        if name not in DRIVERS:
            raise SystemExit(f"unknown driver '{name}'. Known: {sorted(DRIVERS)}")
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)

        poisoner = LabelPoisoner(DRIVERS[name](seed=args.seed), num_classes,
                                 seed=args.seed)
        frame, _ = train(config, outdir=None, progress=False, batch_hook=poisoner)

        # The loop logs every `log_every` steps; the driver fires every step. Align by
        # sampling the driver at the logged steps, which is also what a real monitor
        # would have available.
        logged_steps = frame["step"].to_numpy().astype(int)
        driver = np.asarray(poisoner.values, dtype=float)
        realised = np.asarray(poisoner.realised, dtype=float)
        keep = logged_steps < len(driver)
        out = frame.loc[keep].copy()
        out["driver"] = driver[logged_steps[keep]]
        out["realised_fraction"] = realised[logged_steps[keep]]

        path = outdir / f"inject_{name}.csv"
        out.to_csv(path, index=False)
        grokked = out[out["val_acc"] >= 0.95]["step"]
        print(f"    wrote {path} ({len(out)} rows)")
        print(f"    driver mean {out.driver.mean():.4f} sd {out.driver.std():.4f}; "
              f"realised mean {out.realised_fraction.mean():.4f}")
        print(f"    grokking: "
              f"{'not reached' if len(grokked) == 0 else f'step {grokked.iloc[0]:.0f}'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

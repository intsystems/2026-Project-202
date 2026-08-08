"""Prove the probe cannot change the training run.

Grokking is fragile and the trainer's RNG discipline is subtle: ``grok.tasks`` seeds
the global torch RNG, and the train/val split, the weight initialisation and the
mini-batch order all continue that same stream. One stray draw shifts the initial
weights and the run is a different experiment -- possibly one that never grokks.

So before any GPU time is spent, train the *same* config twice, once with
``observer=None`` and once with the probe attached, and require the two logs to be
equal bit for bit. Any deviation means the probe is participating in training.

    python verify_noninvasive.py            # 400 steps on CPU
    python verify_noninvasive.py 200        # fewer, when CPU time is scarce
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import numpy as np                                          # noqa: E402
import runs                                                 # noqa: E402
from grok import train                                      # noqa: E402
from probe import LogitProbe                                # noqa: E402

STEPS = 400


def main(steps=STEPS):
    config = runs.get("mod_wd1").with_overrides(
        {"max_steps": steps, "device": "cpu", "log_every": 10}
    )
    print(f"config: {config.summary()}\n")

    print("--- run A: no observer (the proven path) ---")
    baseline, _ = train(config, outdir=None, progress=False)

    print("\n--- run B: identical config, probe attached ---")
    probe = LogitProbe(n_probe=64, n_projections=8, seed=0, snapshot_every=5)
    probed, _ = train(config, outdir=None, progress=False, observer=probe)

    print("\n=== training logs: A vs B ===")
    failures = []
    if list(baseline.columns) != list(probed.columns):
        failures.append(f"columns differ: {list(baseline.columns)} vs {list(probed.columns)}")
    else:
        for column in baseline.columns:
            a = baseline[column].to_numpy()
            b = probed[column].to_numpy()
            identical = np.array_equal(a, b)
            worst = 0.0 if identical else float(np.nanmax(np.abs(a - b)))
            print(f"  {column:<12} identical={identical}  max|diff|={worst:.3e}")
            if not identical:
                failures.append(f"{column} differs by up to {worst:.3e}")

    print("\n=== probe output sanity ===")
    frame = probe.to_frame()
    print(f"  rows: {len(frame)} (expected {len(baseline)})")
    print(f"  columns: {len(frame.columns)}")
    if len(frame) != len(baseline):
        failures.append(f"probe logged {len(frame)} rows, training logged {len(baseline)}")

    for source in ("train", "val"):
        series = frame[f"{source}_p00"].to_numpy()
        velocity = frame[f"{source}_velocity"].to_numpy()[1:]      # first is NaN by design
        print(f"  {source}: p00 range [{series.min():.4f}, {series.max():.4f}]  "
              f"std={series.std():.2e}")
        print(f"  {source}: velocity range [{np.nanmin(velocity):.2e}, "
              f"{np.nanmax(velocity):.2e}]")
        if not np.all(np.isfinite(series)):
            failures.append(f"{source}_p00 contains non-finite values")
        if series.std() == 0.0:
            failures.append(f"{source}_p00 is constant -- the observable is dead")
        if not np.all(np.isfinite(velocity)):
            failures.append(f"{source}_velocity contains non-finite values after the first")

    print()
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        return 1
    print("PASS: the probe is non-invasive -- training logs are bit-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else STEPS))

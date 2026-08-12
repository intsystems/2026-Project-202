"""The training log must be bit-identical with and without :class:`RankProbe`.

``grok/tasks.py`` seeds one global torch RNG stream that the train/val split, the weight
initialisation and the mini-batch order all continue, so a single stray draw changes the
initial weights and can destroy grokking.  This check is the reason the probe builds its
hash and its probe indices from NumPy and saves/restores the torch RNG state around every
forward pass.

    python verify_noninvasive.py            # ~1 min on CPU
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import runs                                                 # noqa: E402
from grok import train                                      # noqa: E402
from rank_probe import RankProbe                            # noqa: E402

STEPS = 400


def main():
    config = runs.get("mod_wd1").with_overrides(
        {"max_steps": str(STEPS), "log_every": "10", "device": "cpu"})

    plain, _ = train(config, outdir=None, progress=False)
    probed, _ = train(config, outdir=None, progress=False,
                      observer=RankProbe(dim=256, n_sketch=2, n_probe=64))

    if list(plain.columns) != list(probed.columns) or len(plain) != len(probed):
        raise SystemExit(f"FAIL shape: {plain.shape} vs {probed.shape}")

    worst = 0.0
    for col in plain.columns:
        a, b = plain[col].to_numpy(), probed[col].to_numpy()
        if not np.array_equal(a, b):
            d = float(np.nanmax(np.abs(a - b)))
            worst = max(worst, d)
            print(f"  DIFFERS {col}: max |delta| = {d:.3e}")
    if worst == 0.0:
        print(f"PASS  {len(plain)} rows x {len(plain.columns)} columns bit-identical "
              f"over {STEPS} steps")
        return 0
    print(f"FAIL  largest difference {worst:.3e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

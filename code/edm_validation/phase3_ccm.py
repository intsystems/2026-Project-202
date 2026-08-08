"""Phase 3: can cross mapping recover a known driver from a 1-D training log?

The question the project actually needs answered, on data where the answer is known. Each
run logs `poison_fraction` -- the driver that was injected -- next to `train_loss`. So for
every run we can ask "does the loss reveal the driver?" and check the answer.

Detection requires *both* conditions (see `ccm.py`): beating an IAAFT surrogate null, and
rho increasing with library size. Beating the null alone is what a shared trend does.

Ground truth:

* **must detect** -- LogisticMap (chaotic), Sinusoidal, Discrete, StochSquare (structured)
* **must not**   -- Random, ProgressiveNoise (stochastic drivers), and every Ghost run,
  whose "driver" is a constant or i.i.d. noise and was never coupled to training at all.

The ghost runs give the empirical false-positive rate, which is the only calibration that
counts: it is measured on these logs rather than assumed from theory.

    python phase3_ccm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccm import ccm_test                                    # noqa: E402

CODE = Path(__file__).resolve().parent.parent
POISONED = CODE / "poisoned_batch"

E, TAU, N_SURROGATES = 3, 1, 39

# Ground truth is *coupling*, not determinism -- CCM tests whether one series causally
# drives another, and a stochastic driver drives just as surely as a chaotic one. The
# first version of this table marked `Random` and `ProgressiveNoise` as negatives on the
# grounds that they are not low-dimensional, which asks a different question. Reading
# `batch_poisoning.py` settles it: every real poisoner alters the labels, while every
# Ghost poisoner returns `images, labels` untouched and merely logs a fake number. The
# ghosts are therefore the only genuine negatives -- and they are ideal ones, since the
# logged series exists but was never coupled to anything.
TRUTH = {
    "LogisticMap": True, "Sinusoidal": True, "Discrete": True,
    "Random": True, "ProgressiveNoise": True,
    "Ghost_Const_0.0": False, "Ghost_Const_0.1": False,
    "Ghost_Normal": False, "Ghost_Uniform": False,
}

# Libraries start far below the manifold's sampling scale so that convergence is
# observable at all. With the first attempt's smallest library (5% of 7840 points) the
# LogisticMap curve was already saturated at rho=0.89, so "gain" measured nothing.
LIBRARY_SIZES = (20, 40, 80, 160, 320, 640, 1600, 4000, 7800)


def runs():
    for path in sorted((POISONED / "folder_for_raw_series").glob("*.csv")):
        yield path.stem.replace("resnet_cifar_", "").replace("_logs", ""), pd.read_csv(path)
    for path in sorted((POISONED / "ghost_raw_series_logs").glob("*.csv")):
        yield path.stem.replace("resnet_cifar_", "").replace("_logs", ""), pd.read_csv(path)


def main():
    rows = []
    for name, frame in runs():
        if "poison_fraction" not in frame.columns:
            continue
        driver = frame["poison_fraction"].to_numpy(dtype=float)
        if driver.std() == 0:
            print(f"  {name:<18} driver is constant -- no variation to recover, skipped")
            rows.append({"run": name, "expected": TRUTH.get(name), "rho_raw": np.nan,
                         "gain_raw": np.nan, "p_iaaft": np.nan, "p_shift": np.nan,
                         "rho_differenced": np.nan, "p_differenced": np.nan,
                         "detected": False, "note": "constant driver"})
            continue

        response = frame["train_loss"].to_numpy(dtype=float)
        # Embed the *differenced* response -- Phase 2 showed the training trend dominates
        # the embedding geometry and survives into the surrogates -- but predict the
        # driver in its own units. Differencing the driver too (the first attempt) asks
        # the loss to recover the driver's *increments*, which for a smooth driver are
        # small and quantization-dominated; that alone cost the Sinusoidal detection.
        # Both preprocessings, because the contrast is itself the finding. Differencing
        # was the obvious way to kill the training trend -- but it is a high-pass filter,
        # so it erases precisely the *slow* drivers. Discrete and Sinusoidal have the
        # strongest direct coupling of any run (|corr| 0.58 and 0.69) and yet cross-map at
        # rho 0.12 and 0.09 once differenced; on the raw series they recover to 0.80 and
        # 0.81. The trend does not need removing by hand: the ghost runs share the same
        # training trend and cross-map at rho ~ 0, and IAAFT surrogates preserve it too,
        # so both nulls already control for it.
        variants = {"raw": (response, driver),
                    "differenced": (np.diff(response), driver[1:])}
        outcome = {}
        for kind in ("iaaft", "shift"):
            embed, target = variants["raw"]
            outcome[kind] = ccm_test(embed, target, E=E, tau=TAU,
                                     library_sizes=LIBRARY_SIZES, surrogate_kind=kind,
                                     n_surrogates=N_SURROGATES, seed=0)
        embed, target = variants["differenced"]
        differenced = ccm_test(embed, target, E=E, tau=TAU, library_sizes=LIBRARY_SIZES,
                               surrogate_kind="iaaft", n_surrogates=N_SURROGATES, seed=0)
        # Either null may reject: IAAFT is the strict test of "encodes this realization
        # rather than this spectrum", while the shift null is the one that has power
        # against a periodic driver, whose IAAFT surrogates are the same waveform.
        detected = outcome["iaaft"]["detected"] or outcome["shift"]["detected"]
        expected = TRUTH.get(name)
        ok = (detected == expected) if expected is not None else None

        rows.append({"run": name, "expected": expected, "detected": detected,
                     "rho_raw": outcome["iaaft"]["rho_max"],
                     "gain_raw": outcome["iaaft"]["gain"],
                     "p_iaaft": outcome["iaaft"]["p"], "p_shift": outcome["shift"]["p"],
                     "rho_differenced": differenced["rho_max"],
                     "p_differenced": differenced["p"], "note": ""})

        curve = " ".join(f"{v:.2f}" for v in outcome["iaaft"]["curve"].values())
        print(f"  {name:<18} raw rho={outcome['iaaft']['rho_max']:+.3f} "
              f"p_iaaft={outcome['iaaft']['p']:.3f} p_shift={outcome['shift']['p']:.3f} "
              f"| diff rho={differenced['rho_max']:+.3f} "
              f"-> {'DETECT' if detected else 'none  '} "
              f"{'OK' if ok else 'MISMATCH' if ok is False else ''}   [{curve}]",
              flush=True)

    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "phase3_ccm.csv", index=False)

    judged = table.dropna(subset=["expected"])
    tp = int(((judged.detected) & (judged.expected)).sum())
    fp = int(((judged.detected) & (~judged.expected.astype(bool))).sum())
    fn = int(((~judged.detected) & (judged.expected)).sum())
    tn = int(((~judged.detected) & (~judged.expected.astype(bool))).sum())

    print(f"\n  true positives  {tp}    false negatives {fn}")
    print(f"  false positives {fp}    true negatives  {tn}")
    print(f"\n  false-positive rate on drivers that were never coupled: "
          f"{fp}/{fp + tn}" if (fp + tn) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

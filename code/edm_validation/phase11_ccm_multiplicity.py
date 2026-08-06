"""Audit the multiplicity of the CCM detection rule, which is the load-bearing result.

`ccm.py` declares a detection when *either* of two surrogate nulls rejects at 0.05:

    detected = iaaft["detected"] or shift["detected"]

Two nulls at 0.05, combined by OR, admit a family-wise false-positive rate approaching
0.10 rather than the nominal 0.05. That inflation was never stated and never corrected.

Worse, it cannot be corrected at the ensemble size used. The rank p-value is
(n_not_beaten + 1) / (n + 1), so with 39 surrogates the smallest attainable value is
1/40 = 0.025, which is exactly the Bonferroni threshold for two tests. The corrected test
could therefore only ever reject by the narrowest possible margin, and one detection in
the published table (Sinusoidal, p_iaaft = 0.050) sits on the wrong side of it.

This script re-runs both experiments with 199 surrogates, where the attainable floor is
1/200 = 0.005 and a corrected threshold of 0.025 is a real test rather than a knife edge.
It reports the outcome under both rules so the effect of the correction is explicit.

    python phase11_ccm_multiplicity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ccm import ccm_test                                     # noqa: E402

CODE = HERE.parent
POISONED = CODE / "poisoned_batch"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

E, TAU = 3, 1
N_SURROGATES = 199
LIBRARY_SIZES = (20, 40, 80, 160, 320, 640, 1600, 4000, 7800)
ALPHA = 0.05                       # the original per-test level
ALPHA_CORRECTED = ALPHA / 2        # Bonferroni over the two nulls

PHASE3 = {
    "LogisticMap": True, "Sinusoidal": True, "Discrete": True,
    "Random": True, "ProgressiveNoise": True,
    "Ghost_Const_0.1": False, "Ghost_Normal": False, "Ghost_Uniform": False,
}
PHASE5 = {"logistic": True, "sinusoid": True, "iid": True, "ghost": False}
PERIODIC = {"Sinusoidal", "sinusoid"}


def phase3_series(name):
    folder = "ghost_raw_series_logs" if name.startswith("Ghost") else "folder_for_raw_series"
    frame = pd.read_csv(POISONED / folder / f"resnet_cifar_{name}_logs.csv")
    return (frame["train_loss"].to_numpy(dtype=float),
            frame["poison_fraction"].to_numpy(dtype=float))


def phase5_series(name):
    frame = pd.read_csv(OUT / "inject" / f"inject_{name}.csv")
    return (frame["train_loss"].to_numpy(dtype=float),
            frame["driver"].to_numpy(dtype=float))


def evaluate(response, driver, library_sizes):
    out = {}
    for kind in ("iaaft", "shift"):
        out[kind] = ccm_test(response, driver, E=E, tau=TAU, library_sizes=library_sizes,
                             surrogate_kind=kind, n_surrogates=N_SURROGATES, seed=0)
    return out


def main():
    print(f"Re-running both CCM experiments with {N_SURROGATES} surrogates "
          f"(p floor {1 / (N_SURROGATES + 1):.4f})\n")

    rows = []
    experiments = [("phase3 / ResNet-18", PHASE3, phase3_series, LIBRARY_SIZES),
                   ("phase5 / transformer", PHASE5, phase5_series,
                    tuple(L for L in LIBRARY_SIZES if L <= 1600))]

    for title, truth, loader, sizes in experiments:
        print(f"=== {title} ===")
        for name, coupled in truth.items():
            response, driver = loader(name)
            if np.std(driver) == 0:
                print(f"  {name:<18} constant driver, skipped")
                continue
            out = evaluate(response, driver, sizes)
            p_i, p_s = out["iaaft"]["p"], out["shift"]["p"]
            rho = out["iaaft"]["rho_max"]
            effect = np.isfinite(rho) and rho > 0.05
            loose = effect and (p_i <= ALPHA or p_s <= ALPHA)
            strict = effect and (p_i <= ALPHA_CORRECTED or p_s <= ALPHA_CORRECTED)
            rows.append({"experiment": title, "run": name, "coupled": coupled,
                         "rho": rho, "p_iaaft": p_i, "p_shift": p_s,
                         "detected_uncorrected": loose, "detected_corrected": strict,
                         "periodic": name in PERIODIC})
            flag = "" if loose == strict else "   <- changes under correction"
            print(f"  {name:<18} rho={rho:+.3f} p_iaaft={p_i:.4f} p_shift={p_s:.4f}"
                  f"  uncorrected={'yes' if loose else 'no ':<3}"
                  f" corrected={'yes' if strict else 'no '}{flag}", flush=True)
        print()

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase11_ccm_multiplicity.csv", index=False)

    print("=== Confusion under each rule ===")
    for column, label in (("detected_uncorrected", "OR of two tests at 0.05 (as published)"),
                          ("detected_corrected", "Bonferroni, 0.025 per test")):
        tp = int((table[column] & table.coupled).sum())
        fp = int((table[column] & ~table.coupled).sum())
        fn = int((~table[column] & table.coupled).sum())
        tn = int((~table[column] & ~table.coupled).sum())
        print(f"  {label:<40} TP={tp} FN={fn} FP={fp} TN={tn}")

    print("\n=== Where the periodic limitation sits ===")
    for column in ("detected_uncorrected", "detected_corrected"):
        aper = table[table.coupled & ~table.periodic]
        per = table[table.coupled & table.periodic]
        print(f"  {column:<24} aperiodic {int(aper[column].sum())}/{len(aper)}"
              f"   periodic {int(per[column].sum())}/{len(per)}")
    print("\n  A rule that detects every aperiodic driver and no periodic one is a")
    print("  cleaner statement of the same limitation than a split verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

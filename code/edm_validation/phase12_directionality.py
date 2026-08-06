"""Test that cross mapping recovers the direction of coupling, not merely its presence.

Sections 4 of the report claim a causal recovery. Cross mapping earns that word only if it
distinguishes "driver forces loss" from "loss forces driver", and the reverse map had never
been run. This script runs it.

Convention (`ccm.py`): embedding Y and predicting X, written "Y xmap X", is evidence that
**X drives Y**. So

    forward   loss  xmap driver     evidence that the driver forces the loss
    reverse   driver xmap loss      evidence that the loss forces the driver

These logs are an unusually clean test bed for this, because the ground truth is known by
construction rather than assumed: `poison_fraction` is a schedule evaluated from a fixed
seed before any gradient is taken, so training cannot influence it. The coupling is
unidirectional, and the reverse map must therefore fail. If it does not, the word "causal"
is unsupported and the claim reduces to detection of a shared signal.

The reverse direction is given the better of two advantages so that a failure cannot be
blamed on a handicapped embedding: its embedding dimension is scanned over E = 2..6 and the
best result kept, while the forward direction stays at the published E = 3.

    python phase12_directionality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ccm import ccm_test, ccm_convergence                    # noqa: E402

CODE = HERE.parent
POISONED = CODE / "poisoned_batch"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TAU, N_SURROGATES = 1, 99          # p floor 0.01, so a 0.025 threshold is expressible
E_FORWARD, E_REVERSE_SCAN = 3, (2, 3, 4, 5, 6)
BIG = (20, 40, 80, 160, 320, 640, 1600, 4000, 7800)
SMALL = (20, 40, 80, 160, 320, 640, 1600)

PHASE3 = {"LogisticMap": True, "Sinusoidal": True, "Discrete": True,
          "Random": True, "ProgressiveNoise": True,
          "Ghost_Normal": False, "Ghost_Uniform": False}
PHASE5 = {"logistic": True, "sinusoid": True, "iid": True, "ghost": False}


def phase3_series(name):
    folder = "ghost_raw_series_logs" if name.startswith("Ghost") else "folder_for_raw_series"
    frame = pd.read_csv(POISONED / folder / f"resnet_cifar_{name}_logs.csv")
    return (frame["train_loss"].to_numpy(dtype=float),
            frame["poison_fraction"].to_numpy(dtype=float))


def phase5_series(name):
    frame = pd.read_csv(OUT / "inject" / f"inject_{name}.csv")
    return (frame["train_loss"].to_numpy(dtype=float),
            frame["driver"].to_numpy(dtype=float))


def best_reverse(driver, response, sizes):
    """Strongest reverse result over a scan of E, so the test is not a straw man.

    The scan itself uses convergence curves only. Running a full surrogate ensemble at
    every E costs about fifteen minutes per run and buys nothing, because the ensemble is
    needed only to judge the E the scan selects.
    """
    best_E, best_rho = None, -np.inf
    for E in E_REVERSE_SCAN:
        curve = ccm_convergence(driver, response, E=E, tau=TAU, library_sizes=sizes)
        finite = [v for v in (curve[L] for L in sorted(curve)) if np.isfinite(v)]
        if finite and finite[-1] > best_rho:
            best_E, best_rho = E, finite[-1]
    out = ccm_test(driver, response, E=best_E, tau=TAU, library_sizes=sizes,
                   surrogate_kind="iaaft", n_surrogates=N_SURROGATES, seed=0)
    return best_E, out


def main():
    rows = []
    for title, truth, loader, sizes in (
            ("ResNet-18 / CIFAR-10", PHASE3, phase3_series, BIG),
            ("transformer / modular addition", PHASE5, phase5_series, SMALL)):
        print(f"=== {title} ===")
        print(f"  {'run':<18} {'fwd rho':>8} {'fwd gain':>9} {'fwd p':>7} | "
              f"{'rev rho':>8} {'rev gain':>9} {'rev p':>7}  {'E':>2}  verdict")
        for name, coupled in truth.items():
            response, driver = loader(name)
            if np.std(driver) == 0:
                continue

            fwd = ccm_test(response, driver, E=E_FORWARD, tau=TAU, library_sizes=sizes,
                           surrogate_kind="iaaft", n_surrogates=N_SURROGATES, seed=0)
            e_rev, rev = best_reverse(driver, response, sizes)

            # A direction is supported when it beats its own surrogate null and its skill
            # grows with library size. Absolute rho is not comparable across directions,
            # because predicting a smooth driver from a rough loss is not the same task as
            # the converse; convergence and the null are.
            fwd_ok = fwd["p"] <= 0.025 and fwd["rho_max"] > 0.05 and fwd["gain"] > 0
            rev_ok = rev["p"] <= 0.025 and rev["rho_max"] > 0.05 and rev["gain"] > 0
            verdict = ("driver -> loss" if fwd_ok and not rev_ok else
                       "loss -> driver" if rev_ok and not fwd_ok else
                       "bidirectional" if fwd_ok else "none")
            rows.append({"experiment": title, "run": name, "coupled": coupled,
                         "rho_fwd": fwd["rho_max"], "gain_fwd": fwd["gain"],
                         "p_fwd": fwd["p"], "rho_rev": rev["rho_max"],
                         "gain_rev": rev["gain"], "p_rev": rev["p"],
                         "E_rev": e_rev, "verdict": verdict,
                         "asymmetry": fwd["rho_max"] - rev["rho_max"]})
            print(f"  {name:<18} {fwd['rho_max']:>8.3f} {fwd['gain']:>9.3f} "
                  f"{fwd['p']:>7.4f} | {rev['rho_max']:>8.3f} {rev['gain']:>9.3f} "
                  f"{rev['p']:>7.4f}  {e_rev:>2}  {verdict}", flush=True)
        print()

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase12_directionality.csv", index=False)

    coupled = table[table.coupled]
    ghosts = table[~table.coupled]
    correct = int((coupled.verdict == "driver -> loss").sum())
    print("=== Summary ===")
    print(f"  coupled runs assigned the correct direction: {correct}/{len(coupled)}")
    print(f"  coupled runs called bidirectional:           "
          f"{int((coupled.verdict == 'bidirectional').sum())}/{len(coupled)}")
    print(f"  coupled runs assigned the reverse direction: "
          f"{int((coupled.verdict == 'loss -> driver').sum())}/{len(coupled)}")
    print(f"  ghost runs with any direction detected:      "
          f"{int((ghosts.verdict != 'none').sum())}/{len(ghosts)}")
    print(f"\n  forward skill exceeds reverse in {int((coupled.asymmetry > 0).sum())}"
          f"/{len(coupled)} coupled runs")
    print("  Ground truth is unidirectional by construction: the driver schedule is")
    print("  evaluated from a fixed seed before training and cannot depend on the loss.")

    # Convergence curves for the figure: the reverse direction should stay flat.
    curves = []
    for name in ("LogisticMap", "Random", "Ghost_Normal"):
        response, driver = phase3_series(name)
        f = ccm_convergence(response, driver, E=E_FORWARD, tau=TAU, library_sizes=BIG)
        r = ccm_convergence(driver, response, E=E_FORWARD, tau=TAU, library_sizes=BIG)
        for L in BIG:
            curves.append({"run": name, "library": L, "forward": f[L], "reverse": r[L]})
    pd.DataFrame(curves).to_csv(OUT / "phase12_direction_curves.csv", index=False)
    print("\n  wrote phase12_directionality.csv and phase12_direction_curves.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

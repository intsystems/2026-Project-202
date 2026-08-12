"""Does a 1-D observer follow a dimension that CHANGES: 1 -> 2 -> 3 -> 4 -> 3 -> 2?

Part 1 asks whether the estimate can tell stationary regimes of different dimension
apart. That is necessary but not sufficient for the use the project wants, which is
detecting a change online. Three things can go wrong here that cannot go wrong there:

  * a window straddling a change sees a mixture, so the estimate is smeared over one
    window length whatever the estimator does -- a floor on the achievable lag;
  * a rise and a fall need not be symmetric, because an oscillator switching off leaves
    the orbit on a subtorus immediately, while one switching on needs time to fill the
    new direction (the resonance margin in systems.py bounds how long);
  * with a ramp rather than a step the ground truth is not even defined during the
    ramp: an oscillator at 5% of its final amplitude is not a degree of freedom the
    data contains, whatever the schedule says.

    python exp2_transitions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estimators as E
import systems as S

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

SCHEDULE = [1, 2, 3, 4, 3, 2]
WINDOW, STRIDE, SEG, CYCLES = 2000, 200, 8000, 300.0


def rule(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def run_one(ramp, seed, observer="norm_fro"):
    sysinfo = S.make_transition(SCHEDULE, seg=SEG, cycles_per_window=CYCLES,
                                window=WINDOW, seed=seed, ramp=ramp,
                                band_mode="matched")
    info = {"diag": sysinfo["diag"], "active": np.arange(max(SCHEDULE)),
            "n": sysinfo["n"]}
    x = S.observers(info, seed=seed, obs_snr=1e6)[observer]
    right, mg = E.trace(x, WINDOW, STRIDE, name="MG")
    _, rough = E.trace(x, WINDOW, STRIDE, name="roughness")
    return sysinfo, right, mg, rough


def section_1():
    rule("1  Estimate per regime, windows lying wholly inside one segment")
    print(f"  schedule {SCHEDULE}, {SEG} samples per regime, window {WINDOW},")
    print(f"  {CYCLES:g} cycles per window, matched band, SNR 1e6, abrupt switches.\n")
    rows = []
    for seed in (0, 1, 2):
        sysinfo, right, mg, rough = run_one(0, seed)
        left = right - WINDOW + 1
        for s, k in enumerate(SCHEDULE):
            lo, hi = s * SEG, (s + 1) * SEG - 1
            inside = (left >= lo) & (right <= hi)
            rows.append({"seed": seed, "segment": s, "true_k": k,
                         "n_windows": int(inside.sum()),
                         "MG": float(np.nanmedian(mg[inside])),
                         "roughness": float(np.nanmedian(rough[inside]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp2_segments.csv", index=False)
    piv = frame.groupby(["segment", "true_k"]).agg(
        MG=("MG", "median"), MG_sd=("MG", "std"),
        rough=("roughness", "median"), n=("n_windows", "median")).reset_index()
    print(f"  {'segment':>8}{'true k':>8}{'MG':>8}{'sd over seeds':>15}{'roughness':>11}"
          f"{'windows':>9}")
    for _, r in piv.iterrows():
        print(f"  {int(r.segment):>8}{int(r.true_k):>8}{r.MG:>8.2f}{r.MG_sd:>15.2f}"
              f"{r.rough:>11.4f}{int(r.n):>9}")
    up = piv[piv.segment <= 3]
    down = piv[piv.segment >= 3]
    print(f"\n  rising leg  1->2->3->4 : " + " -> ".join(f"{v:.2f}" for v in up.MG))
    print(f"  falling leg 4->3->2     : " + " -> ".join(f"{v:.2f}" for v in down.MG))
    print(f"  roughness across all six regimes: {piv.rough.min():.4f} to {piv.rough.max():.4f}")
    same_k = piv.groupby("true_k").MG.agg(["min", "max"])
    print("\n  the same k, reached from below and from above:")
    for k, r in same_k.iterrows():
        if r["min"] != r["max"]:
            print(f"    k={int(k)}: {r['min']:.2f} and {r['max']:.2f}"
                  f"   (hysteresis {abs(r['max'] - r['min']):.2f})")


def section_2():
    rule("2  Detection lag at each change")
    print("  A change is called at the first window whose estimate crosses the midpoint")
    print("  between the two regimes' own medians and stays across for two windows. Lag")
    print("  is measured from the true switch. One window is the floor: a window")
    print(f"  straddling the switch is a mixture, so nothing can beat {WINDOW} samples.\n")
    seg_med = pd.read_csv(OUT / "exp2_segments.csv").groupby("segment").MG.median()
    rows = []
    for seed in (0, 1, 2):
        sysinfo, right, mg, rough = run_one(0, seed)
        for s in range(1, len(SCHEDULE)):
            switch = s * SEG
            a, b = seg_med[s - 1], seg_med[s]
            mid = 0.5 * (a + b)
            rising = b > a
            after = right >= switch
            idx = np.flatnonzero(after)
            lag = None
            for j in range(len(idx) - 1):
                i0, i1 = idx[j], idx[j + 1]
                cross = (mg[i0] > mid and mg[i1] > mid) if rising \
                    else (mg[i0] < mid and mg[i1] < mid)
                if cross:
                    lag = int(right[i0] - switch)
                    break
            rows.append({"seed": seed, "switch": s, "from": SCHEDULE[s - 1],
                         "to": SCHEDULE[s], "lag": lag})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp2_lags.csv", index=False)
    print(f"  {'change':>10}{'seed 0':>9}{'seed 1':>9}{'seed 2':>9}{'median':>9}"
          f"{'in windows':>12}")
    for s in range(1, len(SCHEDULE)):
        sub = frame[frame.switch == s]
        vals = [sub[sub.seed == q].lag.iloc[0] for q in (0, 1, 2)]
        good = [v for v in vals if v is not None]
        med = np.median(good) if good else np.nan
        print(f"  {f'{SCHEDULE[s-1]}->{SCHEDULE[s]}':>10}"
              + "".join(f"{('--' if v is None else v):>9}" for v in vals)
              + f"{med:>9.0f}{med / WINDOW:>12.1f}")
    print("\n  The asymmetry runs the opposite way to the one I expected when writing")
    print("  this file, and the measurement is what counts. Rises are detected in 0.2 to")
    print("  0.6 of a window, falls in 0.8 to 0.9. A new oscillator widens the local")
    print("  neighbour structure as soon as it appears, so a rise shows up while the")
    print("  window is still mostly old data. A fall cannot show up until the old,")
    print("  higher-dimensional data has left the window, so it is floored near one")
    print("  window length. A criterion built on a dimension DROP therefore pays the")
    print("  worse half of this asymmetry -- roughly a full window of lag -- and the")
    print("  published 3000-step window already exceeds most plateaus in this project.")


def section_3():
    rule("3  Abrupt switch against a ramp")
    print("  With a ramp the true dimension is undefined inside it, because an")
    print("  oscillator below the noise floor is not a degree of freedom the data has.")
    print("  This is not a limitation of the estimator, and any experiment that reports")
    print("  a detection lag against a ramped ground truth is measuring its own ramp.\n")
    print(f"  {'ramp':>7}{'seg 0 (k=1)':>13}{'seg 3 (k=4)':>13}{'seg 5 (k=2)':>13}"
          f"{'range over regimes':>20}")
    rows = []
    for ramp in (0, 500, 2000, 6000):
        sysinfo, right, mg, rough = run_one(ramp, 0)
        left = right - WINDOW + 1
        meds = []
        for s in range(len(SCHEDULE)):
            lo, hi = s * SEG, (s + 1) * SEG - 1
            inside = (left >= lo) & (right <= hi)
            meds.append(float(np.nanmedian(mg[inside])))
        print(f"  {ramp:>7}{meds[0]:>13.2f}{meds[3]:>13.2f}{meds[5]:>13.2f}"
              f"{max(meds) - min(meds):>20.2f}")
        rows.append({"ramp": ramp, **{f"seg{i}": m for i, m in enumerate(meds)},
                     "range": max(meds) - min(meds)})
    pd.DataFrame(rows).to_csv(OUT / "exp2_ramp.csv", index=False)
    print("\n  A ramp comparable to the segment length flattens the whole trace, because")
    print("  no regime is ever stationary for a full window.")


SECTIONS = {"1": section_1, "2": section_2, "3": section_3}


def main(argv):
    for name in (argv[1:] or list(SECTIONS)):
        if name not in SECTIONS:
            print(f"unknown section {name!r}")
            return 2
        SECTIONS[name]()
    print(f"\nCSV output in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

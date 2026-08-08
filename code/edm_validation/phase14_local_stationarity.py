"""Does dense logging make a locally stationary analysis possible?

Report section 3 measures *global* recurrence and finds it absent. That rules out treating
a whole run as one invariant set, but it does not settle the weaker and more useful
question: whether the dynamics are locally stationary, so that a short window can be
analysed as a stationary system in its own right.

At one row per ten optimisation steps the question could not even be posed. A window short
enough to be locally stationary holds too few points to embed, and a window with enough
points spans more than the whole transition. With one row per step the two constraints
separate, and this script measures whether they actually do.

Three measurements, all on the dense logs from phase 13:

1. **How long is a window locally stationary?** For a window of L steps, compare the shift
   of the mean across the window against the fluctuations inside it, and express that in
   units of what white noise of the same length gives. The normalisation matters: for a
   stationary series the raw ratio falls like 1/sqrt(L), so a fixed threshold grows more
   permissive with length and would report long windows as *more* stationary than short
   ones by arithmetic alone.

2. **How many points does such a window hold?** This is the quantity dense logging changes,
   and it is the whole argument for doing it.

3. **Does a locally stationary window recur?** Stationarity is necessary but not
   sufficient: a slowly drifting smooth arc is locally stationary and still never returns
   near its past states. Recurrence is measured inside the stationary scale, at matched
   *physical* exclusion, for the dense series and for the same series decimated to the old
   rate, so the comparison isolates sampling from dynamics.

    python phase14_local_stationarity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecast import recurrence_profile                      # noqa: E402


def lorenz_x(n, dt=0.01, s=10.0, r=28.0, b=8 / 3, burn=1000):
    state, out = np.array([1.0, 1.0, 1.0]), []
    for _ in range(n + burn):
        x, y, z = state
        state = state + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        out.append(state[0])
    return np.array(out[burn:])

DENSE = HERE / "results" / "dense"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

RUNS = {"grok_dense": "generalises", "lowdata15_dense": "never generalises",
        "wd0_dense": "never generalises"}
OBSERVABLES = ("train_loss", "weight_norm")
LENGTHS = (25, 50, 100, 200, 400, 800, 1600, 3000)


def drift_ratio(series, length, stride=None):
    """Median |shift of the mean across a window| / |fluctuation inside it|."""
    stride = stride or max(length // 2, 1)
    ratios = []
    for start in range(0, len(series) - length + 1, stride):
        w = series[start:start + length]
        half = length // 2
        shift = abs(np.mean(w[half:]) - np.mean(w[:half]))
        detrended = w - np.polyval(np.polyfit(np.arange(length), w, 1), np.arange(length))
        scale = np.std(detrended)
        if scale > 0:
            ratios.append(shift / scale)
    return float(np.median(ratios)) if ratios else float("nan")


def stationarity_index(series, length, reference, stride=None):
    """Drift ratio in units of what a stationary series of the same length would give.

    The raw ratio is not scale free. For a stationary series the shift of the mean across
    a window is a sampling fluctuation of order ``2 / sqrt(length)`` relative to the
    within-window spread, so the raw ratio *falls* as the window grows and a fixed
    threshold such as "below 1" silently becomes more permissive at longer windows. That
    would have made long windows look more stationary than short ones purely by
    arithmetic. Dividing by the value measured on white noise of the same length removes
    the length dependence: an index near 1 means indistinguishable from stationary, and a
    large index means drift dominates.
    """
    raw = drift_ratio(series, length, stride)
    return raw / reference if reference else float("nan")


def recurrence_at(series, exclusion_steps, interval, E=5, tau_steps=5):
    """Recurrence rate at a physical exclusion, expressed in optimisation steps.

    Both tau and the exclusion window are converted from steps into samples, so a dense
    series and a decimated one are asked the same question about the same dynamics.
    """
    tau = max(tau_steps // interval, 1)
    windows = [0, max(exclusion_steps // interval, 1)]
    profile, _ = recurrence_profile(series, E=E, tau=tau, windows=windows)
    if not profile or len(profile) < 2:
        return float("nan")
    keys = sorted(profile)
    base = profile[keys[0]]
    return profile[keys[-1]] / base if base else float("nan")


def main():
    # Calibrate the drift ratio on a series that is stationary by construction, so the
    # criterion is not a bare threshold on a length-dependent quantity.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=6000)
    reference = {L: drift_ratio(noise, L) for L in LENGTHS}

    print("=== 0. Calibration of the stationarity criterion ===")
    print("    white noise, stationary by construction, sets the scale at each length\n")
    print(f"    {'series':<28} " + " ".join(f"{L:>6}" for L in LENGTHS))
    print(f"    {'white noise (raw ratio)':<28} "
          + " ".join(f"{reference[L]:6.2f}" for L in LENGTHS))
    ramp = np.linspace(0.0, 1.0, 6000) ** 2
    print(f"    {'monotone ramp (index)':<28} "
          + " ".join(f"{stationarity_index(ramp, L, reference[L]):6.1f}" for L in LENGTHS))

    print("\n=== 1. Over what window length are the dynamics locally stationary? ===")
    print("    stationarity index: 1 means indistinguishable from stationary,")
    print("    large means drift dominates\n")
    rows = []
    print(f"    {'run':<16} {'observable':<12} " + " ".join(f"{L:>6}" for L in LENGTHS))
    for run in RUNS:
        frame = pd.read_csv(DENSE / f"{run}.csv")
        for observable in OBSERVABLES:
            series = frame[observable].to_numpy()
            index = [stationarity_index(series, L, reference[L]) for L in LENGTHS]
            for L, value in zip(LENGTHS, index):
                rows.append({"run": run, "observable": observable, "window_steps": L,
                             "drift_ratio": drift_ratio(series, L),
                             "stationarity_index": value})
            print(f"    {run:<16} {observable:<12} "
                  + " ".join(f"{v:6.1f}" for v in index))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase14_drift_ratio.csv", index=False)

    print("\n=== 2. How many points does a locally stationary window hold? ===")
    print("    (taking an index below 3 as locally stationary)\n")
    for run in RUNS:
        for observable in OBSERVABLES:
            sub = table[(table.run == run) & (table.observable == observable)]
            ok = sub[sub.stationarity_index < 3.0]
            if len(ok):
                longest = int(ok.window_steps.max())
                print(f"    {run:<16} {observable:<12} stationary up to {longest:>5} steps"
                      f"  ->{longest:>5} points dense, {longest // 10:>4} at the old rate")
            else:
                print(f"    {run:<16} {observable:<12} never stationary at any tested length")

    print("\n=== 3. Does a locally stationary window recur? ===")
    print("    Recurrence against exclusion inside one 800-step segment, normalised by")
    print("    the rate at zero exclusion. A plateau means genuine returns; a decay to")
    print("    zero means every close pair was merely adjacent in time.\n")

    # Every exclusion is a multiple of 10 so the decimated series can represent it
    # exactly. With 8 and 16 in the list they collapsed to 1 and 4 samples, the row
    # lost two columns to deduplication, and dense and decimated were compared
    # against different physical exclusions.
    seg_len, excl = 800, (0, 10, 20, 40, 80, 160)

    def profile_of(series, interval=1):
        tau = max(5 // interval, 1)
        windows = sorted({max(e // interval, 0) for e in excl})
        prof, _ = recurrence_profile(series, E=5, tau=tau, windows=windows)
        if not prof:
            return [float("nan")] * len(windows)
        keys = sorted(prof)
        base = prof[keys[0]]
        return [prof[k] / base if base else float("nan") for k in keys]

    print("    " + f"{'series':<36}" + " ".join(f"{e:>6}" for e in excl))
    rec_rows = []
    for label, series in (("Lorenz-63 (attractor)", lorenz_x(seg_len)),
                          ("monotone ramp (transient)", np.linspace(0, 1, seg_len) ** 2)):
        vals = profile_of(series)
        print(f"    {label:<36}" + " ".join(f"{v:6.2f}" for v in vals))
        rec_rows.append({"series": label, "sampling": "reference",
                         **{f"excl_{e}": v for e, v in zip(excl, vals)}})
    print()

    for run in RUNS:
        frame = pd.read_csv(DENSE / f"{run}.csv")
        for observable in OBSERVABLES:
            segment = frame[observable].to_numpy()[2000:2000 + seg_len]
            for tag, series, interval in (("every step", segment, 1),
                                          ("every 10th", segment[::10], 10)):
                vals = profile_of(series, interval)
                name = f"{run.replace('_dense', '')} {observable}, {tag}"
                print(f"    {name:<36}" + " ".join(f"{v:6.2f}" for v in vals))
                rec_rows.append({"series": f"{run} {observable}", "sampling": tag,
                                 **{f"excl_{e}": v for e, v in zip(excl, vals)}})
        print()

    pd.DataFrame(rec_rows).to_csv(OUT / "phase14_local_recurrence.csv", index=False)
    print("    Columns are the exclusion in optimisation steps. The dense and decimated")
    print("    rows ask the same physical question, so a difference between them would")
    print("    be an effect of sampling rather than of the dynamics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate MG on the Frobenius norm for known dimensions k=1..10.

The experiment deliberately separates three questions:

1. Can one *fixed* hyperparameter set numerically recover stationary k=1..10?
2. Does that set transfer to seeds not used for selection?
3. Does it follow a non-monotone, branching schedule of dimension changes?

No separate setting is allowed for each k.  Such a lookup table would make exact
recovery tautological.  The calibration seed is 0; seeds 1 and 2 and the transition
trajectory are held out until the setting has been selected.

Quick run (the version used in report_exp9.md):

    python exp9_frobenius_k10.py

A larger grid can be requested with ``--full``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import estimators as E
import systems as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp9_frobenius_k10"
OUT.mkdir(parents=True, exist_ok=True)

KS = np.arange(1, 11)
CAL_SEED = 0
TEST_SEEDS = (1, 2)
SCHEDULE = (2, 7, 4, 9, 3, 10, 6, 1, 8, 5, 2)


def make_observer(k: int, seed: int, window: int, cycles: float,
                  n: int | None = None, snr: float = 1e6) -> np.ndarray:
    n = int(n or window)
    info = S.make_system(
        "quasiperiodic", k=k, D=64, n=n, cycles_per_window=cycles,
        window=window, amp=0.1, seed=seed, band_mode="matched",
    )
    return S.observers(info, seed=seed, obs_snr=snr)["norm_fro"]


def score(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    err = values - KS
    inversions = int(np.sum(np.diff(values) <= 0))
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "max_error": float(np.max(np.abs(err))),
        "rho": float(spearmanr(KS, values).statistic),
        "inversions": inversions,
        # Selection is numerical, not merely ordinal.  Inversions and a large
        # worst-case miss are penalised so a low average cannot hide saturation.
        "objective": float(np.mean(np.abs(err)) + 0.20 * np.max(np.abs(err))
                           + 0.25 * inversions),
    }


def grid(full: bool):
    if full:
        return itertools.product(
            (2000, 4000, 8000), (100.0, 300.0, 1000.0),
            (21, 25, 31), (5, 10, 20), (1, 2, 4),
        )
    # A compact grid centred on the best regimes found by exp6.  It is large
    # enough to test the important coverage / scale interaction without turning
    # the experiment into a per-k hyperparameter search.
    return itertools.product(
        (2000, 4000), (300.0, 1000.0),
        (25,), (5, 10, 20, 30), (1, 2, 4),
    )


def calibrate(full: bool) -> tuple[pd.DataFrame, dict]:
    rows = []
    cache: dict[tuple[int, float, int], np.ndarray] = {}
    for window, cycles, max_E, kn, tau in grid(full):
        values = []
        for k in KS:
            key = (window, cycles, int(k))
            if key not in cache:
                cache[key] = make_observer(int(k), CAL_SEED, window, cycles)
            values.append(E.mg(cache[key], max_E=max_E, k=kn, tau=tau))
        stats = score(np.asarray(values))
        rows.append({
            "window": window, "cycles": cycles, "max_E": max_E,
            "k_neighbors": kn, "tau": tau, **stats,
            **{f"d{k}": float(values[k - 1]) for k in KS},
        })
    frame = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    frame.to_csv(OUT / "calibration_grid.csv", index=False)
    best = frame.iloc[0].to_dict()
    return frame, best


def validate(best: dict) -> pd.DataFrame:
    cfg = {k: int(best[k]) for k in ("window", "max_E", "k_neighbors", "tau")}
    cfg["cycles"] = float(best["cycles"])
    rows = []
    for seed in (CAL_SEED,) + TEST_SEEDS:
        vals = []
        for k in KS:
            x = make_observer(int(k), seed, cfg["window"], cfg["cycles"])
            vals.append(E.mg(x, max_E=cfg["max_E"], k=cfg["k_neighbors"],
                             tau=cfg["tau"]))
        st = score(np.asarray(vals))
        for k, val in zip(KS, vals):
            rows.append({"seed": seed, "split": "calibration" if seed == 0 else "held_out",
                         "k": int(k), "MG": float(val), "error": float(val-k), **st})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "stationary_validation.csv", index=False)
    return frame


def transition(best: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = int(best["window"])
    # At least three independent estimator windows per stationary segment; the
    # order deliberately zigzags and revisits k=2 from unrelated histories.
    seg = 3 * window
    # The transition trace is a held-out validation, not a dense change-point
    # benchmark.  Half-window sampling supplies five stationary readings per
    # segment while keeping the kNN calculation practical up to k=10.
    stride = max(50, window // 2)
    info = S.make_transition(
        SCHEDULE, D=64, seg=seg, cycles_per_window=float(best["cycles"]),
        window=window, amp=0.1, snr=np.inf, seed=101, ramp=0,
        band_mode="matched",
    )
    obs_info = {"diag": info["diag"], "active": np.arange(max(SCHEDULE)),
                "n": info["n"]}
    x = S.observers(obs_info, seed=101, obs_snr=1e6)["norm_fro"]
    right, mg = E.trace(
        x, window, stride, name="MG", max_E=int(best["max_E"]),
        k=int(best["k_neighbors"]), tau=int(best["tau"]),
    )
    trace = pd.DataFrame({"right": right, "MG": mg,
                          "true_k": info["truth"][right]})
    trace.to_csv(OUT / "zigzag_trace.csv", index=False)

    left = right - window + 1
    rows = []
    for j, k in enumerate(SCHEDULE):
        lo, hi = j * seg, (j + 1) * seg - 1
        inside = (left >= lo) & (right <= hi)
        val = float(np.nanmedian(mg[inside]))
        rows.append({"segment": j, "true_k": k, "MG": val,
                     "error": val-k, "n_windows": int(inside.sum())})
    segments = pd.DataFrame(rows)
    segments.to_csv(OUT / "zigzag_segments.csv", index=False)
    return trace, segments


def figures(validation: pd.DataFrame, trace: pd.DataFrame, segments: pd.DataFrame,
            best: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.plot(KS, KS, "--", color="black", lw=1.2, label="ideal: estimate = k")
    for seed, sub in validation.groupby("seed"):
        ax.plot(sub.k, sub.MG, marker="o", lw=1.5,
                label=f"seed {seed}" + (" (calibration)" if seed == 0 else " (held out)"))
    ax.set(xlabel="True dimension k", ylabel="MG estimate from Frobenius norm",
           title="Stationary diagonal systems: absolute recovery")
    ax.set_xticks(KS); ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUT / "stationary_k1_k10.png", dpi=220)
    fig.savefig(OUT / "stationary_k1_k10.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.0, 5.5), constrained_layout=True)
    scale = int(best["window"])
    ax.step(trace.right / scale, trace.true_k, where="post", color="black", lw=1.3,
            label="true k")
    ax.plot(trace.right / scale, trace.MG, color="#28669b", lw=1.2,
            label="MG(norm_fro)")
    ax.scatter(((segments.segment + 1) * 3 - .5), segments.MG,
               color="#c33c54", s=26, zorder=3, label="within-segment median")
    ax.set(xlabel="Time / estimator-window length", ylabel="Dimension",
           title="Held-out zigzag schedule: " + " → ".join(map(str, SCHEDULE)))
    ax.grid(alpha=.22); ax.legend(ncol=3)
    fig.savefig(OUT / "zigzag_transition.png", dpi=220)
    fig.savefig(OUT / "zigzag_transition.pdf")
    plt.close(fig)


def write_report(grid_frame: pd.DataFrame, best: dict, val: pd.DataFrame,
                 segments: pd.DataFrame, elapsed: float) -> None:
    summaries = val.groupby(["seed", "split"], as_index=False).first()
    cfg = (f"window={int(best['window'])}, cycles/window={best['cycles']:g}, "
           f"E={int(best['max_E'])}, kNN={int(best['k_neighbors'])}, "
           f"tau={int(best['tau'])}")
    held = val[val.split == "held_out"]
    held_med = held.groupby("k").MG.median()
    z_mae = float(np.mean(np.abs(segments.error)))
    text = [
        "# Frobenius-norm dimension recovery, k=1..10", "",
        "## Protocol", "",
        "One fixed MG configuration is selected on seed 0. Seeds 1 and 2 and the "
        "non-monotone transition schedule are held out. No per-k tuning is allowed.", "",
        f"Selected configuration: `{cfg}`.", "",
        "## Stationary validation", "",
        "| seed | split | MAE | max error | Spearman | inversions |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in summaries.iterrows():
        text.append(f"| {int(r.seed)} | {r.split} | {r.mae:.3f} | "
                    f"{r.max_error:.3f} | {r.rho:.3f} | {int(r.inversions)} |")
    text += ["", "Held-out median by true dimension:", "",
             "| true k | MG | error |", "|---:|---:|---:|"]
    for k in KS:
        v = float(held_med.loc[k])
        text.append(f"| {k} | {v:.3f} | {v-k:+.3f} |")
    text += ["", "## Held-out zigzag schedule", "",
             "Schedule: `" + " -> ".join(map(str, SCHEDULE)) + "`.", "",
             "| segment | true k | median MG | error | windows |",
             "|---:|---:|---:|---:|---:|"]
    for _, r in segments.iterrows():
        text.append(f"| {int(r.segment)} | {int(r.true_k)} | {r.MG:.3f} | "
                    f"{r.error:+.3f} | {int(r.n_windows)} |")
    exact = bool((held.groupby("seed").apply(lambda q: np.max(np.abs(q.error)) <= .5,
                                              include_groups=False)).all())
    text += ["", "## Verdict", "",
             f"Zigzag MAE: **{z_mae:.3f} components**.", "",
             ("The pre-registered ±0.5-component target is met on every held-out seed."
              if exact else
              "The ±0.5-component target is **not** met on every held-out seed. "
              "The experiment therefore does not establish exact absolute recovery; "
              "the residual error must be reported rather than hidden by rounding."), "",
             "The selected point is a calibration result for this torus family. It does "
             "not by itself transfer to neural-network training logs, whose coverage is unknown.", "",
             f"Runtime: {elapsed:.1f} seconds. Grid points: {len(grid_frame)}."]
    (OUT / "report_exp9.md").write_text("\n".join(text), encoding="utf-8")
    (OUT / "best_config.json").write_text(json.dumps({k: (float(v) if isinstance(v, np.floating) else v)
                                                        for k, v in best.items()}, indent=2),
                                           encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use the larger calibration grid")
    args = ap.parse_args()
    t0 = time.time()
    frame, best = calibrate(args.full)
    val = validate(best)
    trace, segments = transition(best)
    figures(val, trace, segments, best)
    write_report(frame, best, val, segments, time.time() - t0)
    print("best:", {k: best[k] for k in ("window", "cycles", "max_E", "k_neighbors", "tau",
                                           "mae", "max_error", "rho", "inversions")})
    print("outputs:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

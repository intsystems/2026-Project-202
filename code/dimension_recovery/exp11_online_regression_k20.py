"""Known-dimension ML task: online orthogonal regression, k=1..20.

The model is a linear regressor on the D one-hot inputs.  At every step it
performs a genuine full-batch gradient update against a slowly changing
teacher.  Exactly k teacher coordinates are independently quasiperiodic and
the other D-k coordinates are constant and already fitted.  After burn-in the
weight trajectory therefore has known intrinsic dimension k.

One estimator configuration per scalar observer is selected on seed 0.  The
configuration is then evaluated unchanged on held-out seeds 1 and 2.  There is
no per-k calibration and no remapping of MG to the known answer.

Run:
    python exp11_online_regression_k20.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
OUT = HERE / "results" / "exp11_online_regression_k20"
OUT.mkdir(parents=True, exist_ok=True)

KS = np.arange(1, 21)
D = 64
OBSERVERS = ("weight_fro", "weight_projection", "weight_trace",
             "gradient_fro", "gradient_projection", "loss")

# Curated joint grid: data coverage, delay coordinates, neighbourhood scale,
# delay and optimiser response.  The grid is shared by every k; selection is
# never allowed to see the held-out seeds.
GRID = (
    dict(window=2000, cycles=500.0, max_E=31, kn=20, tau=1, eta=.20),
    dict(window=3000, cycles=800.0, max_E=31, kn=30, tau=2, eta=.20),
    dict(window=4000, cycles=1200.0, max_E=41, kn=40, tau=2, eta=.20),
    dict(window=4000, cycles=1800.0, max_E=41, kn=50, tau=4, eta=.20),
)


def simulate(k: int, seed: int, cfg: dict) -> dict[str, np.ndarray]:
    """Train a one-hot linear regressor and return scalar training logs.

    For X=I the full-batch objective is 0.5*||w-y(t)||^2 and its gradient is
    w-y(t).  Thus this is ordinary GD, not a hand-written weight trajectory.
    """
    W = int(cfg["window"])
    burn = 1000
    n = W + burn
    eta = float(cfg["eta"])
    rng = np.random.default_rng(10_000 + seed)
    t = np.arange(n, dtype=float)

    baseline = 1.0 + .25 * rng.standard_normal(D)
    baseline[np.abs(baseline) < .5] = .5
    target = np.tile(baseline, (n, 1))
    freq = S.frequencies(k, float(cfg["cycles"]), W, seed=seed,
                         band_mode="matched")
    phase = rng.uniform(0, 2*np.pi, k)
    amp = .10 * (.5 + .5*rng.random(k))
    target[:, :k] += amp * np.sin(2*np.pi*t[:, None]*freq[None, :]
                                  + phase[None, :])

    # Full-batch GD on the D one-hot training examples.
    weights = np.empty_like(target)
    gradients = np.empty_like(target)
    w = baseline.copy()
    for j in range(n):
        g = w - target[j]
        gradients[j] = g
        weights[j] = w
        w = w - eta*g

    weights = weights[burn:]
    gradients = gradients[burn:]
    r_w = np.random.default_rng(20_000 + seed).standard_normal(D)
    r_g = np.random.default_rng(30_000 + seed).standard_normal(D)

    raw = {
        "weight_fro": np.linalg.norm(weights, axis=1),
        "weight_projection": weights @ r_w,
        "weight_trace": weights.sum(axis=1),
        "gradient_fro": np.linalg.norm(gradients, axis=1),
        "gradient_projection": gradients @ r_g,
        "loss": .5*np.sum(gradients**2, axis=1),
    }
    # Scalar scaling must not influence a distance-based dimension estimate.
    # A tiny measurement noise avoids exact duplicate distances.
    noise_rng = np.random.default_rng(40_000 + seed)
    out = {}
    for name, x in raw.items():
        z = (x-x.mean())/(x.std()+1e-15)
        out[name] = z + 1e-6*noise_rng.standard_normal(W)
    return out


def one_job(cfg_id: int, k: int, seed: int) -> list[dict]:
    cfg = GRID[cfg_id]
    obs = simulate(k, seed, cfg)
    rows = []
    for name, x in obs.items():
        value = E.mg(x, max_E=int(cfg["max_E"]), k=int(cfg["kn"]),
                     tau=int(cfg["tau"]))
        rows.append(dict(cfg_id=cfg_id, seed=seed, k=k, observer=name,
                         MG=float(value), error=float(value-k), **cfg))
    return rows


def score(group: pd.DataFrame) -> dict:
    g = group.sort_values("k")
    err = g.MG.to_numpy()-g.k.to_numpy()
    inv = int(np.sum(np.diff(g.MG.to_numpy()) <= 0))
    return dict(mae=float(np.mean(np.abs(err))),
                max_error=float(np.max(np.abs(err))),
                rho=float(spearmanr(g.k, g.MG).statistic),
                inversions=inv,
                objective=float(np.mean(np.abs(err)) + .15*np.max(np.abs(err))
                                + .15*inv))


def run_jobs(jobs: list[tuple], workers: int = 4) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(one_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(fs), 1):
            rows.extend(f.result())
            if i % 10 == 0:
                print(f"completed {i}/{len(fs)} jobs", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()
    calibration = run_jobs([(c, int(k), 0) for c in range(len(GRID)) for k in KS])
    calibration.to_csv(OUT/"calibration_raw.csv", index=False)

    summary_rows = []
    for (observer, cfg_id), g in calibration.groupby(["observer", "cfg_id"]):
        summary_rows.append(dict(observer=observer, cfg_id=cfg_id, **score(g)))
    summary = pd.DataFrame(summary_rows).sort_values(["observer", "objective"])
    summary.to_csv(OUT/"calibration_summary.csv", index=False)
    best = summary.groupby("observer", as_index=False).first()
    best.to_csv(OUT/"best_by_observer.csv", index=False)

    jobs = sorted({(int(r.cfg_id), int(k), seed)
                   for _, r in best.iterrows() for seed in (1, 2) for k in KS})
    held = run_jobs(jobs)
    # A job returns all observers; retain only observer/config pairs selected above.
    keep = {(r.observer, int(r.cfg_id)) for _, r in best.iterrows()}
    held = held[[((r.observer, int(r.cfg_id)) in keep) for _, r in held.iterrows()]]
    held.to_csv(OUT/"heldout_raw.csv", index=False)

    held_summary = []
    for (observer, seed), g in held.groupby(["observer", "seed"]):
        held_summary.append(dict(observer=observer, seed=seed, **score(g)))
    hs = pd.DataFrame(held_summary)
    hs.to_csv(OUT/"heldout_summary.csv", index=False)

    # Plot all observers against the identity line, held-out median across seeds.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True,
                             sharex=True, sharey=True)
    med = held.groupby(["observer", "k"], as_index=False).MG.median()
    for ax, name in zip(axes.flat, OBSERVERS):
        g = med[med.observer == name]
        ax.plot(KS, KS, "k--", lw=1, label="ideal")
        ax.plot(g.k, g.MG, "o-", ms=3.5, lw=1.3, color="#28669b")
        ax.set_title(name); ax.grid(alpha=.22)
        ax.set_xlabel("True k"); ax.set_ylabel("MG estimate")
    fig.suptitle("Online regression: held-out dimension recovery", fontsize=14)
    fig.savefig(OUT/"heldout_observers_k20.png", dpi=220)
    fig.savefig(OUT/"heldout_observers_k20.pdf")
    plt.close(fig)

    ranking = (hs.groupby("observer", as_index=False)
                 .agg(MAE=("mae", "mean"), max_error=("max_error", "max"),
                      rho=("rho", "mean"), inversions=("inversions", "mean"))
                 .sort_values("MAE"))
    ranking.to_csv(OUT/"observer_ranking.csv", index=False)
    winner = ranking.iloc[0].observer
    win_med = med[med.observer == winner]

    lines = [
        "# Known-dimension online-regression experiment", "",
        "A D=64 linear model was trained by full-batch gradient descent on one-hot inputs. "
        "Exactly k target coordinates changed quasiperiodically; all other targets were "
        "constant and already fitted. Therefore the active training dynamics had known "
        "dimension k. Values k=1,...,20 were tested.", "",
        "A configuration was selected separately for each scalar observer on seed 0, but "
        "one common configuration was used for all k. Seeds 1 and 2 were held out. No "
        "per-k correction or output remapping was used.", "",
        "## Held-out ranking", "",
        "| observer | MAE | max error | mean rho | mean inversions |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in ranking.iterrows():
        lines.append(f"| {r.observer} | {r.MAE:.3f} | {r.max_error:.3f} | "
                     f"{r.rho:.3f} | {r.inversions:.1f} |")
    lines += ["", f"Best observer by held-out MAE: **{winner}**.", "",
              "| true k | held-out median MG | error |", "|---:|---:|---:|"]
    for _, r in win_med.iterrows():
        lines.append(f"| {int(r.k)} | {r.MG:.3f} | {r.MG-r.k:+.3f} |")
    lines += ["", "![All observers](heldout_observers_k20.png)", "",
              f"Runtime: {time.time()-started:.1f} seconds."]
    (OUT/"final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT/"grid.json").write_text(json.dumps(GRID, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False))
    print("outputs", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Corrected controlled dimension recovery on a real image dataset.

Real sklearn-digits images and labels train a nonlinear MLP. Around that trained
network we construct a k-dimensional parameter adapter whose Jacobian in function
space is full-rank and whitened. A time-dependent teacher moves quasiperiodically
in all k adapter coordinates; a student tracks it by exact backpropagation.

This benchmark distinguishes:
  * available dimension: k adapter coordinates;
  * functional dimension: rank d f(X_probe) / d z (checked numerically);
  * active dynamic dimension: covariance rank of z(t) and updates (checked).

Compared with invalid v1/v2: no labels are inserted into logits, projections are
fixed, every coordinate is persistently excited, frequencies fill the attractor,
bandwidth is matched across k, E > 2k, and temporal neighbours are excluded.
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import estimators as E
import systems as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp15_real_digits_functional_subspace_v3"
OUT.mkdir(parents=True, exist_ok=True)

KS = np.arange(1, 9)                 # reliable finite-sample range first
SEEDS = (0, 1, 2)
N_IN, N_HIDDEN, N_OUT = 64, 16, 10
N_PROBE_PER_CLASS = 10
KMAX = int(KS.max())
P = N_HIDDEN*N_IN + N_HIDDEN + N_OUT*N_HIDDEN + N_OUT
OBSERVERS = (
    "parameter_fro", "parameter_projection",
    "latent_fro", "latent_projection",
    "gradient_fro", "gradient_projection",
    "output_fro", "output_projection", "loss",
)

# All embeddings exceed 2*KMAX. Frequencies occupy a matched band for every k.
# cycles/window is high enough to produce recurrences rather than one smooth arc.
GRID = (
    dict(window=3000, cycles=450.0, max_E=21, kn=20, tau=1, eta=.18),
    dict(window=4000, cycles=800.0, max_E=25, kn=30, tau=1, eta=.15),
    dict(window=5000, cycles=1200.0, max_E=25, kn=40, tau=2, eta=.12),
    dict(window=5000, cycles=1500.0, max_E=31, kn=50, tau=2, eta=.10),
)


def layout(theta):
    q = 0
    W1 = theta[q:q+N_HIDDEN*N_IN].reshape(N_HIDDEN, N_IN); q += N_HIDDEN*N_IN
    b1 = theta[q:q+N_HIDDEN]; q += N_HIDDEN
    W2 = theta[q:q+N_OUT*N_HIDDEN].reshape(N_OUT, N_HIDDEN); q += N_OUT*N_HIDDEN
    b2 = theta[q:q+N_OUT]
    return W1, b1, W2, b2


def pack(clf):
    return np.concatenate([
        clf.coefs_[0].T.ravel(), clf.intercepts_[0],
        clf.coefs_[1].T.ravel(), clf.intercepts_[1],
    ]).astype(np.float64)


def forward(theta, X):
    W1, b1, W2, b2 = layout(theta)
    hidden = np.tanh(X @ W1.T + b1)
    logits = hidden @ W2.T + b2
    return logits, hidden


def loss_gradient(theta, X, target):
    """MSE and exact full-parameter gradient through both MLP layers."""
    W1, b1, W2, b2 = layout(theta)
    output, hidden = forward(theta, X)
    residual = output - target
    doutput = residual / len(X)
    gW2 = doutput.T @ hidden
    gb2 = doutput.sum(axis=0)
    dh = (doutput @ W2) * (1.0-hidden*hidden)
    gW1 = dh.T @ X
    gb1 = dh.sum(axis=0)
    grad = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
    return .5*float(np.sum(residual*residual))/len(X), grad, output


def balanced_probe(X, y, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                          for c in range(N_OUT)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def prepare_seed(seed):
    """Train a real classifier and construct a full-rank whitened adapter."""
    path = OUT / f"prepared_seed{seed}.npz"
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.25, random_state=seed, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(N_HIDDEN,), activation="tanh",
                        solver="lbfgs", alpha=1e-4, max_iter=500,
                        random_state=10_000+seed)
    clf.fit(Xtr, ytr)
    theta0 = pack(clf)
    Xp, yp = balanced_probe(Xte, yte, N_PROBE_PER_CLASS, 20_000+seed)
    base_acc = float((forward(theta0, Xp)[0].argmax(1) == yp).mean())

    # Candidate directions are orthonormal in parameter space.
    rng = np.random.default_rng(30_000+seed)
    V, _ = np.linalg.qr(rng.standard_normal((P, KMAX)), mode="reduced")
    eps = 1e-5
    F = np.empty((len(Xp)*N_OUT, KMAX))
    for j in range(KMAX):
        yp1 = forward(theta0+eps*V[:, j], Xp)[0]
        ym1 = forward(theta0-eps*V[:, j], Xp)[0]
        F[:, j] = ((yp1-ym1)/(2*eps)).ravel()

    # F = Q R. U = V R^-1 sqrt(N) gives (J U)'(J U)/N = I locally.
    # Thus all adapter directions are functionally visible and equally scaled.
    Q, R = np.linalg.qr(F, mode="reduced")
    U = V @ np.linalg.inv(R) * np.sqrt(len(Xp))
    FU = F @ np.linalg.inv(R) * np.sqrt(len(Xp))
    sv = np.linalg.svd(FU, compute_uv=False)
    functional_rank = int(np.sum(sv > sv[0]*1e-8))
    np.savez(path, Xp=Xp, yp=yp, theta0=theta0, U=U,
             base_acc=base_acc, functional_rank=functional_rank,
             jacobian_ratio=float(sv[-1]/sv[0]))
    return dict(seed=seed, base_acc=base_acc, functional_rank=functional_rank,
                jacobian_ratio=float(sv[-1]/sv[0]))


def numerical_rank_and_pr(X, rel=1e-7):
    X = np.asarray(X, dtype=float)
    X = X-X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(X, compute_uv=False)
    if len(sv) == 0 or sv[0] == 0:
        return 0, 0.0, 0.0
    rank = int(np.sum(sv > rel*sv[0]))
    eig = sv*sv
    pr = float(eig.sum()**2/(np.sum(eig*eig)+1e-30))
    return rank, pr, float(sv[-1]/sv[0])


def functional_rank_at(theta0, U, X, z):
    """Rank of z -> probe logits at an actual point, not only at z=0."""
    k = len(z); eps = 1e-5; cols = []
    eye = np.eye(k)
    for j in range(k):
        p = forward(theta0+U[:, :k]@(z+eps*eye[j]), X)[0]
        m = forward(theta0+U[:, :k]@(z-eps*eye[j]), X)[0]
        cols.append(((p-m)/(2*eps)).ravel())
    sv = np.linalg.svd(np.column_stack(cols), compute_uv=False)
    return int(np.sum(sv > sv[0]*1e-8)), float(sv[-1]/sv[0])


def simulate(cfg_id, k, seed):
    cfg = GRID[cfg_id]; W = int(cfg["window"]); burn = 1500; n = W+burn
    p = np.load(OUT/f"prepared_seed{seed}.npz")
    X, labels = p["Xp"], p["yp"]
    theta0, U = p["theta0"], p["U"][:, :k]
    rng = np.random.default_rng(40_000+101*seed+k)
    freq = S.frequencies(k, float(cfg["cycles"]), W,
                         seed=50_000+seed, band_mode="matched")
    phase = rng.uniform(0, 2*np.pi, k)
    amp = .14*(.8+.2*rng.random(k))
    centre = .02*rng.standard_normal(k)
    t = np.arange(n, dtype=float)
    zstar = centre[None, :] + amp[None, :]*np.sin(
        2*np.pi*t[:, None]*freq[None, :]+phase[None, :])

    rp_theta = rng.standard_normal(P); rp_theta /= np.linalg.norm(rp_theta)
    rp_grad = rng.standard_normal(P); rp_grad /= np.linalg.norm(rp_grad)
    rp_z = rng.standard_normal(k); rp_z /= np.linalg.norm(rp_z)
    rp_out = rng.standard_normal(len(X)*N_OUT); rp_out /= np.linalg.norm(rp_out)
    series = {name: np.empty(n) for name in OBSERVERS}
    zs = np.empty((n, k)); updates = np.empty((n, k)); acc = np.empty(n)
    z = centre.copy(); eta = float(cfg["eta"])
    for j in range(n):
        theta = theta0+U@z
        teacher = forward(theta0+U@zstar[j], X)[0]
        loss, gtheta, output = loss_gradient(theta, X, teacher)
        gz = U.T@gtheta
        zs[j], updates[j] = z, -eta*gz
        series["parameter_fro"][j] = np.linalg.norm(theta)
        series["parameter_projection"][j] = theta@rp_theta
        series["latent_fro"][j] = np.linalg.norm(z)
        series["latent_projection"][j] = z@rp_z
        series["gradient_fro"][j] = np.linalg.norm(gtheta)
        series["gradient_projection"][j] = gtheta@rp_grad
        series["output_fro"][j] = np.linalg.norm(output)
        series["output_projection"][j] = output.ravel()@rp_out
        series["loss"][j] = loss
        acc[j] = np.mean(output.argmax(1) == labels)
        z = z-eta*gz

    zs, updates, acc = zs[burn:], updates[burn:], acc[burn:]
    zrank, zpr, zratio = numerical_rank_and_pr(zs)
    urank, upr, uratio = numerical_rank_and_pr(updates)
    ranks = [functional_rank_at(theta0, U, X, zpoint)
             for zpoint in (np.zeros(k), zs[len(zs)//2], zs[-1])]
    frank = min(x[0] for x in ranks); fratio = min(x[1] for x in ranks)
    nrng = np.random.default_rng(60_000+seed)
    out = {}
    for name, x in series.items():
        x = x[burn:]
        out[name] = (x-x.mean())/(x.std()+1e-15) + 1e-6*nrng.standard_normal(W)
    diagnostics = dict(
        functional_rank=frank, functional_ratio=fratio,
        trajectory_rank=zrank, trajectory_pr=zpr, trajectory_ratio=zratio,
        update_rank=urank, update_pr=upr, update_ratio=uratio,
        mean_accuracy=float(acc.mean()), min_accuracy=float(acc.min()),
        resonance_margin=S.resonance_margin(freq),
    )
    return out, diagnostics


def one_job(cfg_id, k, seed):
    cfg = GRID[cfg_id]
    observers, diag = simulate(cfg_id, k, seed)
    rows = []
    theiler = (int(cfg["max_E"])-1)*int(cfg["tau"])
    for name, x in observers.items():
        value = E.mg(x, max_E=int(cfg["max_E"]), k=int(cfg["kn"]),
                     tau=int(cfg["tau"]), theiler=theiler)
        rows.append(dict(cfg_id=cfg_id, seed=seed, k=k, observer=name,
                         MG=float(value), error=float(value-k),
                         roughness=E.roughness(x), theiler=theiler,
                         **diag, **cfg))
    return rows


def score(g):
    g = g.sort_values("k"); error = g.MG.to_numpy()-g.k.to_numpy()
    return dict(mae=float(np.mean(np.abs(error))),
                max_error=float(np.max(np.abs(error))),
                rho=float(spearmanr(g.k, g.MG).statistic),
                inversions=int(np.sum(np.diff(g.MG.to_numpy()) <= 0)),
                objective=float(np.mean(np.abs(error))+.15*np.max(np.abs(error))))


def run_jobs(jobs, workers=4):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(one_job, *job) for job in jobs]
        for i, future in enumerate(as_completed(fs), 1):
            rows.extend(future.result())
            if i % 8 == 0: print(f"completed {i}/{len(fs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    started = time.time()
    preparation = pd.DataFrame([prepare_seed(s) for s in SEEDS])
    preparation.to_csv(OUT/"prepared_models.csv", index=False)
    print(preparation.to_string(index=False), flush=True)

    cal = run_jobs([(c, int(k), 0) for c in range(len(GRID)) for k in KS])
    cal.to_csv(OUT/"calibration_raw.csv", index=False)
    rows = []
    for (obs, cfg), g in cal.groupby(["observer", "cfg_id"]):
        rows.append(dict(observer=obs, cfg_id=cfg, **score(g)))
    cs = pd.DataFrame(rows).sort_values(["observer", "objective"])
    cs.to_csv(OUT/"calibration_summary.csv", index=False)
    best = cs.groupby("observer", as_index=False).first()
    best.to_csv(OUT/"best_by_observer.csv", index=False)

    configs = sorted(set(int(x) for x in best.cfg_id))
    held = run_jobs([(c, int(k), seed) for c in configs
                     for seed in (1, 2) for k in KS])
    keep = {(r.observer, int(r.cfg_id)) for _, r in best.iterrows()}
    held = held[[((r.observer, int(r.cfg_id)) in keep) for _, r in held.iterrows()]]
    held.to_csv(OUT/"heldout_raw.csv", index=False)
    rows = []
    for (obs, seed), g in held.groupby(["observer", "seed"]):
        rows.append(dict(observer=obs, seed=seed, **score(g)))
    hs = pd.DataFrame(rows); hs.to_csv(OUT/"heldout_summary.csv", index=False)
    ranking = (hs.groupby("observer", as_index=False)
        .agg(MAE=("mae", "mean"), max_error=("max_error", "max"),
             rho=("rho", "mean"), inversions=("inversions", "mean"))
        .sort_values("MAE"))
    ranking.to_csv(OUT/"observer_ranking.csv", index=False)

    diagcols = ["seed", "k", "cfg_id", "functional_rank", "functional_ratio",
                "trajectory_rank", "trajectory_pr", "trajectory_ratio",
                "update_rank", "update_pr", "update_ratio", "mean_accuracy",
                "min_accuracy", "resonance_margin"]
    diagnostics = held[diagcols].drop_duplicates()
    diagnostics.to_csv(OUT/"rank_diagnostics.csv", index=False)
    rank_ok = bool(np.all(diagnostics.functional_rank == diagnostics.k) and
                   np.all(diagnostics.trajectory_rank == diagnostics.k) and
                   np.all(diagnostics.update_rank == diagnostics.k))

    med = held.groupby(["observer", "k"], as_index=False).MG.median()
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, OBSERVERS):
        g = med[med.observer == name]
        ax.plot(KS, KS, "k--", lw=1, label="ideal")
        ax.plot(g.k, g.MG, "o-", ms=3.5, lw=1.2, label="MG")
        ax.set(title=name, xlabel="True active dimension k", ylabel="MG estimate")
        ax.grid(alpha=.22)
    fig.suptitle("Corrected Digits MLP functional-subspace benchmark", fontsize=14)
    fig.savefig(OUT/"heldout_observers.png", dpi=220)
    fig.savefig(OUT/"heldout_observers.pdf"); plt.close(fig)

    winner = ranking.iloc[0].observer
    wm = med[med.observer == winner]
    report = [
        "# Corrected real-Digits dimension-recovery experiment", "",
        "## Setup", "",
        "A nonlinear 64-16-10 tanh MLP was trained on the real sklearn Digits labels. "
        "A local adapter was then constructed around the trained network. Its function-space "
        "Jacobian was QR-whitened, so its first k columns have rank k and equal local scale. "
        "A quasiperiodically moving teacher excites every adapter coordinate; the student "
        "tracks it using exact backpropagation through both MLP layers.", "",
        "The sweep uses k=1,...,8, fixed projections, matched spectral bandwidth, E>2k, "
        "and a Theiler exclusion equal to the delay-vector span. Hyperparameters were selected "
        "on seed 0 and frozen for held-out seeds 1 and 2.", "",
        "## Ground-truth checks", "",
        f"All functional, trajectory-covariance and update-covariance ranks equal k: **{rank_ok}**.", "",
        "## Held-out results", "",
        "| Observer | MAE | Max error | Spearman rho | Mean inversions |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in ranking.iterrows():
        report.append(f"| `{r.observer}` | {r.MAE:.3f} | {r.max_error:.3f} | "
                      f"{r.rho:.3f} | {r.inversions:.1f} |")
    report += ["", f"Best observer: **`{winner}`**.", "",
               "| True k | Median held-out MG | Error |", "|---:|---:|---:|"]
    for _, r in wm.iterrows():
        report.append(f"| {int(r.k)} | {r.MG:.3f} | {r.MG-r.k:+.3f} |")
    report += ["", "![Held-out observer comparison](heldout_observers.png)", "",
               "## Interpretation", "",
               "This experiment tests recovery of an explicitly verified active dynamical "
               "dimension on real inputs and in a nonlinear trained network. It does not claim "
               "that every scalar observer is generic or that available parameter count alone "
               "equals dynamic dimension.", "", f"Runtime: {time.time()-started:.1f} s."]
    (OUT/"final_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT/"run_metadata.json").write_text(json.dumps({
        "rank_checks_passed": rank_ok, "best_observer": winner,
        "runtime_seconds": time.time()-started}, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False)); print("rank checks", rank_ok)
    print("outputs", OUT)


if __name__ == "__main__":
    main()

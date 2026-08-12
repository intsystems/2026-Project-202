"""Real-input controlled benchmark for recovering a known functional dimension.

Digits images are real. A frozen nonlinear feature map produces 20 independent
logit directions on a fixed probe set. The teacher coefficients are driven by
independent incommensurate oscillations, so all k directions are excited. The
student tracks this teacher using gradient descent in the first k directions.
This avoids the rank-one transient of a fixed-target linear quadratic loss.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import estimators as E

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp15_real_digits_functional_subspace_v2"
OUT.mkdir(parents=True, exist_ok=True)
KS = np.arange(1, 21)
SEEDS = (0, 1, 2)
N_FEATURES, N_CLASSES, KMAX = 64, 10, 20
STEPS, WINDOW, STRIDE = 7000, 4000, 250

def make_basis(seed: int):
    d = load_digits()
    X = d.data.astype(float) / 16.0
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    rng = np.random.default_rng(1000 + seed)
    W = rng.standard_normal((N_FEATURES, X.shape[1])) / np.sqrt(X.shape[1])
    b = 0.1 * rng.standard_normal(N_FEATURES)
    Htr = np.tanh(Xtr @ W.T + b)
    Hte = np.tanh(Xte @ W.T + b)
    A = rng.standard_normal((KMAX, N_CLASSES, N_FEATURES)) / np.sqrt(N_FEATURES)
    Ftr = np.stack([(Htr @ A[j].T).ravel() for j in range(KMAX)], axis=1)
    Fte = np.stack([(Hte @ A[j].T).ravel() for j in range(KMAX)], axis=1)
    # QR makes directions independent in function space on the real probe inputs.
    Q, _ = np.linalg.qr(Ftr, mode="reduced")
    R = np.linalg.lstsq(Ftr, Q, rcond=None)[0]
    Qte = Fte @ R
    Btr = Q.reshape(len(Xtr), N_CLASSES, KMAX)
    Bte = Qte.reshape(len(Xte), N_CLASSES, KMAX)
    # A frozen baseline classifier from the real labels.
    Y = np.zeros((len(ytr), N_CLASSES), dtype=float)
    Y[np.arange(len(ytr)), ytr] = 1.0
    baseline = 2.0 * (Y - 0.1)
    return Btr, Bte, ytr, yte, baseline

def one(k: int, seed: int):
    Btr, Bte, ytr, yte, baseline = make_basis(seed)
    n = len(ytr); t = np.arange(STEPS, dtype=float)
    # Independent frequencies: an observable scalar has a genuine k-mode attractor.
    freq = (0.70 + 0.071 * np.arange(k)) / WINDOW
    phase = np.random.default_rng(2000 + seed).uniform(0, 2*np.pi, k)
    amp = 0.8 * (0.75 + 0.25*np.random.default_rng(3000 + seed).random(k))
    zstar = amp[None, :] * np.sin(2*np.pi*t[:, None]*freq[None, :] + phase[None, :])
    z = np.zeros(k); rng = np.random.default_rng(4000 + 17*seed + k)
    # Draw every random projection once; never regenerate it inside the loop.
    rp_z = rng.standard_normal(k); rp_z /= np.linalg.norm(rp_z)
    rp_g = rng.standard_normal(k); rp_g /= np.linalg.norm(rp_g)
    rows = []
    for step in range(STEPS):
        pred = baseline + np.einsum("nck,k->nc", Btr[:, :, :k], z)
        target = baseline + np.einsum("nck,k->nc", Btr[:, :, :k], zstar[step])
        err = pred - target
        loss = 0.5 * float(np.mean(err * err))
        # Full-batch gradient in the known k-dimensional adapter.
        # Q columns have unit Euclidean norm on the complete probe vector.
        # Use the corresponding least-squares gradient; dividing by n here would
        # make the dynamics almost frozen because the basis is already normalized.
        gz = np.einsum("nck,nc->k", Btr[:, :, :k], err)
        z -= 0.8 * gz
        if step % 1 == 0:
            logits = baseline + np.einsum("nck,k->nc", Btr[:, :, :k], z)
            v = {
                "latent_fro": float(np.linalg.norm(z)),
                "latent_projection": float(z @ rp_z),
                "gradient_fro": float(np.linalg.norm(gz)),
                "gradient_projection": float(gz @ rp_g),
                "output_fro": float(np.linalg.norm(logits)),
                "loss": loss,
                "train_acc": float((logits.argmax(1) == ytr).mean()),
            }
            v.update(step=step, k=k, seed=seed); rows.append(v)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"trajectory_k{k}_seed{seed}.csv", index=False)
    estimates = []
    for name in ("latent_fro", "latent_projection", "gradient_fro",
                 "gradient_projection", "output_fro", "loss"):
        x = df[name].to_numpy(); x = (x - x.mean()) / (x.std() + 1e-12)
        # Use a sliding-window median, not one estimate over a convergence transient.
        vals = []
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            try: vals.append(E.mg(x[start:start+WINDOW], max_E=31, k=20, tau=1))
            except Exception: pass
        estimates.append(dict(observer=name, k=k, seed=seed,
                              MG=float(np.nanmedian(vals)) if vals else np.nan))
    return estimates

def main():
    t0 = time.time(); rows = []
    for seed in SEEDS:
        for k in KS:
            print(f"k={k}, seed={seed}", flush=True)
            rows.extend(one(int(k), seed))
    raw = pd.DataFrame(rows); raw["error"] = raw.MG - raw.k
    raw.to_csv(OUT / "estimates_raw.csv", index=False)
    summary = (raw.groupby("observer", as_index=False)
        .agg(MAE=("error", lambda x: float(np.nanmean(np.abs(x)))),
             max_error=("error", lambda x: float(np.nanmax(np.abs(x)))),
             rho=("k", lambda x: float(spearmanr(x, raw.loc[x.index, "MG"], nan_policy="omit").statistic))))
    summary.to_csv(OUT / "observer_summary.csv", index=False)
    med = raw.groupby(["observer", "k"], as_index=False).MG.median()
    names = list(summary.observer); fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        g = med[med.observer == name]; ax.plot(KS, KS, "k--", lw=1); ax.plot(g.k, g.MG, "o-", ms=3)
        ax.set(title=name, xlabel="true k", ylabel="MG estimate"); ax.grid(alpha=.2)
    for ax in axes.flat[len(names):]: ax.axis("off")
    fig.suptitle("Real digits inputs with a known k-dimensional nonlinear function subspace")
    fig.savefig(OUT / "observer_vs_k.png", dpi=180); fig.savefig(OUT / "observer_vs_k.pdf"); plt.close(fig)
    report = ["# Real-digits functional-subspace benchmark (v2)", "",
      "The input data are real sklearn digits. A frozen tanh feature map and QR-orthogonalized logit functions define a known k-dimensional function subspace. The teacher coefficients have independent incommensurate temporal drives, and the student tracks them by gradient descent. This explicitly excites all k directions; the earlier fixed-target version produced a rank-one exponential transient. Random projections are fixed for the entire run.", "",
      "| observer | MAE | max error | Spearman rho |", "|---|---:|---:|---:|"]
    for _, r in summary.iterrows(): report.append(f"| {r.observer} | {r.MAE:.3f} | {r.max_error:.3f} | {r.rho:.3f} |")
    report += ["", "The known k is the rank of the probe-set Jacobian by construction. This validates recoverability of a controlled functional dimension, not the claim that every scalar norm is informative.", "", "![Observers](observer_vs_k.png)", "", f"Runtime: {time.time()-t0:.1f} s."]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False)); print(OUT)

if __name__ == "__main__": main()

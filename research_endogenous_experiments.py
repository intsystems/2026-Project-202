"""Run the three proposed endogenous-dynamics experiments.

Outputs are written to ``research_endogenous_results``.  The script uses the
article's estimator and diagnostics from ``code/actdim``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from actdim.estimator.config import EstimatorConfig
from actdim.estimator.diagnostics import diagnose
from actdim.estimator.windows import score

OUT = ROOT / "research_endogenous_results"
OUT.mkdir(exist_ok=True)


def heavy_ball_quadratic(rank: int, n: int, seed: int, eta: float = 0.55):
    """Undamped heavy-ball on a diagonal quadratic.

    For 0 < eta*lambda_i < 4, each excited mode is elliptic.  Incommensurate
    eigenvalues produce recurrent motion on a torus without external forcing.
    """
    rng = np.random.default_rng(seed)
    dim = 12
    lambdas = np.linspace(0.17, 1.31, dim)
    theta = np.zeros(dim)
    momentum = np.zeros(dim)
    theta[:rank] = rng.normal(size=rank)
    momentum[:rank] = rng.normal(0.0, 0.15, size=rank)
    rows, states = [], []
    for t in range(n):
        grad = lambdas * theta
        momentum -= eta * grad
        theta += momentum
        if t % 2 == 0:
            rows.append((t, 0.5 * float(np.dot(lambdas * theta, theta)),
                         np.linalg.norm(theta), np.linalg.norm(grad)))
            states.append(np.r_[theta[:rank], momentum[:rank]])
    return np.asarray(rows), np.asarray(states)


def matrix_factorization(rank: int, n: int, seed: int, eta: float = 0.02,
                         beta: float = 1.0, d: int = 8, width: int = 8):
    """Dense low-rank matrix factorization with full-batch momentum.

    The target is a dense matrix with known rank ``rank``.  The learned
    factors are full dense matrices; no coordinate-wise or diagonal
    decoupling is imposed.
    """
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(d, d)))
    right, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sigma = np.linspace(1.0, 0.35, rank)
    target = left[:, :rank] @ np.diag(sigma) @ right[:, :rank].T

    # Start close to a balanced factorization, with a small dense perturbation.
    u = np.zeros((d, width))
    v = np.zeros((d, width))
    u[:, :rank] = left[:, :rank] @ np.diag(np.sqrt(sigma))
    v[:, :rank] = right[:, :rank] @ np.diag(np.sqrt(sigma))
    u += 0.02 * rng.normal(size=u.shape)
    v += 0.02 * rng.normal(size=v.shape)
    vu = np.zeros_like(u)
    vv = np.zeros_like(v)
    rows, states = [], []
    for t in range(n):
        err = u @ v.T - target
        gu = err @ v
        gv = err.T @ u
        vu = beta * vu + gu
        vv = beta * vv + gv
        u -= eta * vu
        v -= eta * vv
        if t % 2 == 0:
            rows.append((t, 0.5 * float(np.sum(err * err)),
                         np.linalg.norm(np.r_[u.ravel(), v.ravel()]),
                         np.linalg.norm(np.r_[vu.ravel(), vv.ravel()])))
            states.append(np.r_[u.ravel(), v.ravel(), vu.ravel(), vv.ravel()])
    return np.asarray(rows), np.asarray(states)


def teacher_student(rank: int, n: int, seed: int, eta: float = 0.02,
                    beta: float = 1.0, hidden: int = 8):
    """Two-layer linear teacher-student regression with full-batch momentum."""
    rng = np.random.default_rng(seed)
    d_in = d_out = 8
    q, _ = np.linalg.qr(rng.normal(size=(d_in, d_in)))
    p, _ = np.linalg.qr(rng.normal(size=(d_out, d_out)))
    singular = np.zeros(min(d_in, d_out))
    singular[:rank] = np.linspace(1.0, 0.35, rank)
    teacher = p @ np.diag(singular) @ q.T
    x = rng.normal(size=(512, d_in))
    y = x @ teacher.T
    w1 = 0.08 * rng.normal(size=(hidden, d_in))
    w2 = 0.08 * rng.normal(size=(d_out, hidden))
    v1 = np.zeros_like(w1)
    v2 = np.zeros_like(w2)
    rows, states = [], []
    for t in range(n):
        pred = x @ w1.T @ w2.T
        err = pred - y
        loss = 0.5 * float(np.mean(err * err))
        gw2 = (err.T @ x @ w1.T) / len(x)
        gw1 = ((err @ w2).T @ x) / len(x)
        v2 = beta * v2 + gw2
        v1 = beta * v1 + gw1
        w2 -= eta * v2
        w1 -= eta * v1
        if t % 5 == 0:
            rows.append((t, loss, np.linalg.norm(np.r_[w1.ravel(), w2.ravel()]),
                         np.linalg.norm(np.r_[v1.ravel(), v2.ravel()])))
            states.append(np.r_[w1.ravel(), w2.ravel(), v1.ravel(), v2.ravel()])
        if not np.isfinite(w1).all() or not np.isfinite(w2).all():
            break
    return np.asarray(rows), np.asarray(states)


def scalar_observers(rows: np.ndarray, states: np.ndarray, seed: int = 123):
    rng = np.random.default_rng(seed)
    projection = states @ rng.normal(size=states.shape[1])
    return {
        "loss": rows[:, 1],
        "state_norm": np.linalg.norm(states, axis=1),
        "velocity_norm": rows[:, 3],
        "random_projection": projection,
    }


def score_series(series: np.ndarray, cfg: EstimatorConfig, seed: int = 0):
    result = score(series, cfg, seed=seed)
    doubled = score(series, cfg.replace(max_E=2 * cfg.max_E), seed=seed)
    ident = (doubled["MG"] / result["MG"]
             if np.isfinite(doubled["MG"]) and np.isfinite(result["MG"])
             and result["MG"] > 0 else np.nan)
    trend = diagnose(series, cfg, seed=seed).trend_crossings
    return {
        "MG": result["MG"], "LB": result["LB"], "PRdelay": result["PRdelay"],
        "roughness": result["roughness"], "tau_used": result["tau_used"],
        "theiler_used": result["theiler_used"], "degenerate": result["degenerate"],
        "ident_ratio": ident,
        "trend_crossings": trend,
    }


def run_family(name, simulator, ranks, seeds, n, **kwargs):
    records = []
    for rank in ranks:
        for seed in seeds:
            rows, states = simulator(rank, n, seed, **kwargs)
            for observer, series in scalar_observers(rows, states).items():
                for tau in ("acorr", 4, 8):
                    cfg = EstimatorConfig(
                        max_E=6, tau=tau, k_neighbors=4,
                        theiler="embedding", theiler_cap=150,
                        window=min(500, len(series)), stride=min(500, len(series)))
                    try:
                        out = score_series(series, cfg, seed=seed)
                    except Exception as exc:
                        out = {"error": type(exc).__name__}
                    records.append({"family": name, "rank": rank, "seed": seed,
                                    "observer": observer, "tau": str(tau), **out})
    return pd.DataFrame(records)


def choose_lag_without_truth(frame: pd.DataFrame):
    """Select lag from stability/diagnostics only, then report the selected values."""
    records = []
    for keys, group in frame.groupby(["family", "rank", "seed", "observer"]):
        group = group.copy()
        finite = np.isfinite(group["MG"].astype(float))
        valid = finite & (~group["degenerate"].astype(bool))
        group["selection_score"] = (
            group["ident_ratio"].sub(1.0).abs().fillna(10.0)
            + 0.02 * group["trend_crossings"].fillna(100.0)
            + 10.0 * (~valid)
        )
        best = group.sort_values(["selection_score", "tau"]).iloc[0]
        records.append({**dict(zip(["family", "rank", "seed", "observer"], keys)),
                        "selected_tau": best["tau"],
                        "selected_MG": best["MG"],
                        "selected_ident_ratio": best["ident_ratio"],
                        "selected_crossings": best["trend_crossings"],
                        "selection_score": best["selection_score"]})
    return pd.DataFrame(records)


def main():
    frames = [
        run_family("matrix_factorization_recurrent", matrix_factorization,
                   ranks=(1, 2, 4), seeds=(0, 1), n=4000,
                   eta=0.02, beta=1.0),
        run_family("matrix_factorization_damped", matrix_factorization,
                   ranks=(1, 2, 4), seeds=(0, 1), n=4000,
                   eta=0.02, beta=0.9),
        run_family("teacher_student", teacher_student,
                   ranks=(1, 2, 4), seeds=(0, 1), n=3000,
                   eta=0.02, beta=1.0, hidden=8),
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT / "raw_scores.csv", index=False)
    selected = choose_lag_without_truth(raw)
    selected.to_csv(OUT / "automatic_lag_selection.csv", index=False)
    summary = (raw.groupby(["family", "observer", "tau"], dropna=False)
                  .agg(n=("MG", "count"), median_MG=("MG", "median"),
                       median_ident_ratio=("ident_ratio", "median"),
                       median_crossings=("trend_crossings", "median"),
                       frac_degenerate=("degenerate", "mean"))
                  .reset_index())
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps({
        "families": sorted(raw.family.unique().tolist()),
        "n_rows": int(len(raw)),
        "outputs": ["raw_scores.csv", "automatic_lag_selection.csv", "summary.csv"],
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nAutomatic lag selection:")
    print(selected.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

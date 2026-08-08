"""Known-dimension nonlinear ML task: online logistic regression, k=1..20.

A D=64 logistic regressor is trained with full-batch GD and binary
cross-entropy on one-hot inputs. Exactly k target logits vary with independent
quasiperiodic phases. Thus the driven optimisation dynamics has known
intrinsic dimension k, while both prediction and gradient are nonlinear.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

import estimators as E
import systems as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp12_logistic_regression_k20"
OUT.mkdir(parents=True, exist_ok=True)
KS = np.arange(1, 21)
D = 64
OBSERVERS = ("weight_fro", "weight_trace", "weight_projection",
             "gradient_fro", "gradient_projection", "loss",
             "probability_fro")

GRID = (
    dict(window=2000, cycles=500.0, max_E=31, kn=20, tau=1, eta=.5),
    dict(window=3000, cycles=800.0, max_E=31, kn=30, tau=2, eta=1.0),
    dict(window=4000, cycles=1200.0, max_E=41, kn=40, tau=2, eta=1.0),
    dict(window=4000, cycles=1800.0, max_E=41, kn=50, tau=4, eta=2.0),
)


def simulate(k: int, seed: int, cfg: dict) -> dict[str, np.ndarray]:
    W = int(cfg["window"])
    burn = 1500
    n = W + burn
    eta = float(cfg["eta"])
    rng = np.random.default_rng(51_000 + seed)
    t = np.arange(n, dtype=float)

    # The targets are probabilities, but their independent drivers live in
    # logit space. Amplitude 1.2 makes the sigmoid nonlinearity substantial.
    baseline = .4 * rng.standard_normal(D)
    target_logits = np.tile(baseline, (n, 1))
    freq = S.frequencies(k, float(cfg["cycles"]), W, seed=seed,
                         band_mode="matched")
    phase = rng.uniform(0, 2*np.pi, k)
    amp = 1.2 * (.7 + .3*rng.random(k))
    target_logits[:, :k] += amp*np.sin(2*np.pi*t[:, None]*freq[None, :]
                                      + phase[None, :])
    targets = expit(target_logits)

    # For one-hot inputs, model prediction for example i is sigmoid(w_i).
    # BCE gradient with respect to logit w_i is sigmoid(w_i)-target_i.
    weights = np.empty_like(targets)
    gradients = np.empty_like(targets)
    probabilities = np.empty_like(targets)
    losses = np.empty(n)
    w = baseline.copy()
    for j in range(n):
        p = expit(w)
        g = p-targets[j]
        weights[j], gradients[j], probabilities[j] = w, g, p
        losses[j] = np.sum(np.logaddexp(0.0, w)-targets[j]*w)
        w = w-eta*g

    weights = weights[burn:]
    gradients = gradients[burn:]
    probabilities = probabilities[burn:]
    losses = losses[burn:]
    rw = np.random.default_rng(52_000+seed).standard_normal(D)
    rg = np.random.default_rng(53_000+seed).standard_normal(D)
    raw = {
        "weight_fro": np.linalg.norm(weights, axis=1),
        "weight_trace": weights.sum(axis=1),
        "weight_projection": weights@rw,
        "gradient_fro": np.linalg.norm(gradients, axis=1),
        "gradient_projection": gradients@rg,
        "loss": losses,
        "probability_fro": np.linalg.norm(probabilities, axis=1),
    }
    nrng = np.random.default_rng(54_000+seed)
    result = {}
    for name, x in raw.items():
        z = (x-x.mean())/(x.std()+1e-15)
        result[name] = z+1e-6*nrng.standard_normal(W)
    return result


def one_job(cfg_id: int, k: int, seed: int) -> list[dict]:
    cfg = GRID[cfg_id]
    rows = []
    for name, x in simulate(k, seed, cfg).items():
        value = E.mg(x, max_E=int(cfg["max_E"]), k=int(cfg["kn"]),
                     tau=int(cfg["tau"]))
        rows.append(dict(cfg_id=cfg_id, seed=seed, k=k, observer=name,
                         MG=float(value), error=float(value-k), **cfg))
    return rows


def score(g: pd.DataFrame) -> dict:
    g = g.sort_values("k")
    err = g.MG.to_numpy()-g.k.to_numpy()
    inv = int(np.sum(np.diff(g.MG.to_numpy()) <= 0))
    return dict(mae=float(np.mean(np.abs(err))),
                max_error=float(np.max(np.abs(err))),
                rho=float(spearmanr(g.k, g.MG).statistic),
                inversions=inv,
                objective=float(np.mean(np.abs(err))+.15*np.max(np.abs(err))+.15*inv))


def run_jobs(jobs, workers=4):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(one_job, *job) for job in jobs]
        for i, f in enumerate(as_completed(fs), 1):
            rows.extend(f.result())
            if i % 20 == 0:
                print(f"completed {i}/{len(fs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    start = time.time()
    cal = run_jobs([(c, int(k), 0) for c in range(len(GRID)) for k in KS])
    cal.to_csv(OUT/"calibration_raw.csv", index=False)
    summaries = []
    for (obs, cfg), g in cal.groupby(["observer", "cfg_id"]):
        summaries.append(dict(observer=obs, cfg_id=cfg, **score(g)))
    cs = pd.DataFrame(summaries).sort_values(["observer", "objective"])
    cs.to_csv(OUT/"calibration_summary.csv", index=False)
    best = cs.groupby("observer", as_index=False).first()
    best.to_csv(OUT/"best_by_observer.csv", index=False)

    configs = sorted(set(int(x) for x in best.cfg_id))
    held = run_jobs([(c, int(k), seed) for c in configs for seed in (1, 2) for k in KS])
    keep = {(r.observer, int(r.cfg_id)) for _, r in best.iterrows()}
    held = held[[((r.observer, int(r.cfg_id)) in keep) for _, r in held.iterrows()]]
    held.to_csv(OUT/"heldout_raw.csv", index=False)
    hs = []
    for (obs, seed), g in held.groupby(["observer", "seed"]):
        hs.append(dict(observer=obs, seed=seed, **score(g)))
    hs = pd.DataFrame(hs)
    hs.to_csv(OUT/"heldout_summary.csv", index=False)
    ranking = (hs.groupby("observer", as_index=False)
                 .agg(MAE=("mae", "mean"), max_error=("max_error", "max"),
                      rho=("rho", "mean"), inversions=("inversions", "mean"))
                 .sort_values("MAE"))
    ranking.to_csv(OUT/"observer_ranking.csv", index=False)

    med = held.groupby(["observer", "k"], as_index=False).MG.median()
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, name in zip(axes.flat, OBSERVERS):
        g = med[med.observer == name]
        ax.plot(KS, KS, "k--", lw=1)
        ax.plot(g.k, g.MG, "o-", ms=3.2, lw=1.2, color="#28669b")
        ax.set(title=name, xlabel="True k", ylabel="MG estimate")
        ax.grid(alpha=.22)
    for ax in axes.flat[len(OBSERVERS):]:
        ax.axis("off")
    fig.suptitle("Nonlinear logistic regression: held-out recovery", fontsize=14)
    fig.savefig(OUT/"heldout_observers_k20.png", dpi=220)
    fig.savefig(OUT/"heldout_observers_k20.pdf")
    plt.close(fig)

    winner = ranking.iloc[0].observer
    wm = med[med.observer == winner]
    report = [
        "# Нелинейная логистическая регрессия с известной размерностью", "",
        "## Постановка", "",
        "Логистическая модель размерности `D=64` обучалась полным градиентным спуском "
        "с binary cross-entropy на one-hot входах. Ровно `k` целевых логитов менялись "
        "квазипериодически и независимо; остальные были постоянны. Поэтому истинная "
        "размерность активной динамики равна `k`.", "",
        "Проверены `k=1,...,20`, семь одномерных наблюдателей и четыре набора параметров "
        "MG. Параметры выбирались на seed 0 сразу для всех `k`; seeds 1 и 2 оставались "
        "тестовыми. Коррекция отдельно для каждого `k` не применялась.", "",
        "## Результаты", "",
        "| Наблюдатель | MAE | Макс. ошибка | Spearman rho | Инверсии |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in ranking.iterrows():
        report.append(f"| `{r.observer}` | {r.MAE:.3f} | {r.max_error:.3f} | "
                      f"{r.rho:.3f} | {r.inversions:.1f} |")
    report += ["", f"Лучший наблюдатель: **`{winner}`**.", "",
               "| Истинное k | Медиана MG | Ошибка |", "|---:|---:|---:|"]
    for _, r in wm.iterrows():
        report.append(f"| {int(r.k)} | {r.MG:.3f} | {r.MG-r.k:+.3f} |")
    report += ["", "![Сравнение наблюдателей](heldout_observers_k20.png)", "",
               "## Вывод", "",
               "MG по одномерным логам сохраняет информацию о числе активных степеней "
               "свободы и в нелинейной задаче с BCE. Абсолютную точность и насыщение "
               "при больших `k` следует оценивать по таблице, а не скрывать калибровкой.", "",
               f"Время расчёта: {time.time()-start:.1f} с."]
    # UTF-8 BOM matches the project's Russian Markdown files in VS Code.
    (OUT/"final_report.md").write_text("\n".join(report), encoding="utf-8-sig")
    print(ranking.to_string(index=False))
    print("outputs", OUT)


if __name__ == "__main__":
    main()

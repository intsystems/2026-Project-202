"""Known-dimension nonlinear decoder experiment, k=1..20.

A latent vector z in R^k is optimised by backpropagation through a frozen
nonlinear MLP decoder F.  The time-dependent target is F(z_star(t)), where the
k coordinates of z_star are driven by independent quasiperiodic phases.
Consequently the optimiser has exactly k available coordinates and the driven
state has known intrinsic dimension k.  The decoder includes a linear skip to
keep its Jacobian full-rank while a tanh branch supplies genuine nonlinearity.
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
from scipy.stats import spearmanr

import estimators as E
import systems as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "exp13_frozen_nonlinear_decoder_k20"
OUT.mkdir(parents=True, exist_ok=True)

KS = np.arange(1, 21)
HIDDEN = 64
OUTPUT = 64
NONLINEAR_SCALE = .45
OBSERVERS = ("latent_fro", "latent_projection", "gradient_fro",
             "gradient_projection", "output_fro", "output_projection", "loss")

GRID = (
    dict(window=2000, cycles=300.0, max_E=31, kn=20, tau=1, eta=.08),
    dict(window=3000, cycles=600.0, max_E=31, kn=30, tau=2, eta=.08),
    dict(window=4000, cycles=1000.0, max_E=41, kn=40, tau=2, eta=.08),
    dict(window=4000, cycles=1500.0, max_E=41, kn=50, tau=4, eta=.05),
)


def decoder_parameters(k: int, seed: int):
    """Frozen decoder parameters; A has orthonormal columns."""
    rng = np.random.default_rng(61_000 + 101*k + seed)
    A, _ = np.linalg.qr(rng.standard_normal((OUTPUT, k)), mode="reduced")
    B = rng.standard_normal((HIDDEN, k))/np.sqrt(k)
    C = rng.standard_normal((OUTPUT, HIDDEN))/np.sqrt(HIDDEN)
    bias = .35*rng.standard_normal(HIDDEN)
    return A, B, C, bias


def forward_and_jacobian(z, A, B, C, bias):
    h = np.tanh(B@z+bias)
    y = A@z + NONLINEAR_SCALE*(C@h)
    # Jacobian dy/dz, needed for exact backpropagation.
    J = A + NONLINEAR_SCALE*((C*(1.0-h*h)[None, :])@B)
    return y, J


def simulate(k: int, seed: int, cfg: dict):
    W = int(cfg["window"])
    burn = 1500
    n = W+burn
    eta = float(cfg["eta"])
    rng = np.random.default_rng(62_000+seed)
    A, B, C, bias = decoder_parameters(k, seed)
    t = np.arange(n, dtype=float)
    freq = S.frequencies(k, float(cfg["cycles"]), W, seed=seed,
                         band_mode="matched")
    phase = rng.uniform(0, 2*np.pi, k)
    amp = .75*(.7+.3*rng.random(k))
    centre = .25*rng.standard_normal(k)
    z_star = centre[None, :] + amp[None, :]*np.sin(
        2*np.pi*t[:, None]*freq[None, :]+phase[None, :])

    # Targets are produced once by the same fixed nonlinear decoder.
    targets = np.empty((n, OUTPUT))
    for j in range(n):
        targets[j] = forward_and_jacobian(z_star[j], A, B, C, bias)[0]

    latent = np.empty((n, k))
    gradient = np.empty((n, k))
    outputs = np.empty((n, OUTPUT))
    losses = np.empty(n)
    z = centre.copy()
    for j in range(n):
        y, J = forward_and_jacobian(z, A, B, C, bias)
        residual = y-targets[j]
        grad = J.T@residual                         # exact backprop through F
        latent[j], gradient[j], outputs[j] = z, grad, y
        losses[j] = .5*(residual@residual)
        z = z-eta*grad

    latent, gradient = latent[burn:], gradient[burn:]
    outputs, losses = outputs[burn:], losses[burn:]
    rz = np.random.default_rng(63_000+seed).standard_normal(k)
    rg = np.random.default_rng(64_000+seed).standard_normal(k)
    ry = np.random.default_rng(65_000+seed).standard_normal(OUTPUT)
    raw = {
        "latent_fro": np.linalg.norm(latent, axis=1),
        "latent_projection": latent@rz,
        "gradient_fro": np.linalg.norm(gradient, axis=1),
        "gradient_projection": gradient@rg,
        "output_fro": np.linalg.norm(outputs, axis=1),
        "output_projection": outputs@ry,
        "loss": losses,
    }
    nrng = np.random.default_rng(66_000+seed)
    out = {}
    for name, x in raw.items():
        zscore = (x-x.mean())/(x.std()+1e-15)
        out[name] = zscore+1e-6*nrng.standard_normal(W)
    return out


def one_job(cfg_id: int, k: int, seed: int):
    cfg = GRID[cfg_id]
    rows = []
    for name, x in simulate(k, seed, cfg).items():
        value = E.mg(x, max_E=int(cfg["max_E"]), k=int(cfg["kn"]),
                     tau=int(cfg["tau"]))
        rows.append(dict(cfg_id=cfg_id, seed=seed, k=k, observer=name,
                         MG=float(value), error=float(value-k), **cfg))
    return rows


def score(g):
    g = g.sort_values("k")
    error = g.MG.to_numpy()-g.k.to_numpy()
    inversions = int(np.sum(np.diff(g.MG.to_numpy()) <= 0))
    return dict(mae=float(np.mean(np.abs(error))),
                max_error=float(np.max(np.abs(error))),
                rho=float(spearmanr(g.k, g.MG).statistic),
                inversions=inversions,
                objective=float(np.mean(np.abs(error))+.15*np.max(np.abs(error))
                                +.15*inversions))


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
    rows = []
    for (observer, cfg_id), g in cal.groupby(["observer", "cfg_id"]):
        rows.append(dict(observer=observer, cfg_id=cfg_id, **score(g)))
    cs = pd.DataFrame(rows).sort_values(["observer", "objective"])
    cs.to_csv(OUT/"calibration_summary.csv", index=False)
    best = cs.groupby("observer", as_index=False).first()
    best.to_csv(OUT/"best_by_observer.csv", index=False)

    configs = sorted(set(int(v) for v in best.cfg_id))
    held = run_jobs([(c, int(k), seed) for c in configs for seed in (1, 2) for k in KS])
    keep = {(r.observer, int(r.cfg_id)) for _, r in best.iterrows()}
    held = held[[((r.observer, int(r.cfg_id)) in keep) for _, r in held.iterrows()]]
    held.to_csv(OUT/"heldout_raw.csv", index=False)
    rows = []
    for (observer, seed), g in held.groupby(["observer", "seed"]):
        rows.append(dict(observer=observer, seed=seed, **score(g)))
    hs = pd.DataFrame(rows)
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
    fig.suptitle("Frozen nonlinear decoder: held-out recovery", fontsize=14)
    fig.savefig(OUT/"heldout_observers_k20.png", dpi=220)
    fig.savefig(OUT/"heldout_observers_k20.pdf")
    plt.close(fig)

    winner = ranking.iloc[0].observer
    wm = med[med.observer == winner]
    report = [
        "# Замороженный нелинейный декодер с известной размерностью", "",
        "## Постановка", "",
        "Обучаемый вектор `z` размерности `k` оптимизировался через замороженный "
        "нелинейный MLP-декодер `F(z)`. Целью служил `F(z*(t))`, где все `k` "
        "координат `z*` менялись независимо. Градиент вычислялся точным "
        "backpropagation через tanh-слой.", "",
        "Линейный skip в декодере сохранял полный ранг отображения, поэтому доступное "
        "пространство оптимизации и истинная динамическая размерность равны `k`. "
        "Проверены `k=1,...,20`, семь одномерных логов и четыре конфигурации MG.", "",
        "Конфигурации выбирались на seed 0 сразу для всех `k`; seeds 1 и 2 были "
        "отложены для теста. Подгонка отдельно под каждое `k` не применялась.", "",
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
               "Эксперимент проверяет восстановление известной размерности после "
               "нелинейного смешивания и backpropagation. Итоговую точность следует "
               "оценивать по held-out результатам выше.", "",
               f"Время расчёта: {time.time()-start:.1f} с."]
    (OUT/"final_report.md").write_text("\n".join(report), encoding="utf-8-sig")
    print(ranking.to_string(index=False))
    print("outputs", OUT)


if __name__ == "__main__":
    main()

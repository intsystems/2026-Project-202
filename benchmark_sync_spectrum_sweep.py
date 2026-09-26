"""Same-seed spectrum/MG sweep across the synchronization transition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.windows import score
from benchmark_sync_computation import complete_graph, full_spectrum, rollout

OUT = ROOT / "research_sync_control_results"


def scalar_log(sensor: np.ndarray, seed: int, window: int = 4096):
    cfg = EstimatorConfig(max_E=12, tau=20, k_neighbors=8,
                          theiler=700, theiler_cap=700,
                          window=window, stride=window)
    result = score(sensor[-window:], cfg, seed=seed)
    return {"MG": float(result["MG"]), "LB": float(result["LB"]),
            "degenerate": bool(result["degenerate"]),
            "PRdelay": float(result["PRdelay"])}


def kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """Kaplan--Yorke dimension of a finite-time Lyapunov spectrum."""
    values = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    cumulative = np.cumsum(values)
    nonnegative = np.flatnonzero(cumulative >= 0)
    if len(nonnegative) == 0:
        return 0.0
    j = int(nonnegative[-1])
    if j == len(values) - 1:
        return float(len(values))
    denom = abs(values[j + 1])
    return float(j + 1 + cumulative[j] / denom) if denom else float(j + 1)


def main():
    n = 64
    seed = 474
    steps = 8192
    dt = 0.03
    gains = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                      0.60, 0.80, 1.00, 1.30, 1.80])
    # One fixed plant, one fixed initial condition, and one fixed scalar sensor.
    rng = np.random.default_rng(seed)
    omega = 1.0 + 0.20 * rng.normal(size=n)
    theta0 = rng.uniform(-np.pi, np.pi, size=n)
    sensor_a = rng.normal(size=n); sensor_b = rng.normal(size=n)
    sensor_a /= np.linalg.norm(sensor_a); sensor_b /= np.linalg.norm(sensor_b)
    A = complete_graph(n)

    spectra, records = [], []
    for gain in gains:
        # rollout() is not used here because we need the identical initial state
        # and identical sensor for every point on the sweep.
        theta = theta0.copy()
        state = np.empty((steps, n), dtype=np.float64)
        sensor = np.empty(steps, dtype=np.float64)
        kappa = 0.10 + gain
        for t in range(steps):
            diff = theta[None, :] - theta[:, None]
            theta += dt * (omega + kappa * (A * np.sin(diff)).sum(1))
            state[t] = theta
            sensor[t] = (sensor_a @ np.sin(theta) + sensor_b @ np.cos(theta)) / np.sqrt(n)
        spectrum = full_spectrum(state, gain, A, dt, burn=1024, qr_stride=4)
        scalar = scalar_log(sensor, seed=seed, window=4096)
        order = np.abs(np.exp(1j * state).mean(axis=1))
        spectra.append(spectrum)
        records.append({
            "seed": seed, "n": n, "gain": gain,
            "MG": scalar["MG"], "LB": scalar["LB"],
            "PRdelay": scalar["PRdelay"], "degenerate": scalar["degenerate"],
            "lambda_max": float(np.max(spectrum)),
            "lambda_second": float(np.sort(spectrum)[-2]),
            "lambda_sum": float(np.sum(spectrum)),
            "kaplan_yorke": kaplan_yorke_dimension(spectrum),
            "n_positive": int(np.sum(spectrum > 1e-4)),
            "n_weak": int(np.sum(spectrum > -1e-4)),
            "order_mean": float(order[-4096:].mean()),
            "order_sd": float(order[-4096:].std()),
        })

    spectra = np.asarray(spectra)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "same_seed_spectrum_MG_sweep.csv", index=False)
    np.save(OUT / "same_seed_lyapunov_spectra.npy", spectra)
    (OUT / "same_seed_spectrum_MG_metadata.json").write_text(json.dumps({
        "seed": seed, "n": n, "steps": steps, "dt": dt,
        "gains": gains.tolist(),
        "fixed_across_sweep": ["omega", "initial phases", "scalar sensor", "graph"],
        "spectrum": "full tangent-space Lyapunov spectrum with QR reorthogonalization",
        "scalar_method": "MG/LB on the same fixed scalar projection",
    }, indent=2), encoding="utf-8")

    # One figure: full spectrum and scalar MG share the same horizontal axis.
    # The spectrum is sorted at each gain; this displays the ordered finite-time
    # spectrum and avoids pretending that crossing exponents have fixed identities.
    sorted_spectrum = np.sort(spectra, axis=1)[:, ::-1].T
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    extent = [gains[0], gains[-1], 1, n]
    im = ax.imshow(sorted_spectrum, origin="upper", aspect="auto", extent=extent,
                   cmap="coolwarm", vmin=-0.08, vmax=0.08,
                   interpolation="nearest")
    ax.axhline(0, color="black", lw=0.8, alpha=.35)
    ax.set_ylabel("ordered Lyapunov direction")
    ax.set_title("Same seed: full finite-time Lyapunov spectrum and scalar-log MG")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Lyapunov exponent")
    ax2.plot(gains, frame.MG, color="black", marker="o", linewidth=2.2,
             label="MG from the same scalar log")
    ax2.plot(gains, frame.n_positive, color="#d73027", marker="s", linestyle="--",
             linewidth=1.6, label=r"number of $\lambda_i>10^{-4}$")
    ax2.plot(gains, frame.n_weak, color="#1a9850", marker="^", linestyle="-.",
             linewidth=1.6, label=r"number of $\lambda_i>-10^{-4}$")
    ax2.plot(gains, frame.kaplan_yorke, color="#756bb1", marker="D", linestyle=":",
             linewidth=1.6, label="Kaplan--Yorke dimension")
    ax2.set_xlabel("coupling gain $g$ (one fixed plant, initial state, and seed)")
    ax2.set_ylabel("MG / spectrum summaries")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=.2)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "same_seed_spectrum_MG_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT / "same_seed_spectrum_MG_sweep.png", dpi=240, bbox_inches="tight")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

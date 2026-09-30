"""Independent full-vector reference for functional trajectory simplification.

The reference uses the complete logits on a fixed probe subset, never the scalar
logs consumed by MG.  For each local trajectory window it computes the singular
value spectrum and reports scale-free effective-rank quantities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def spectrum(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 1e-30:
        return dict(pr=0.0, rank90=0, rank99=0, sigma1=0.0, total_energy=0.0)
    p = energy / total
    cumulative = np.cumsum(p)
    return dict(
        pr=float(1.0 / np.sum(p * p)),
        rank90=int(np.searchsorted(cumulative, 0.90) + 1),
        rank99=int(np.searchsorted(cumulative, 0.99) + 1),
        sigma1=float(singular[0]),
        total_energy=total,
    )


def analyze(root: Path, window: int, stride: int) -> None:
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        run_dir = metadata_path.parent
        meta = json.loads(metadata_path.read_text())
        logits_path = run_dir / "reference_logits.npy"
        steps_path = run_dir / "reference_steps.npy"
        if not logits_path.exists() or not steps_path.exists():
            print(f"Skipping {run_dir.name}: reference snapshots are missing")
            continue

        logits = np.load(logits_path, mmap_mode="r")
        steps = np.load(steps_path)
        if len(logits) < window:
            print(f"Skipping {run_dir.name}: only {len(logits)} snapshots")
            continue

        switch = meta.get("switch")
        if switch is None:
            switch = meta.get("steps", int(steps[-1] + 1)) // 2

        for left in range(0, len(steps) - window + 1, stride):
            right = left + window
            local = spectrum(logits[left:right])
            start_step = int(steps[left])
            end_step = int(steps[right - 1])
            if end_step < switch:
                phase = "pre"
            elif start_step > switch:
                phase = "post"
            else:
                phase = "transition"
            rows.append(dict(
                run=run_dir.name,
                arm=meta.get("arm"),
                mode=meta.get("mode"),
                seed=meta.get("seed"),
                start_step=start_step,
                end_step=end_step,
                phase=phase,
                snapshots=window,
                **local,
            ))

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "reference_spectrum.csv", index=False)
    summary = (
        frame[frame.phase.isin(["pre", "post"])]
        .groupby(["run", "arm", "mode", "seed", "phase"], as_index=False)
        .agg(
            pr=("pr", "median"),
            rank90=("rank90", "median"),
            rank99=("rank99", "median"),
            sigma1=("sigma1", "median"),
            total_energy=("total_energy", "median"),
            windows=("pr", "size"),
        )
    )
    summary.to_csv(root / "reference_summary.csv", index=False)

    if len(summary):
        effects = summary.pivot_table(
            index=["run", "arm", "mode", "seed"],
            columns="phase",
            values=["pr", "rank90", "rank99", "sigma1", "total_energy"],
        ).reset_index()
        effects.columns = [
            "_".join(str(v) for v in col if str(v) != "")
            if isinstance(col, tuple) else str(col)
            for col in effects.columns
        ]
        if "pr_pre" in effects and "pr_post" in effects:
            effects["pr_post_pre_ratio"] = effects["pr_post"] / effects["pr_pre"]
        if "rank99_pre" in effects and "rank99_post" in effects:
            effects["rank99_post_pre_difference"] = effects["rank99_post"] - effects["rank99_pre"]
        effects.to_csv(root / "reference_effects.csv", index=False)

    make_plot(root, frame)
    print(f"Wrote {root / 'reference_summary.csv'}")


def make_plot(root: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for run, group in frame.groupby("run"):
        label = run.replace("adam_", "")
        axes[0].plot(group.end_step, group.pr, marker="o", ms=2.5, lw=0.8, label=label)
        axes[1].plot(group.end_step, group.rank99, marker="o", ms=2.5, lw=0.8, label=label)
    for ax in axes:
        ax.axvline(1200, color="gray", ls=":", lw=1)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Function-space PR")
    axes[1].set_ylabel("Rank for 99% variance")
    axes[1].set_xlabel("Optimizer step")
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(root / "reference_spectrum.png", dpi=180)
    fig.savefig(root / "reference_spectrum.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    analyze(args.root, args.window, args.stride)

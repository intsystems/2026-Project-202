"""Replay self-supervised encoder selection on the five evaluation seeds."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def select(data, rule):
    per_seed = []
    for _, candidates in data.groupby("seed"):
        scores = rule.sign * candidates[rule.col].to_numpy()
        finite = np.isfinite(scores)
        if finite.sum() < 5:
            continue
        scores[~finite] = scores[finite].min() - 1
        order = np.argsort(-scores)
        accuracy = candidates["acc"].to_numpy()
        best = accuracy.max()
        per_seed.append([
            spearmanr(scores, accuracy).statistic,
            best - accuracy[order[0]],
            best - accuracy[order[:3]].max(),
        ])
    return np.nanmean(per_seed, axis=0)


def main():
    merged = pd.concat([pd.read_csv(HERE / name) for name in ("test_final.csv", "confirm_final.csv")])
    rules = pd.read_csv(HERE / "T1_config_ranking.csv").dropna(subset=["col"])
    metrics = ["rho", "regret1", "regret3"]
    report = pd.DataFrame(
        [select(merged, rule) for rule in rules.itertuples()],
        index=rules["stat"],
        columns=metrics,
    )
    saved = pd.read_csv(HERE / "T1_confirm.csv")
    saved = saved.query("set == 'pooled_10_12_20_21'").set_index("stat")
    np.testing.assert_allclose(report, saved.loc[report.index, metrics], rtol=1e-10, atol=1e-14)
    print(f"MG accuracy gap: {report.loc['MG', 'regret1'] * 100:.3f} percentage points")


if __name__ == "__main__":
    main()

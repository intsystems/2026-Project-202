"""Replay the published transformer loss-spike warnings on the saved windows."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
FIRST_STEP = 1500
STRIDE = 50
HORIZON = 500


def scores(series, rule):
    if rule.fam == "level":
        return rule.sign * series
    previous = pd.concat([series.shift(HORIZON // STRIDE + offset) for offset in range(rule.B)], axis=1)
    previous.iloc[: HORIZON // STRIDE + rule.B - 1] = np.nan
    return rule.sign * (series - previous.median(axis=1))


def evaluate(runs, onsets, rule):
    hits = spikes = alarms = steps = 0
    labels = []
    values = []
    leads = []
    for run, data in runs:
        if int(run.rsplit("_s", 1)[1]) not in (10, 11, 12, 13):
            continue
        t = data["t"].to_numpy()
        events = np.asarray(onsets[run], dtype=int)
        since_event = t[:, None] - events
        scored = (t >= FIRST_STEP) & ~((since_event >= 0) & (since_event <= HORIZON)).any(axis=1)
        warning = (since_event >= -HORIZON) & (since_event < 0)
        positive = warning.any(axis=1)
        score = scores(data[rule.col], rule).to_numpy()
        alarm = scored & (score >= rule.theta)
        steps += scored.sum() * STRIDE
        for event, event_window in zip(events, warning.T):
            if event > FIRST_STEP:
                spikes += 1
                detected = t[alarm & event_window]
                if detected.size:
                    hits += 1
                    leads.append(event - detected[0])
        last_alarm = -HORIZON
        for step in t[alarm & ~positive]:
            if step - last_alarm >= HORIZON:
                alarms += 1
                last_alarm = step
        valid = scored & np.isfinite(score)
        labels.extend(positive[valid])
        values.extend(score[valid])
    return {
        "test_hits": f"{hits}/{spikes}",
        "test_fa10k": alarms / steps * 10000,
        "test_lead": float(np.median(leads)) if leads else np.nan,
        "test_AUROC": roc_auc_score(labels, values),
        "test_AP": average_precision_score(labels, values),
    }


def main():
    onsets = json.loads((HERE / "onsets.json").read_text(encoding="utf-8"))["onsets"]
    records = pd.read_csv(HERE / "window_scores.csv.gz").sort_values(["run", "t"])
    rules = pd.read_csv(HERE / "results_cal_percolumn.csv").set_index(["budget", "col"], drop=False)
    outcomes = pd.read_csv(HERE / "results_test.csv")
    rows = []
    for row in outcomes.itertuples():
        rule = rules.loc[row.budget, row.column]
        row_scores = evaluate(records.groupby("run"), onsets, rule)
        rows.append({"budget": row.budget, "method": row.method, **row_scores})
    report = pd.DataFrame(rows)
    np.testing.assert_array_equal(report["test_hits"], outcomes["test_hits"])
    numeric = ["test_fa10k", "test_lead", "test_AUROC", "test_AP"]
    np.testing.assert_allclose(report[numeric], outcomes[numeric], rtol=1e-10)
    print(report.query("budget == 2").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

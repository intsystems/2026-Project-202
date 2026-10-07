from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
from extra_baselines import STATS, scalar_feature, W, S, WARM, HORIZON, score_series

OUT = H / 'new_results'
LOGS = OUT / 'fresh_cnn'
rules = json.loads((OUT / 'rules.json').read_text())
events = {int(k): int(v) for k, v in json.loads((OUT / 'event_assignment.json').read_text()).items()}

def features():
    rows = []
    for path in sorted(LOGS.glob('logs_*.npz')):
        arm, seed = path.stem[5:].rsplit('_s', 1)
        seed = int(seed)
        x = np.load(path)['param_norm'].astype(float)
        for start in range(0, len(x) - W + 1, S):
            seg = x[start:start + W]
            row = {stat: scalar_feature(seg, stat) for stat in STATS}
            row.update(arm=arm, seed=seed, start=start, end=start + W,
                       event=events[seed])
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'fresh_cnn_features.csv', index=False)
    return result

def first_alarm(group, stat, rule):
    for end, value in score_series(group.sort_values('start'), stat, rule):
        if value > rule['delta']:
            return int(end)
    return None

def main():
    frame = features()
    rows = []
    for (arm, seed), group in frame.groupby(['arm', 'seed']):
        event = int(group.event.iloc[0])
        for stat in STATS:
            rule = rules[stat + '_block']
            alarm = first_alarm(group, stat, rule)
            if arm == 'base':
                hit = False
                false_alarm = alarm is not None
                delay = np.nan
            else:
                hit = alarm is not None and event < alarm <= event + HORIZON
                false_alarm = alarm is not None and alarm <= event
                delay = alarm - event if hit else np.nan
            rows.append(dict(arm=arm, seed=seed, stat=stat, event=event,
                             alarm=alarm, hit=hit, false_alarm=false_alarm,
                             delay=delay))
    records = pd.DataFrame(rows)
    records.to_csv(OUT / 'fresh_cnn_records.csv', index=False)
    summary = []
    for stat, group in records.groupby('stat'):
        interventions = group[group.arm != 'base']
        summary.append(dict(stat=stat, hits=int(interventions.hit.sum()),
                            n=len(interventions), false_alarms=int(group.false_alarm.sum()),
                            delay=float(interventions.loc[interventions.hit, 'delay'].median())
                            if interventions.hit.any() else np.nan))
    summary = pd.DataFrame(summary).sort_values('stat')
    summary.to_csv(OUT / 'fresh_cnn_summary.csv', index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()

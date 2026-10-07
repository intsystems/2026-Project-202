from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parent
summary = pd.read_csv(H / 'new_results' / 'baseline_summary.csv')
order = ['MG', 'abs_diff', 'norm_diff', 'perm_entropy', 'sample_entropy',
         'spectral_entropy', 'crossings', 'lag1', 'det_std', 'level']
labels = {
    'MG': 'MG', 'abs_diff': 'Abs. increments', 'norm_diff': 'Norm. increments',
    'perm_entropy': 'Perm. entropy', 'sample_entropy': 'Sample entropy',
    'spectral_entropy': 'Spectral entropy', 'crossings': 'Trend crossings',
    'lag1': 'Lag-one corr.', 'det_std': 'Detrended std.', 'level': 'Level'
}

block = summary[summary.detector == 'block'].copy()
block['order'] = block.stat.map({name: i for i, name in enumerate(order)})
block = block.sort_values('order')
lines = [r'\begin{tabular}{lrrr}', r'\toprule',
         r'Feature & Hits & False positives & Median delay \\', r'\midrule']
for _, row in block.iterrows():
    delay = '---' if pd.isna(row.delay) else f'{int(round(row.delay)):,}'
    alarms = int(row.alarms_base + row.alarms_batch_up + row.alarms_scale + row.alarms_smooth)
    lines.append(f'{labels[row.stat]} & {int(row.hits)}/{int(row.n)} & {alarms}/16 & {delay} ' + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(H / 'tables' / 'extra_baselines.tex').write_text('\n'.join(lines), encoding='utf-8')

by_arm = pd.read_csv(H / 'new_results' / 'baseline_by_arm.csv')
by_arm = by_arm[by_arm.detector == 'block']
pivot = by_arm.pivot(index='stat', columns='arm', values='alarms').fillna(0)
pivot = pivot.reindex(order).fillna(0)
arms = ['base', 'batch_up', 'scale', 'smooth']
lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
         r'Feature & Base & Batch $\times4$ & Rescaled & Smoothed \\', r'\midrule']
for stat, row in pivot.iterrows():
    values = ' & '.join(str(int(row.get(arm, 0))) for arm in arms)
    lines.append(f'{labels[stat]} & {values} ' + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(H / 'tables' / 'baseline_controls.tex').write_text('\n'.join(lines), encoding='utf-8')

bib = H / 'references.bib'
text = bib.read_text(encoding='utf-8')
entries = r'''
@article{bandt2002permutation,
  title={Permutation entropy: a natural complexity measure for time series},
  author={Bandt, Christoph and Pompe, Bernd},
  journal={Physical Review Letters}, volume={88}, number={17}, pages={174102}, year={2002},
  doi={10.1103/PhysRevLett.88.174102}
}

@article{richman2000physiological,
  title={Physiological time-series analysis using approximate entropy and sample entropy},
  author={Richman, Joshua S. and Moorman, J. R.},
  journal={American Journal of Physiology--Heart and Circulatory Physiology},
  volume={278}, number={6}, pages={H2039--H2049}, year={2000},
  doi={10.1152/ajpheart.2000.278.6.H2039}
}

@article{page1954continuous,
  title={Continuous inspection schemes}, author={Page, E. S.},
  journal={Biometrika}, volume={41}, number={1--2}, pages={100--115}, year={1954},
  doi={10.1093/biomet/41.1-2.100}
}
'''
if 'bandt2002permutation' not in text:
    bib.write_text(text.rstrip() + '\n' + entries, encoding='utf-8')

from review_analysis import main as review_tables
review_tables()

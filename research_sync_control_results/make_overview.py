from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / 'summary.csv')
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
order = ['before', 'mid', 'after']
labels = ['Before training', 'Mid training', 'After training']
colors = ['#b2182b', '#ef8a62', '#2166ac']

for i, stage in enumerate(order):
    rows = d[d.stage == stage]
    ax[0].scatter(np.repeat(i, len(rows)), rows.sensor_MG, s=35, color=colors[i])
    ax[0].plot([i], [rows.sensor_MG.median()], 'k_', ms=13)
    ax[1].scatter(np.repeat(i, len(rows)), rows.state_PR, s=35, color=colors[i])
    ax[1].plot([i], [rows.state_PR.median()], 'k_', ms=13)
    ax[2].scatter(np.repeat(i, len(rows)), rows.order_mean, s=35, color=colors[i])
    ax[2].plot([i], [rows.order_mean.median()], 'k_', ms=13)

for a, title, ylabel in zip(ax,
    ['Scalar-log estimate', 'Full-state participation ratio', 'Synchrony order parameter'],
    ['MG', 'PR', 'R']):
    a.set_xticks(range(3), labels, rotation=25, ha='right')
    a.set_title(title, fontsize=9)
    a.set_ylabel(ylabel)
    a.grid(alpha=.25, axis='y')
fig.suptitle('Learned synchronization: independent evidence of dynamical simplification', fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / 'overview.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'overview.png', dpi=220, bbox_inches='tight')

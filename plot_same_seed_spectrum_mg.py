"""Make a direct same-seed Lyapunov-spectrum/MG comparison figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_sync_control_results'
df = pd.read_csv(OUT / 'same_seed_spectrum_MG_sweep.csv')
spectra = np.load(OUT / 'same_seed_lyapunov_spectra.npy')
gains = df.gain.to_numpy()

fig, ax = plt.subplots(figsize=(11, 6.2))

# Every coloured line is one Lyapunov exponent tracked across the same gain sweep.
# The rank is sorted by its average value only for visual continuity; the plotted
# values themselves are the complete spectrum at each gain.
order = np.argsort(np.mean(spectra, axis=0))[::-1]
palette = plt.cm.viridis(np.linspace(0.05, 0.95, spectra.shape[1]))
for j, idx in enumerate(order):
    ax.plot(gains, spectra[:, idx], color=palette[j], lw=0.8, alpha=0.48)

ax.axhline(0.0, color='0.25', lw=1.0, ls='--')
ax.axhline(-1e-4, color='0.55', lw=0.8, ls=':')
ax.set_xlabel('coupling gain $g$ (one fixed plant, initial state, and seed)')
ax.set_ylabel('Lyapunov exponent $\\lambda_i$')
ax.set_title('Same-seed full Lyapunov spectrum and scalar-log MG')
ax.set_ylim(-2.1, 0.025)

ax2 = ax.twinx()
ax2.plot(gains, df.MG, color='black', marker='o', ms=5, lw=2.4,
         label='MG from the same scalar log')
ax2.plot(gains, df.n_positive, color='#d73027', marker='s', ms=4, lw=1.7,
         ls='--', label=r'$\#\{\lambda_i>10^{-4}\}$')
ax2.plot(gains, df.n_weak, color='#1a9850', marker='^', ms=4, lw=1.7,
         ls='-.', label=r'$\#\{\lambda_i>-10^{-4}\}$')
ax2.set_ylabel('MG / number of non-contracting directions')
ax2.set_ylim(0, max(22, float(df.MG.max()) * 1.12))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax.grid(alpha=.20)
fig.tight_layout()
fig.savefig(OUT / 'same_seed_spectrum_MG_direct.pdf', bbox_inches='tight')
fig.savefig(OUT / 'same_seed_spectrum_MG_direct.png', dpi=240, bbox_inches='tight')

# Compact quantitative summary for the report.
summary = pd.DataFrame({
    'gain': gains,
    'MG': df.MG,
    'n_positive': df.n_positive,
    'n_weak': df.n_weak,
    'lambda_max': df.lambda_max,
    'lambda_second': df.lambda_second,
    'kaplan_yorke': df.kaplan_yorke,
})
summary.to_csv(OUT / 'same_seed_spectrum_MG_direct_summary.csv', index=False)
print(summary.to_string(index=False))

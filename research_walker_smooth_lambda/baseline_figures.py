from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
names={'MG':'MG','entropy':'Spectral entropy','recurrence':'Scalar recurrence error','increments':'Normalized increments'}
d=pd.read_csv(H/'baselines_by_seed.csv')
s=pd.read_csv(H/'baselines_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(10.5,3.0),layout='constrained')
for ax,signal,title in zip(axes,['action_norm','delta_action_norm','mean_action'],['Action norm (primary)','Increment norm (secondary)','Mean action (secondary)']):
    for metric,name in names.items():
        g=s[(s.signal==signal)&(s.metric==metric)].sort_values('coef')
        ax.plot(np.arange(4),g.ratio,'o-',label=name)
    ax.axhline(1,c='gray',ls='--',lw=.8)
    ax.set_xticks(range(4),['0','0.25','1','4']);ax.set_xlabel('Regularization coefficient')
    ax.set_title(title);ax.grid(alpha=.2)
axes[0].set_ylabel('Median paired ratio to control')
axes[1].legend(fontsize=7,loc='upper left')
fig.savefig(H/'baselines_comparison.pdf');fig.savefig(H/'baselines_comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(10.5,2.7),layout='constrained')
for ax,(metric,name) in zip(axes,names.items()):
    sub=d[(d.signal=='action_norm')&(d.metric==metric)]
    for seed,g in sub.groupby('seed'):
        ax.plot(range(4),g.sort_values('coef').ratio,'o-',alpha=.65,lw=1,label=str(seed))
    ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_title(name,fontsize=10)
    ax.set_xticks(range(4),['0','.25','1','4']);ax.set_xlabel('Coefficient');ax.grid(alpha=.2)
axes[0].set_ylabel('Ratio to paired control');axes[0].legend(fontsize=6,ncol=2)
fig.savefig(H/'baselines_seed_curves.pdf');plt.close(fig)

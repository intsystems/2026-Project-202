from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
d=pd.read_csv(H/'checkpoint_timecourse.csv')
g=d.groupby('step')
fig,ax=plt.subplots(1,2,figsize=(9,3.2))
for col,label,color in [('MG_ratio','MG: норма действия','#1f77b4'),('J1_ratio','J1','#d62728'),('J2_ratio','J2','#2ca02c')]:
    med=g[col].median();lo=g[col].quantile(.25);hi=g[col].quantile(.75)
    ax[0].plot(med.index/1048576,med.values,'o-',label=label,color=color)
    ax[0].fill_between(med.index/1048576,lo.values,hi.values,color=color,alpha=.12)
ax[0].axhline(1,color='gray',ls='--');ax[0].set(xlabel='Доля бюджета обучения',ylabel='Сглаживание / контроль')
ax[0].legend(fontsize=8)
for seed,part in d.groupby('seed'):
    ax[1].plot(part.step/1048576,part.MG_ratio,'o-',label=str(seed))
ax[1].axhline(1,color='gray',ls='--');ax[1].set(xlabel='Доля бюджета обучения',ylabel='MG отношение');ax[1].legend(title='Сид',fontsize=7)
fig.tight_layout();fig.savefig(H/'checkpoint_timecourse.pdf');fig.savefig(H/'checkpoint_timecourse.png',dpi=180);plt.close(fig)

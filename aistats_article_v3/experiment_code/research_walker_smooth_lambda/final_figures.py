"""Overview for the consolidated final report, using saved per-seed results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
mg=pd.read_csv(H/'action_mg_by_seed.csv');mg=mg[mg.signal=='action_norm']
refs=pd.read_csv(H/'confirmation_summary.csv')
fig,axs=plt.subplots(1,3,figsize=(10.5,2.9),layout='constrained')
xx=np.arange(4);coefs=[0,.25,1,4]
for seed,g in mg.groupby('seed'):
    axs[0].plot(xx,g.sort_values('coef').ratio,'o-',lw=1,alpha=.6,label=f'Сид {seed}')
axs[0].plot(xx,mg.groupby('coef').ratio.median(),'ko-',lw=2,label='Медиана')
axs[0].set_title('MG нормы действия');axs[0].set_ylabel('Отношение к контролю')
axs[0].legend(fontsize=6,ncol=2)
for metric,name in [('J1_ratio','Первые разности J1'),('J2_ratio','Вторые разности J2'),('R_ratio','Повторение состояния R'),('D_ratio','Фазовый разброс D')]:
    axs[1].plot(xx,refs.groupby('coef')[metric].median(),'o-',label=name)
axs[1].set_title('Независимые проверки');axs[1].legend(fontsize=6)
reward=[]
for c in coefs:
    d=pd.concat([pd.read_csv(H/f'seed{s}_lambda{c:g}/test.csv') for s in range(231,236)])
    reward.append(d.padded_reward.mean())
axs[2].plot(xx,refs.groupby('coef').healthy.sum()/50,'o-',label='Доля успешных эпизодов')
axs[2].plot(xx,np.array(reward)/reward[0],'s-',label='Награда / контроль')
axs[2].set_ylim(0,1.15);axs[2].set_title('Качество ходьбы');axs[2].legend(fontsize=6)
for ax in axs:
    ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_xticks(xx,['0','0.25','1','4'])
    ax.set_xlabel('Коэффициент штрафа');ax.grid(alpha=.2)
fig.savefig(H/'final_evidence.pdf');fig.savefig(H/'final_evidence.png',dpi=200);plt.close(fig)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;P=H.parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
d=pd.read_csv(H/'two_cohorts_by_seed.csv')
fig,axs=plt.subplots(1,2,figsize=(10.5,2.8),layout='constrained')
for ax,cohort,title in zip(axs,sorted(d.cohort.unique()),['Первая серия: выбор логов после опыта','Следующая серия: те же настройки MG']):
    g=d[d.cohort==cohort]
    for i,(signal,label) in enumerate([('action_norm','Норма команды'),('delta_action_norm','Норма приращения'),('mean_action','Средняя команда')]):
        x=g[g.signal==signal].sort_values('seed');ax.plot(range(5),x.ratio,'o-',label=label)
    ax.set_xticks(range(5),sorted(g.seed.unique()));ax.axhline(1,c='gray',ls='--',lw=.8)
    ax.set_ylim(.4,1.06);ax.set_title(title);ax.set_xlabel('Сид дообучения');ax.grid(alpha=.2)
axs[0].set_ylabel('MG сглаживания / MG контроля');axs[1].legend(fontsize=7)
fig.savefig(H/'control_logs.pdf');plt.close(fig)
ph=pd.read_csv(H/'phase_confirmation.csv')
fig,axs=plt.subplots(1,2,figsize=(10.5,2.65),layout='constrained')
for col,name in [('recurrence','Ошибка повторения R'),('D_strobe','Разброс при одной фазе D'),('C_cycle','Разброс по всему циклу C'),('amplification','Усиление возмущений A')]:
    axs[0].plot(ph.seed,ph[col],'o-',label=name)
for col,name in [('MG_fixed','Правое колено: основной'),('MG_left','Левое колено'),('MG_tau4','Задержка 4'),('MG_tau16','Задержка 16'),('MG_cycles','Окно 14 циклов')]:
    axs[1].plot(ph.seed,ph[col],'o-',label=name)
for ax in axs:
    ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_xticks(ph.seed);ax.set_xlabel('Сид дообучения');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
axs[0].set_title('Независимые свойства движения');axs[1].set_title('MG: все заданные настройки')
axs[0].set_ylabel('Слежение / контроль')
fig.savefig(H/'whole_gait.pdf');plt.close(fig)

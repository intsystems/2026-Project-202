from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

H=Path(__file__).resolve().parent
S=H.parent/'research_text_vae'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
colors={'base':'#64748b','regularized':'#2563eb','protected':'#d97706'}
labels={'base':'Слабая регуляризация','regularized':'Подавление кода','protected':'Сохранение кода'}
fig,axs=plt.subplots(1,3,figsize=(10.4,3.1),layout='constrained')
windows=pd.read_csv(S/'protection/freebits/seed1/three_arm_windows.csv')
for arm in colors:
    folder=S/(f'protection/freebits/seed1' if arm=='protected' else f'confirmation_seed1/{arm}')
    ref=pd.read_csv(folder/'reference.csv')
    for ax,col,title in zip(axs[:2],['MI','shuffle_symkl'],['Информация в коде, нат','Реакция на замену кода, нат/токен']):
        ax.plot(ref.step,ref[col],color=colors[arm],lw=1.7,label=labels[arm]);ax.set_title(title,fontsize=10)
    w=windows.query('arm == @arm and window == 512 and tau == 1')
    axs[2].plot(w.end,w.MG,color=colors[arm],lw=1.7,label=labels[arm])
for ax in axs:
    ax.axvline(1024,color='#999999',ls=':',lw=1);ax.set_xlabel('Шаг обучения');ax.grid(alpha=.18)
axs[2].set_title('MG из скалярного NLL',fontsize=10)
axs[2].legend(fontsize=7.8,loc='upper left')
fig.savefig(H/'trajectory_seed1.pdf');fig.savefig(H/'trajectory_seed1.png',dpi=180);plt.close(fig)

p=pd.read_csv(H/'tables/protection_all_seeds.csv').query('seed > 0')
fig,axs=plt.subplots(1,2,figsize=(10.4,2.9),layout='constrained')
axs[0].plot(p.seed,p.q_regularized,'o-',color=colors['regularized'],label='Подавление / базовая ветка')
axs[0].plot(p.seed,p.q_protected,'s-',color=colors['protected'],label='Сохранение / базовая ветка')
axs[0].axhline(1,color='#999999',ls=':');axs[0].set_ylim(.35,1.05)
axs[0].set_ylabel('Отношение поздних MG');axs[0].legend(fontsize=8)
axs[1].plot(p.seed,p.R,'o-',color='#059669');axs[1].axhline(1,color='#999999',ls=':')
axs[1].set_ylabel('MG сохранения / MG подавления');axs[1].set_ylim(.95,1.6)
for ax in axs:ax.set_xlabel('Проверочный сид');ax.set_xticks(p.seed);ax.grid(alpha=.18)
fig.savefig(H/'seed_comparison.pdf');fig.savefig(H/'seed_comparison.png',dpi=180);plt.close(fig)
print('Rendered figures from saved per-step and per-seed results.')

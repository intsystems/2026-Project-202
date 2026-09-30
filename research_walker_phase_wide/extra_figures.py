import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion import H
from measure import experiments

def run():
    seed=experiments()[0];reset=json.loads((H/f'pair{seed}.json').read_text())['common_resets'][0]
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));colors=['#245d92','#cb6d20']
    for coef,color,name in zip([0,3],colors,['Контроль','Слежение']):
        root=H/f'seed{seed}_lambda{coef}/step1048576'/f'reset{reset}';d=np.load(root/'trajectory.npz');t=np.arange(800)*.008
        axes[0].plot(t,d['qpos'][:800,4],color=color,label=name,lw=1)
        a=np.load(root/'probe_eps0.001_a2.npz');b=np.load(root/'probe_eps0.0001_a1.npz')
        axes[1].semilogy(np.arange(601)*.008,np.median(a['norms'].reshape(68,601),axis=0)/.001,color=color,label=name)
        for val,ls,eps in [(a,'-','0.001'),(b,'--','0.0001')]:
            axes[2].semilogy(np.arange(1,18),val['singular_values'][0],ls,color=color,label=f'{name}, eps={eps}')
    axes[0].set(xlabel='Время, с',ylabel='Угол правого колена');axes[1].set(xlabel='Время, с',ylabel='Усиление возмущений');axes[2].set(xlabel='Номер сингулярного числа',ylabel='Оценка за 152 шага')
    for ax in axes:ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Первый сид {seed}, reset {reset}; спектры чувствительны к величине возмущения',fontsize=10);fig.tight_layout();fig.savefig(H/'same_seed_diagnostics.pdf');fig.savefig(H/'same_seed_diagnostics.png',dpi=150);plt.close(fig)

if __name__=='__main__':run()

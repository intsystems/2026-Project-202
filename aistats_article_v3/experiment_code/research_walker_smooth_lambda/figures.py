from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
c=[0,.25,1,4];xx=np.arange(4)
refs=pd.read_csv(H/'confirmation_summary.csv')
mg=pd.read_csv(H/'action_mg_by_seed.csv')
fig,axes=plt.subplots(1,3,figsize=(10.5,3.05),layout='constrained')
ax=axes[0]
for seed,g in mg[mg.signal=='action_norm'].groupby('seed'):
    ax.plot(xx,g.sort_values('coef').ratio,'o-',alpha=.45,lw=1,label=f'Seed {seed}')
med=mg[mg.signal=='action_norm'].groupby('coef').ratio.median()
ax.plot(xx,med.values,'ko-',lw=2.5,label='Median')
ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_title('MG of action norm');ax.set_ylabel('Ratio to paired control')
ax.legend(fontsize=7,ncol=2,loc='upper left')
ax=axes[1]
for k,name in [('J1_ratio','J1: action differences'),('J2_ratio','J2: second differences'),('R_ratio','R: state recurrence'),('D_ratio','D: section dispersion')]:
    ax.plot(xx,refs.groupby('coef')[k].median(),'o-',label=name)
ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_title('Independent diagnostics');ax.legend(fontsize=7)
ax=axes[2]
healthy=refs.groupby('coef').healthy.sum()/50
reward=[]
for coef in c:
    d=pd.concat([pd.read_csv(H/f'seed{s}_lambda{coef:g}/test.csv') for s in range(231,236)])
    reward.append(d.padded_reward.mean())
ax.plot(xx,healthy,'o-',label='Healthy fraction')
ax.plot(xx,np.array(reward)/reward[0],'s-',label='Mean reward / control')
ax.set_ylim(0,1.15);ax.set_title('All test episodes');ax.legend(fontsize=8)
for ax in axes:
    ax.set_xticks(xx,[str(v) for v in c]);ax.set_xlabel('Regularization coefficient');ax.grid(alpha=.2)
fig.savefig(H/'lambda_comparison.pdf');fig.savefig(H/'lambda_comparison.png',dpi=180);plt.close(fig)

t=pd.read_csv(H/'timecourse_summary.csv');fig,axes=plt.subplots(1,3,figsize=(10.5,2.8),layout='constrained')
for ax,k,title in zip(axes,['MG_ratio','J1_ratio','J2_ratio'],['MG of action norm','J1','J2']):
    for coef in [.25,1,4]:
        g=t[t.coef==coef];ax.plot(g.step/1e6,g[k],'o-',label=f'lambda={coef:g}')
        if k=='MG_ratio':
            for _,r in g.iterrows():
                ax.annotate(str(int(r.n)),(r.step/1e6,r[k]),xytext=(0,5),textcoords='offset points',fontsize=7)
    ax.axhline(1,c='gray',ls='--',lw=.8);ax.set_xlabel('Additional transitions (millions)');ax.set_title(title);ax.grid(alpha=.2)
axes[0].set_ylabel('Ratio to paired control');axes[2].legend(fontsize=8)
fig.savefig(H/'lambda_timecourse.pdf');fig.savefig(H/'lambda_timecourse.png',dpi=180)
plt.close(fig)

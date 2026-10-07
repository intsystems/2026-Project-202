"""Build manuscript figures/tables from saved evidence, without training models."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parent
E=H/'evidence'
E.mkdir(exist_ok=True)
R=E if '--bundled' in sys.argv else H.parent
manifest=[]
def source(rel):
    p=R/rel
    dest=E/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if p.resolve()!=dest.resolve():shutil.copy2(p,dest)
    manifest.append(dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return p
def csv(rel):return pd.read_csv(source(rel))
def js(rel):return json.loads(source(rel).read_text(encoding='utf-8'))
def table(name,cols,head,rows):
    lines=[r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(head)+r' \\',r'\midrule']
    lines+=[' & '.join(map(str,r))+r' \\' for r in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    (H/'tables'/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

known=csv('research_known_modes_results/stationary_summary.csv')
generator=csv('research_generator/results/runs.csv')
gen=js('research_generator/results/summary.json')
graded=csv('research_trajectory_reference/results_graded/per_run.csv')
gs=js('research_trajectory_reference/results_graded/summary.json')
det=csv('research_trajectory_reference/results_detector/test_per_run.csv')
csv('research_trajectory_reference/results_detector/test_overall.csv')
collapse=js('research_trajectory_reference/results_collapse/summary.json')
vae=csv('research_text_vae/protection/all_seeds.csv').query('seed > 0')
vs=js('research_text_vae/summary.json')
vp=js('research_text_vae/protection/summary.json')
cyclic=csv('research_text_vae/cyclical/all_scores.csv').query('seed > 100')
csv('research_text_vae/cyclical/costs.csv')
js('research_text_vae/cyclical/benchmark.json')
force=js('research_force_motion/results_chaotic/report_summary.json')
csv('research_force_motion/results_chaotic/benchmark.csv')
osc=csv('research_sync_control_results/computation_benchmark_repeated.csv')
csv('research_sync_control_results/same_seed_timing.csv')
resnets={}
for mode in ['scratch','finetune']:
    resnets[mode]=csv(f'research_trajectory_reference/results_resnet/e7_results/{mode}/per_run.csv')
    csv(f'research_trajectory_reference/results_resnet/e7_results/{mode}/detector_per_run.csv')
for rel in ['research_force_motion/PROTOCOL.md','research_generator/REPORT_E9_ru.md',
            'research_trajectory_reference/REPORT_E8_ru.md','research_known_modes_results/report_ru.md',
            'research_text_vae/PROTOCOL.md','research_text_vae/protection/PROTOCOL.md',
            'research_text_vae/cyclical/PROTOCOL.md','research_walker_overview/report_ru.md',
            'research_mnist_dynamics/report_ru.md']:
    source(rel)

names={'MG':'MG','crossings':'Trend crossings','lag1':'Lag-one correlation','det_std':'Detrended std.'}
strong=['lr10','lr100','freeze_head','freeze_bias','prune50','prune80','prune95']
nulls=['base','batch_up','scale','smooth']
# Names of observer-control arms are fixed by the saved detector output.
if not set(nulls).issubset(set(det.arm)):
    print('Available detector arms:',sorted(det.arm.unique()))
    raise ValueError('Unexpected control names')
rows=[]
for stat,name in names.items():
    d=det[det.stat==stat];a=d[d.arm.isin(strong)];b=d[d.arm.isin(nulls)]
    assert len(a)==28 and len(b)==16
    hits=int(a.hit.sum());alarms=int(b.false_alarm.sum())
    delay=a.loc[a.hit,'delay'].median()
    rows.append([name,f'{hits}/28',f'{alarms}/16','---' if np.isnan(delay) else f'{delay:,.0f}'])
    if stat=='MG':assert hits==27 and alarms==0
table('detectors.tex','lrrr',['Statistic','Detected','False positives','Delay'],rows)
# The primary paper table now includes all measured scalar baselines.
# Keep the four-row legacy reconstruction as a validation above, then expand it.
from review_analysis import main as review_tables
review_tables()

rows=[]
for r in [1,2,4,6]:
    vals=[]
    for w,obs in [(2048,'loss'),(8192,'loss'),(8192,'probe_error')]:
        v=known.query('r == @r and window == @w and observer == @obs')
        if len(v)!=1:raise ValueError((r,w,obs,known.observer.unique()))
        vals.append(f'{v.iloc[0]["median"]:.2f}')
    rows.append([r,*vals])
table('known_modes.tex','rrrr',['$r$','Loss, 2,048','Loss, 8,192','Probe, 8,192'],rows)

rows=[]
for arm in ['T1','T2','T3','T4','H2','H4','M4']:
    a=generator[generator.arm==arm]
    assert len(a)==5 and a.d.nunique()==1
    rows.append([arm,int(a.d.iloc[0]),f'{a.MG_n0.median():.2f}',f'{a.PR.median():.2f}'])
table('generator_phases.tex','lrrr',['Task','Phases','MG','Covariance PR'],rows)

labels={'base':'No intervention','batch_up':'Batch $\\times4$','scale':'Log $\\times10$',
        'smooth':'Smoothed log','lr3':'Step / 3','lr10':'Step / 10','lr100':'Step / 100',
        'freeze12':'Freeze first two convolutions','freeze_head':'Train head only',
        'freeze_bias':'Train head bias only','prune50':'Prune 50\\%',
        'prune80':'Prune 80\\%','prune95':'Prune 95\\%'}
order=list(labels)
rows=[]
for arm in order:
    a=graded[graded.arm==arm]
    if not len(a):raise ValueError(arm)
    rows.append([labels[arm],f'{a.MG_pre.median():.2f}',f'{a.MG_post.median():.2f}',
                 f'{100*(a.MG_r.median()-1):+.1f}\\%',f'{100*(a.update_PR_r.median()-1):+.1f}\\%',
                 f'{100*a.moving_frac_post.median():.2f}\\%'])
table('cnn_graded.tex','lrrrrr',['Intervention','MG before','MG after','MG change','Update PR change','Moving parameters'],rows)
rows=[]
for arm,name in [('base','Low regularization'),('regularized','Suppressed code'),('protected','Free bits')]:
    rows.append([name,f'{vae[f"MI_{arm}"].median():.3f}',f'{vae[f"shuffle_symkl_{arm}"].median():.3f}',f'{vae[f"MG_{arm}"].median():.3f}'])
table('vae.tex','lrrr',['Branch','MI','Code response','MG'],rows)
rows=[]
for method,name in [('MG','MG'),('std','Standard deviation'),('entropy','Spectral entropy'),('KL','Training KL'),('beta','Schedule'),('periodic','Periodic')]:
    rows.append([name,*[str(int(cyclic.query('method == @method and budget == @b').hits.sum())) for b in [6,12,24]]])
table('cyclic.tex','lrrr',['Method','$B=6$','$B=12$','$B=24$'],rows)

assert len(generator.query('arm != "chaos"'))==35
assert (generator.query('arm != "chaos"').learned).all()
assert vs['confirmation']['primary_agreement']==9 and vp['protection_valid_count']==9
assert np.isclose(gs['auc_events_vs_base_batchup']['MG'],.9305555555555556)
assert collapse['MG']['hit_collapse']=='12/21'
(H/'tables/numbers.tex').write_text('% Values and validation are recorded in evidence/ and claim_sources.md.\n',encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(7.1,2.15),layout='constrained')
a=known.query('window == 8192 and observer == "loss"')
axs[0].plot(a.r,a['median'],'o-',color='#2563eb',label='Loss MG')
axs[0].plot([1,6],[1,6],'--',color='#888888',lw=1,label='Target')
axs[0].set(xlabel='Independent modes',ylabel='Estimate',title='Controlled regression',xticks=[1,2,4,6]);axs[0].legend(fontsize=6.5)
for ax,metric,title in [(axs[1],'MG_n0','One-neuron MG'),(axs[2],'PR','Full-state covariance PR')]:
    for i,arm in enumerate(['H4','M4','T4']):
        a=generator[generator.arm==arm]
        ax.scatter(i+np.linspace(-.12,.12,len(a)),a[metric],s=13,color=['#2563eb','#d97706','#059669'][i])
        ax.plot([i-.24,i+.24],[a[metric].median()]*2,color='black',lw=1)
    ax.set(xticks=range(3),xticklabels=['H4\n1 phase','M4\n2 phases','T4\n4 phases'],title=title,ylim=(0,8));ax.set_xlabel('Four output spectral lines')
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#555555',markersize=3,label='One seed'),
                       Line2D([],[],color='black',lw=1,label='Median')],
              loc='upper left',fontsize=6.5,frameon=False,ncol=2,columnspacing=.7,handlelength=1.2)
for ax in axs:ax.grid(alpha=.18)
fig.savefig(H/'figures/components.pdf');plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(7.1,2.65),layout='constrained')
selected=['base','batch_up','lr10','lr100','freeze_head','prune80']
for i,arm in enumerate(selected):
    a=graded[graded.arm==arm];axs[0].scatter(a.MG_r,np.full(len(a),i)+np.linspace(-.12,.12,len(a)),s=12,color='#2563eb')
    axs[0].plot([a.MG_r.median()]*2,[i-.3,i+.3],color='black',lw=1)
axs[0].set(yticks=range(len(selected)),yticklabels=['Control','Batch x4','LR / 10','LR / 100','Head only','Prune 80%'],xlabel='After / before MG',title='CNN: new seeds')
axs[0].invert_yaxis();axs[0].axvline(1,color='#888888',ls='--',lw=1)
for mode,c in [('scratch','#2563eb'),('finetune','#d97706')]:
    for i,arm in enumerate(['base','batch_up','freeze_head','lr100','prune95']):
        a=resnets[mode].query('arm == @arm');offset=-.13 if mode=='scratch' else .13
        axs[1].scatter(np.full(len(a),i)+offset,a.MG,s=12,color=c,label=mode if i==0 else None)
axs[1].set(xticks=range(5),xticklabels=['Base','Batch','Head','LR','Prune'],ylabel='After / before MG',title='ResNet: frozen rule')
axs[1].axhline(1,color='#888888',ls='--',lw=1);axs[1].legend(fontsize=6.5)
axs[2].plot(vae.seed,vae.q_regularized,'o-',color='#2563eb',ms=3,label='Suppressed')
axs[2].plot(vae.seed,vae.q_protected,'s-',color='#d97706',ms=3,label='Free bits')
axs[2].set(xlabel='Confirmation seed',ylabel='MG / control MG',title='VAE: paired branches',xticks=[1,3,5,7,9],ylim=(.35,1.06))
axs[2].axhline(1,color='#888888',ls='--',lw=1);axs[2].legend(fontsize=6.5,loc='upper right')
for ax in axs:ax.grid(alpha=.18)
fig.savefig(H/'figures/applications.pdf');plt.close(fig)

# Keep an anonymous numerical transcription, not the legacy source with author metadata.
legacy=R/'aistats_article/aistats2027.tex'
if legacy.exists():
    legacy_record=dict(source='aistats_article/aistats2027.tex',
        sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
        analytic_matrix_mae=.27,driven_MLP_mae=.92,digits_covariance_reference_mae=.90,
        aggregation_note='The head reference is measured covariance effective rank, not exact phase count.',
        observer_mae=dict(probe_loss=.60,parameter_norm=.41,parameter_projection=.34,gradient_norm=1.81))
    (E/'legacy_validation.json').write_text(json.dumps(legacy_record,indent=2),encoding='utf-8')
else:
    assert (E/'legacy_validation.json').exists()
(E/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Generated tables and 2 vector figures; checked primary results against {len(manifest)} evidence files.')

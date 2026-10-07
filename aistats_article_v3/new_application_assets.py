"""Rebuild practical-utility tables from bundled per-run evidence; no training."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
E=H/'evidence/practical_utility'
FILES=['final_audited_decisions.csv','final_audited_comparisons.csv','final_audit.json',
 'query_fresh_comparisons.csv','query_fresh.csv','query_paired_timing.csv','query_paired_summary.csv',
 'query_intervals.csv','query_audit.json','FINAL_CONFIRMATION.md','QUERY_BUDGET.md',
 'ROUTING_V2.md','ROUTING_REPLICATION.md','GENERATOR_SCREEN.md','query_freeze.json',
 'final_freeze.json','routing_v2_manifest.json','routing_v2_cv.csv','final_without_control.csv',
 'query_budget.py','query_compare.py','routing_v2.py','campaign.py','generator_screen.py',
 'fresh_generators.py','geometric_audit.py','audit_final.py','check_query_audit.py']
def table(name,headers,rows):
 lines=[r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
 lines+=[' & '.join(map(str,row))+r' \\' for row in rows]
 lines +=[r'\bottomrule',r'\end{tabular}']
 (H/'tables'/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
 E.mkdir(parents=True,exist_ok=True)
 if '--bundled' not in sys.argv:
  source=H.parent/'research_mg_real_forecast'
  for name in FILES:shutil.copy2(source/name,E/name)
  it=H.parent/'research_mg_interventions'
  for src,dst in [('v2/decisions_10_17.csv','intervention_decisions.csv'),('v2/frozen_rules.json','intervention_rules.json'),('v2/freeze_manifest.json','intervention_freeze.json'),('PROTOCOL.md','INTERVENTION_PROTOCOL.md'),('run.py','intervention_train.py'),('control.py','intervention_control.py')]:shutil.copy2(it/src,E/dst)
 first=pd.read_csv(E/'final_audited_decisions.csv');last=pd.read_csv(E/'query_fresh_comparisons.csv')
 for data,seeds in [(first,set(range(651,681))),(last,set(range(681,701)))]:
  assert set(data.seed)==seeds
  assert data.groupby('method').size().eq(8*len(seeds)).all()
  assert not data.duplicated(['method','seed','arm']).any()
  assert np.isfinite(data.error).all()
 firstmeans=first.groupby('method').error.mean();means=last.groupby('method').error.mean()
 timing=pd.read_csv(E/'query_paired_timing.csv').groupby(['seed','arm','method']).seconds.median().unstack()
 assert len(timing)==12
 a=last.query('method=="MG"').merge(last.query('method=="MG_q128"'),on=['seed','arm'],suffixes=('_full','_q'),validate='one_to_one')
 assert len(a)==160 and a.model_full.eq(a.model_q).all() and np.allclose(a.error_full,a.error_q,rtol=1e-12)
 short=[['MG',f'{means.MG:.4f}',f'{1000*timing.MG.median():.1f}'],
        [r'MG, $q=128$',f'{means.MG_q128:.4f}',f'{1000*timing.MG_q128.median():.1f}'],
        ['Held-out model search',f'{means.validation:.4f}',f'{1000*timing.validation.median():.1f}'],
        ['Cheap + held-out search',f'{means.cheap_val:.4f}','---'],
        ['Held-out errors + MG',f'{means.val_MG:.4f}','---']]
 table('forecast_utility.tex',['Selector','NMSE',r'Time (ms)'],short)
 labels={'MG':'MG','old_MG':'Earlier MG rule','fixed':'Best fixed on development runs','validation':'Held-out model search','cheap':'Cheap features','cheap_val':'Cheap + held-out search','val':'Held-out errors','val_MG':'Held-out errors + MG','cheap_MG':'Cheap + MG','cheap_val_MG':'Cheap + held-out search + MG','TwoNN':'TwoNN','LB':'Levina--Bickel','PRdelay':'Delay covariance PR','permutation_entropy':'Permutation entropy','sample_entropy':'Sample entropy','all_nonMG':'All features except MG','all_withMG':'All features including MG','oracle':'Oracle (future targets)'}
 table('forecast_all.tex',['Selector','30 seeds','20 new seeds'],[[name,f'{firstmeans[key]:.5f}',f'{means[key]:.5f}' if key in means else '---'] for key,name in labels.items()])
 seedmeans=last.groupby(['seed','method']).error.mean().unstack()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.3),layout='constrained')
 for name,label,marker,color in [('MG_q128','MG, 128 queries','o','#2563eb'),('validation','Held-out model search','s','#d97706'),('val_MG','Held-out errors + MG','^','#059669')]:
  axs[0].plot(seedmeans.index,seedmeans[name],label=label,marker=marker,ms=3,lw=.9,color=color)
 axs[0].set(xlabel='New initialization seed',ylabel='Mean NMSE',xticks=[681,685,690,695,700]);axs[0].legend(fontsize=6,frameon=False)
 for i,k in enumerate(['MG','MG_q128','validation']):
  vals=timing[k].to_numpy()*1000;axs[1].scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=12,color=['#2563eb','#059669','#d97706'][i]);axs[1].plot([i-.22,i+.22],[np.median(vals)]*2,color='black',lw=1)
 axs[1].set(xticks=range(3),xticklabels=['Full MG','MG, 128','Validation'],ylabel='Time (ms)');axs[1].tick_params(axis='x',labelsize=7)
 for ax in axs:ax.grid(axis='y',alpha=.18)
 fig.savefig(H/'figures/forecast_utility.pdf');plt.close(fig)
 inter=pd.read_csv(E/'intervention_decisions.csv');assert set(inter.seed)==set(range(10,18))
 summary=inter.groupby(['action','noise','method'])[['test_acc','mac_ratio','step']].mean()
 rows=[]
 for action,noise,label in [('stop',.4,'Stop, 40\\% noise'),('stop',.6,'Stop, 60\\% noise'),('freeze',0,'Freeze, clean'),('prune',0,'Prune, clean')]:
  for method in ['MG','fixed','level','entropy','never']:
   r=summary.loc[(action,noise,method)];rows.append([label,method,f'{100*r.test_acc:.2f}',f'{r.mac_ratio:.3f}'])
 table('intervention_utility.tex',['Task','Rule',r'Accuracy (\%)','MAC ratio'],rows)
 ci=pd.read_csv(E/'query_intervals.csv').set_index('baseline').loc['validation']
 diff=(seedmeans.validation-seedmeans.MG_q128).to_numpy();rng=np.random.default_rng(811)
 rng.choice(np.zeros(len(diff)),size=(20000,len(diff))) # preceding full-MG contrast in the archived bootstrap loop
 expected=np.quantile(rng.choice(diff,size=(20000,len(diff))).mean(1),[.025,.975])
 assert np.allclose(expected,[ci.ci_low,ci.ci_high],atol=1e-12)
 assert np.isclose(firstmeans.MG,.03138477923818985) and np.isclose(means.MG_q128,.036370895240979154)
 checks=dict(forecast_30_seed_coverage=True,forecast_20_seed_coverage=True,all_methods_same_records=True,seed_bootstrap_recomputed=True,
  query_decisions_equal=160,intervention_seed_coverage=True,
  final_MG=float(firstmeans.MG),final_validation=float(firstmeans.validation),
  query_MG=float(means.MG_q128),query_validation=float(means.validation),
  timing_medians_ms={k:float(1000*timing[k].median()) for k in ['MG','MG_q128','validation']},
  median_speedup_full=float((timing.MG/timing.MG_q128).median()),median_speedup_validation=float((timing.validation/timing.MG_q128).median()))
 (H/'utility_validation.json').write_text(json.dumps(checks,indent=2))
 paths=[p for p in E.iterdir() if p.is_file() and p.name!='manifest.json']
 (E/'manifest.json').write_text(json.dumps([dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)],indent=2))
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()

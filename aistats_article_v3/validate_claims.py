"""Audit key claims against packaged per-run evidence; never reruns training."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent
E=H/'evidence'
checks={}
def js(p):return json.loads((E/p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(E/p)
def check(name,condition):
    assert condition,name
    checks[name]=True

nulls=['base','batch_up','scale','smooth']
hits=alarms=controls=events=0
changes=[]
for mode in ['scratch','finetune']:
    d=csv(f'research_trajectory_reference/results_resnet/e7_results/{mode}/detector_per_run.csv')
    d=d[d.stat=='MG'];a=d[d.arm=='freeze_head'];b=d[d.arm.isin(nulls)]
    hits+=int(a.hit.sum());alarms+=int(b.false_alarm.sum());events+=len(a);controls+=len(b)
    p=csv(f'research_trajectory_reference/results_resnet/e7_results/{mode}/per_run.csv')
    changes.append(100*(p.query('arm == "freeze_head"').MG.median()-1))
check('ResNet freezing 6/6, controls 0/24',(hits,events,alarms,controls)==(6,6,0,24))
check('ResNet changes rounded to -27.6 and -30.8',np.allclose(np.round(changes,1),[-27.6,-30.8]))

v=js('research_text_vae/summary.json')['confirmation']
p=js('research_text_vae/protection/summary.json')
check('VAE loss 9/9',v['primary_agreement']==v['reference_events']==9)
check('VAE q and interval',np.allclose(np.round([v['median_paired_MG'],*v['bootstrap_median95']],3),[.517,.480,.542]))
check('VAE preservation and interval',p['protection_valid_count']==p['protected_MG_higher_count']==9 and np.allclose(np.round([p['median_R_valid'],*p['median_R_valid_bootstrap95']],3),[1.286,1.179,1.418]))
cy=csv('research_text_vae/cyclical/all_scores.csv').query('seed > 100 and budget == 12')
check('Cyclic VAE 28/41 vs 38/41',cy.query('method=="MG"').hits.sum()==28 and cy.query('method=="periodic"').hits.sum()==38 and cy.query('method=="MG"').events.sum()==41)

g=csv('research_generator/results/runs.csv')
t=g.pivot(index='seed',columns='arm',values='MG_n0')
check('E9 harmonic order in all five seeds',len(t)==5 and ((t.H4<t.M4)&(t.M4<t.T4)&(t.H2<t.T2)).all())
check('E9 median T1-T4',np.allclose(np.round(g.groupby('arm').MG_n0.median().loc[['T1','T2','T3','T4']].to_numpy(),2),[1.15,2.84,5.90,5.93]))
f=js('research_force_motion/results_chaotic/report_summary.json')
b=f['benchmark_median']
ratios=[b['full_spectrum'][str(step)]/b['mg'][str(step)] for step in [0,40000]]
check('FORCE 14-19x rounded speed ratio',np.allclose(np.round(ratios),[14,19]))
check('FORCE five task successes, four strict cycles',f['all_five_task_success'] and f['strict_endpoint_pass']==4)
o=csv('research_sync_control_results/computation_benchmark_repeated.csv').query('n==128')
check('Oscillator 22-23x rounded speed ratio',np.allclose(np.round(o.full_spectrum_seconds/o.scalar_seconds),[22,23]))
cost=csv('research_text_vae/cyclical/costs.csv').query('budget==12').set_index('method')
check('VAE monitoring total cost',np.allclose(np.round([cost.loc['MG','total_seconds'],cost.loc['periodic','total_seconds'],cost.loc['MG','cached_probe_total_seconds']],2),[54.68,2.72,3.54]))
col=js('research_trajectory_reference/results_collapse/summary.json')
check('ReLU-death comparison retained',col['MG']['hit_collapse']=='12/21' and col['MG']['alarm_non_collapse']=='12/31' and col['crossings']['hit_collapse']=='19/21')
(H/'claims_validation.json').write_text(json.dumps(dict(checks=checks,scope='Checks saved numerical evidence and manuscript counts; no new training or independent scientific certification.'),indent=2),encoding='utf-8')
print(f'{len(checks)} numerical evidence checks passed.')

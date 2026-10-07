"""Rebuild overview tables from saved results; no model training or refitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'research_text_vae'
OUT = HERE / 'tables'
OUT.mkdir(exist_ok=True)
used = set()

def csv(rel):
    p = SOURCE / rel
    used.add(p)
    return pd.read_csv(p)

def js(rel):
    p = SOURCE / rel
    used.add(p)
    return json.loads(p.read_text(encoding='utf-8'))

main = csv('all_seeds.csv')
prot = csv('protection/all_seeds.csv')
assert main.seed.tolist() == list(range(10))
assert prot.seed.tolist() == list(range(10))
conf = main[main.seed > 0].copy()
pc = prot[prot.seed > 0].copy()
# Independently reconstruct late medians and the paired MG contrast from traces.
for row in prot.itertuples():
    base = 'pilot_seed0' if row.seed == 0 else f'confirmation_seed{row.seed}'
    frames = {'base': f'{base}/base', 'regularized': f'{base}/regularized',
              'protected': f'protection/freebits/seed{row.seed}'}
    w = csv(f'protection/freebits/seed{row.seed}/three_arm_windows.csv')
    for arm, folder in frames.items():
        r = csv(f'{folder}/reference.csv')
        late = r[r.step.between(2048,3072)]
        for metric in ['MI', 'shuffle_symkl']:
            assert np.isclose(late[metric].median(), getattr(row, f'{metric}_{arm}'))
        z = w[(w.arm == arm) & (w.window == 512) & (w.tau == 1) & w.end.between(2048,3072)]
        assert np.isclose(z.MG.median(), getattr(row, f'MG_{arm}'))
    assert np.isclose(row.R, row.MG_protected / row.MG_regularized)
assert (conf.event & conf.primary_valid).all()
assert (conf.MG_paired < 1).all() and pc.protection_valid.all() and (pc.R > 1).all()
main.to_csv(OUT/'paired_all_seeds.csv', index=False)
prot.to_csv(OUT/'protection_all_seeds.csv', index=False)

arms=[]
for arm in ['base','regularized','protected']:
    arms.append(dict(arm=arm,MI=pc[f'MI_{arm}'].median(),
                     code_response=pc[f'shuffle_symkl_{arm}'].median(),
                     MG=pc[f'MG_{arm}'].median(),
                     median_paired_MG=(pc[f'MG_{arm}']/pc.MG_base).median()))
pd.DataFrame(arms).to_csv(OUT/'three_arm_medians.csv',index=False)

sens=csv('all_seeds_sensitivity.csv').query('seed > 0')
srows=[]
for (w,t),g in sens.groupby(['window','tau']):
    srows.append(dict(window=w,tau=t,median_q=g.paired_MG.median(),min_q=g.paired_MG.min(),
                      max_q=g.paired_MG.max(),correct=int((g.paired_MG<1).sum()),n=len(g)))
pd.DataFrame(srows).to_csv(OUT/'window_sensitivity.csv',index=False)
csv('protection/sensitivity.csv').to_csv(OUT/'protection_sensitivity.csv',index=False)
score=csv('cyclical/all_scores.csv').query('seed > 100')
assert set(score.seed)==set(range(101,110))
budgets=[]
for (budget,method),g in score.groupby(['budget','method']):
    assert len(g)==9 and g.events.sum()==41
    budgets.append(dict(budget=budget,method=method,hits=int(g.hits.sum()),events=int(g.events.sum()),
                        mean_checks=g.checks.mean(),loss_hits=int(g.loss_hits.sum()),
                        recovery_hits=int(g.recovery_hits.sum()),matched_periodic_hits=int(g.matched_periodic_hits.sum())))
pd.DataFrame(budgets).to_csv(OUT/'monitor_budgets.csv',index=False)
score.to_csv(OUT/'monitor_all_confirmation_scores.csv',index=False)
csv('cyclical/costs.csv').to_csv(OUT/'costs.csv',index=False)
bench=js('cyclical/benchmark.json')
summary=dict(main=js('summary.json')['confirmation'],
             protection={k:v for k,v in js('protection/summary.json').items() if k!='seeds'},
             reference_to_MG_cost_ratio=bench['operations']['reference']['median']/bench['operations']['MG']['median'],
             distinct_initializations=20,confirmation_initializations=18,
             protection_reuses_first_cohort=True)
(HERE/'evidence_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
manifest=[dict(path=str(p.relative_to(HERE.parent)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(used)]
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Validated paired/reference medians for all 10 seeds; {len(used)} source files; 18 confirmation initializations.')

"""Collect existing results across all eight Walker2d branches; no experiments."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
P=H.parent
BRANCHES=['gait','repair','smooth','imitation','phase','phase_wide','combined','smooth_lambda']
def main():
    inventory=[];runs=[];sources=[]
    for branch in BRANCHES:
        root=P/f'research_walker_{branch}'
        for p in sorted(root.glob('*/train*.json')):
            j=json.loads(p.read_text(encoding='utf-8'))
            steps=j.get('steps',j.get('additional_steps'))
            runs.append(dict(branch=branch,run=p.parent.name,file=p.relative_to(P).as_posix(),
                 seed=j.get('seed'),steps=steps,resume=j.get('resume',0),
                 transitions=steps-j.get('resume',0)))
        rr=[r for r in runs if r['branch']==branch]
        inventory.append(dict(branch=branch,training_policies=len({r['run'] for r in rr}),
                              training_segments=len(rr),transitions=sum(r['transitions'] for r in rr)))
        for p in root.iterdir():
            if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name!='MANIFEST.json':
                sources.append(dict(path=p.relative_to(P).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(runs).to_csv(H/'training_runs.csv',index=False)
    inv=pd.DataFrame(inventory);inv.to_csv(H/'series_inventory.csv',index=False)
    assert inv.training_policies.sum()==64
    assert inv.transitions.sum()==70254592
    (H/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    first=pd.read_csv(P/'research_walker_smooth/action_mg_summary.csv')
    second=pd.read_csv(P/'research_walker_smooth_lambda/action_mg_by_seed.csv')
    second=second[second.coef==1].copy()
    first['cohort']='221-225 (post hoc)';second['cohort']='231-235 (follow-up)'
    signals=['action_norm','delta_action_norm','mean_action']
    a=pd.concat([first[['cohort','seed','signal','n','ratio']],second[['cohort','seed','signal','n','ratio']]])
    a=a[a.signal.isin(signals)];a.to_csv(H/'two_cohorts_by_seed.csv',index=False)
    summary=a.groupby(['cohort','signal']).agg(ratio=('ratio','median'),seeds=('seed','count'),
                 decreases=('ratio',lambda x:int((x<1).sum()))).reset_index()
    assert (summary.decreases==5).all()
    summary.to_csv(H/'two_cohorts_summary.csv',index=False)
    rows=[]
    for branch in ['smooth','phase_wide']:
        d=pd.read_csv(P/f'research_walker_{branch}/timings.csv')
        for name,g in d.groupby('method'):
            rows.append(dict(branch=branch,method=name,seconds=g.seconds.median(),measurements=len(g)))
    pd.DataFrame(rows).to_csv(H/'timing_comparison.csv',index=False)
    phase=pd.read_csv(P/'research_walker_phase_wide/all_results.csv')
    assert phase.event.sum()==2
    phase.to_csv(H/'phase_confirmation.csv',index=False)
    (H/'checks.json').write_text(json.dumps(dict(branches=8,training_policies=64,
        training_segments=int(inv.training_segments.sum()),transitions=70254592,
        phase_whole_state_successes=int(phase.event.sum()),
        all_three_action_logs_decrease_in_each_cohort=bool((summary.decreases==5).all()),
        no_new_training=True),indent=2))
    print(inv.to_string(index=False));print(summary.to_string(index=False))
if __name__=='__main__':main()

"""Causal monitoring policies, independent reference events and pilot calibration."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
START=1024;END=7168;HORIZON=512
METHODS=['MG','std','entropy','KL','beta']
BUDGETS=[6,12,24]
THRESHOLDS=[.025,.05,.075,.1,.15,.2,.3,.4,.5,.7,1.,1.5,2.,3.]

def reference_events(ref):
    baseline=ref[ref.step.isin([512,640,768,896,1024])][['MI','shuffle_symkl']].median()
    mi=baseline.MI;response=baseline.shuffle_symkl
    def condition(row):
        if row.MI<=.5*mi and row.shuffle_symkl<=.5*response:return 'low'
        if row.MI>=.65*mi and row.shuffle_symkl>=.65*response:return 'high'
        return 'ambiguous'
    raw={int(r.step):condition(r) for r in ref.itertuples()}
    suitable=bool(mi>.1 and response>1e-4 and raw[START]=='high')
    events=[];state='high';pending=0;trace=[]
    for row in ref[ref.step>=START].itertuples():
        current=raw[int(row.step)]
        if suitable and row.step>START:
            if current!='ambiguous' and current!=state:
                pending+=1
                if pending==2:
                    state=current;pending=0
                    events.append(dict(id=len(events),step=int(row.step),state=state,
                        direction='loss' if state=='low' else 'recovery',
                        MI=float(row.MI),shuffle_symkl=float(row.shuffle_symkl),
                        scored=bool(row.step<=END-HORIZON)))
            else:pending=0
        trace.append(dict(step=int(row.step),raw_condition=current,confirmed_state=state,pending=pending))
    return dict(suitable=suitable,MI0=float(mi),response0=float(response),events=events,trace=trace)

def policy(feature,method,threshold,budget):
    # Only the prefix available at each row is used; no labels or future rows.
    anchor=float(feature[feature.end.isin([512,640,768,896,1024])][method].median())
    last=START;checks=[];decisions=[]
    for row in feature[feature.end>START].itertuples():
        value=float(getattr(row,method));valid=np.isfinite(value) and value>0
        if method=='MG':valid=valid and not row.degenerate
        score=float(abs(np.log(max(value,1e-12)/max(anchor,1e-12)))) if valid else None
        trigger=bool(valid and score>=threshold and row.end-last>=128 and len(checks)<budget)
        decisions.append(dict(step=int(row.end),feature=value,anchor=anchor,score=score,valid=bool(valid),trigger=trigger))
        if trigger:checks.append(int(row.end));anchor=value;last=int(row.end)
    return checks,decisions

def periodic(count):
    if count==0:return []
    grid=np.arange(START+64,END+1,64)
    indices=np.ceil(np.arange(1,count+1)*len(grid)/count).astype(int)-1
    return grid[indices].astype(int).tolist()

def score(checks,truth):
    events=truth['events'];raw={r['step']:r['raw_condition'] for r in truth['trace']}
    matched={};records=[]
    for t in checks:
        past=[e for e in events if e['step']<=t];hit=None
        if past:
            e=past[-1]
            if e['id'] not in matched and t-e['step']<=HORIZON and raw[t]==e['state']:
                hit=e;matched[e['id']]=t-e['step']
        records.append(dict(step=t,event_id=None if hit is None else hit['id'],
            direction=None if hit is None else hit['direction'],delay=None if hit is None else t-hit['step'],
            scored_hit=bool(hit is not None and hit['scored'])))
    scored=[e for e in events if e['scored']];hits=[e for e in scored if e['id'] in matched]
    delays=[matched[e['id']] for e in hits]
    result=dict(checks=len(checks),events=len(scored),hits=len(hits),misses=len(scored)-len(hits),
        recall=len(hits)/len(scored) if scored else None,unproductive=len(checks)-len(hits),
        mean_delay=float(np.mean(delays)) if delays else None,delays=delays,
        censored_events=sum(not e['scored'] for e in events),
        censored_hits=sum(not e['scored'] and e['id'] in matched for e in events),
        records=records)
    for direction in ['loss','recovery']:
        result[direction+'_events']=sum(e['direction']==direction for e in scored)
        result[direction+'_hits']=sum(e['direction']==direction for e in hits)
    return result

def calibrate():
    out=H/'seed100';feat=pd.read_csv(out/'features.csv');truth=reference_events(pd.read_csv(out/'reference.csv'))
    assert truth['suitable'],'Initial reference condition failed: report unsuitable pilot; do not tune secretly.'
    selection=dict(pilot=100,confirmation_seeds=list(range(101,110)),settings={},
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest(),
        feature_sha256=hashlib.sha256((out/'features.csv').read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((out/'reference.csv').read_bytes()).hexdigest(),pilot_events=truth['events'])
    trials=[]
    for budget in BUDGETS:
        selection['settings'][str(budget)]={}
        for method in METHODS:
            candidates=[]
            for threshold in THRESHOLDS:
                checks,_=policy(feat,method,threshold,budget);s=score(checks,truth)
                row=dict(budget=budget,method=method,threshold=threshold,
                    **{k:v for k,v in s.items() if k not in ['delays','records']})
                trials.append(row)
                key=(-s['hits'],s['unproductive'],s['mean_delay'] if s['mean_delay'] is not None else float('inf'),s['checks'],-threshold)
                candidates.append((key,threshold,row))
            _,threshold,best=min(candidates,key=lambda x:x[0])
            selection['settings'][str(budget)][method]=dict(threshold=threshold,pilot=best)
    assert not any((H/f'seed{s}/meta.json').exists() for s in range(101,110)),'Selection must precede confirmation.'
    (H/'selection.json').write_text(json.dumps(selection,indent=2));pd.DataFrame(trials).to_csv(H/'calibration.csv',index=False)
    print(json.dumps(selection['settings']['12'],indent=2),flush=True)

def evaluate(seed):
    out=H/f'seed{seed}';selection=json.loads((H/'selection.json').read_text())
    feat=pd.read_csv(out/'features.csv');truth=reference_events(pd.read_csv(out/'reference.csv'))
    (out/'events.json').write_text(json.dumps(truth,indent=2));rows=[];details=[]
    for budget in BUDGETS:
        for method in METHODS+['periodic']:
            if method=='periodic':checks=periodic(budget);decisions=[];threshold=None
            else:
                threshold=selection['settings'][str(budget)][method]['threshold']
                checks,decisions=policy(feat,method,threshold,budget)
            stats=score(checks,truth);matched=score(periodic(len(checks)),truth)
            base=dict(seed=seed,method=method,budget=budget,threshold=threshold,suitable=truth['suitable'])
            rows.append(dict(**base,**{k:v for k,v in stats.items() if k not in ['delays','records']},
                matched_periodic_hits=matched['hits'],matched_periodic_mean_delay=matched['mean_delay']))
            details.append(dict(**base,checks=checks,decisions=decisions,score=stats,matched_periodic=matched))
    dense=score(periodic(96),truth)
    pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
    result=dict(seed=seed,truth=truth,policies=details,dense=dense,
        selection_sha256=hashlib.sha256((H/'selection.json').read_bytes()).hexdigest())
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    print('EVALUATED',seed,'events',len(truth['events']),'scored',dense['events'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');p.add_argument('--seed',type=int,default=100);a=p.parse_args()
    if a.calibrate:calibrate()
    evaluate(a.seed)

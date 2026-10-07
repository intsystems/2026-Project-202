from pathlib import Path
import argparse,json,subprocess,sys,concurrent.futures
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import H
from sensors import mg_settings,period_info
from perturb_fixed import probe

def experiments():
    s=json.loads((H/'selection.json').read_text())
    return [(seed,s['selected_coef']) for seed in range(241,246)] if s['selected_coef'] is not None else [(240,1),(240,3)]

def one(seed,coef):
    rows=[];labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda{coef:g}'];a,b=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels]
    ok=a.eligible&b.eligible&a.section_eligible.fillna(False)&b.section_eligible.fillna(False)
    common=[int(x) for x in a.index[ok]]
    (H/f'pair{seed}_lambda{coef:g}.json').write_text(json.dumps(dict(seed=seed,coef=coef,common_resets=common,assessable=len(common)>=8),indent=2))
    for label in labels:
        cp=H/label/'step1048576'
        for p in sorted(cp.glob('reset72*/metrics.json')):
            m=json.loads(p.read_text())
            if not m['eligible']:continue
            cached=p.parent/'MG_windows.csv'
            if not cached.exists():
                data=np.load(p.parent/'trajectory.npz');mg_settings(data).to_csv(cached,index=False)
                (p.parent/'period.json').write_text(json.dumps(period_info(data),indent=2))
            d=pd.read_csv(cached)
            for mode,g in d.groupby('mode'):
                valid=np.isfinite(g.MG)&~g.degenerate
                rows.append(dict(seed=seed,coef=coef,label=label,reset=m['reset'],mode=mode,all_valid=bool(valid.all()),MG=float(g.loc[valid,'MG'].median()) if valid.any() else None,ident_max=float(g.ident.max())))
        if common:probe(cp,common[0],[0,1024])
    pd.DataFrame(rows).to_csv(H/f'MG_seed{seed}_lambda{coef:g}.csv',index=False);print('MEASURED',seed,coef,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--coef',type=float,required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):one(a.seed,a.coef)

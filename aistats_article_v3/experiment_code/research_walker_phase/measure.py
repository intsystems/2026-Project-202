import argparse,json
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from motion import H
from environment import RHO
from features import measure
from probes import probe

def experiments():
    selected=json.loads((H/'selection.json').read_text())['selected_coef'];return list(range(261,266)) if selected is not None else [260]

def calculate(data):
    rows=[]
    for mode in ['fixed','cycles','left','tau4','tau16']:
        tau={'fixed':8,'cycles':round((152/RHO)/19),'left':8,'tau4':4,'tau16':16}[mode];w=round(14*152/RHO) if mode=='cycles' else 2048
        x=data['qpos'][:,7 if mode=='left' else 4]
        for end in np.unique(np.linspace(w,len(x),3).astype(int)):rows.append(dict(mode=mode,window=w,tau=tau,end=int(end),**measure(x[end-w:end],w,tau)))
    return pd.DataFrame(rows)

def run(seed):
    labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda3'];a,b=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels]
    ok=a.eligible&b.eligible&a.section_eligible&b.section_eligible;common=[int(x) for x in a.index[ok]];rows=[]
    (H/f'pair{seed}.json').write_text(json.dumps(dict(seed=seed,common_resets=common,assessable=len(common)>=8),indent=2))
    for label in labels:
        cp=H/label/'step1048576'
        for p in cp.glob('reset74*/metrics.json'):
            m=json.loads(p.read_text())
            if not m['eligible']:continue
            cached=p.parent/'MG_windows.csv'
            if not cached.exists():calculate(np.load(p.parent/'trajectory.npz')).to_csv(cached,index=False)
            d=pd.read_csv(cached)
            for mode,g in d.groupby('mode'):
                good=np.isfinite(g.MG)&~g.degenerate
                rows.append(dict(seed=seed,label=label,reset=m['reset'],mode=mode,valid=bool(good.all()),MG=float(g.loc[good,'MG'].median()) if good.any() else None,ident_max=float(g.ident.max())))
        if common:
            probe(cp,common[0])
            if seed==experiments()[0]:probe(cp,common[0],anchors=(0,),eps=.0001)
    pd.DataFrame(rows).to_csv(H/f'MG_seed{seed}.csv',index=False);print('MEASURED',seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(a.seed)

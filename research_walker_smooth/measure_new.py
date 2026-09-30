import json,concurrent.futures,subprocess,sys,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from diagnostic import mg_settings
from perturb_fixed import probe
H=Path(__file__).resolve().parent

def one(seed):
    c=json.loads((H/'selection.json').read_text())['selected_coef'];rows=[]
    labels=[f'seed{seed}_lambda0',f'seed{seed}_lambda{c:g}'];frames=[pd.read_csv(H/l/'test.csv').set_index('reset') for l in labels]
    common=sorted(set(frames[0][frames[0].eligible].index)&set(frames[1][frames[1].eligible].index))
    (H/f'pair{seed}.json').write_text(json.dumps(dict(seed=seed,common_resets=common,assessable=len(common)>=8),indent=2))
    for label in labels:
        checkpoint=H/label/'step1048576'
        for p in sorted(checkpoint.glob('reset62*/metrics.json')):
            m=json.loads(p.read_text())
            if not m['eligible']:continue
            cached=p.parent/'MG_windows.csv'
            if not cached.exists():mg_settings(np.load(p.parent/'trajectory.npz'),('fixed','cycles','left')).to_csv(cached,index=False)
            df=pd.read_csv(cached)
            for mode,g in df.groupby('mode'):
                valid=np.isfinite(g.MG)&~g.degenerate
                rows.append(dict(seed=seed,label=label,reset=m['reset'],mode=mode,all_valid=bool(valid.all()),MG=float(g.loc[valid,'MG'].median()) if valid.any() else None,ident_min=float(g.ident.min()),ident_max=float(g.ident.max())))
        if common:probe(checkpoint,common[0],[0,1024])
    pd.DataFrame(rows).to_csv(H/f'MG_seed{seed}.csv',index=False);print('MEASURED',seed,flush=True)

def orchestrate():
    def run(s):
        with (H/f'measure_seed{s}.log').open('w') as f:
            r=subprocess.run([sys.executable,str(H/'measure_new.py'),'--seed',str(s)],stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,f'measure seed{s} failed';return s
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:print(list(pool.map(run,range(221,226))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.seed:
        with threadpool_limits(limits=1):one(a.seed)
    else:orchestrate()

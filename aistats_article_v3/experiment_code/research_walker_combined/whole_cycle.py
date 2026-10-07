from pathlib import Path
import json
import numpy as np,pandas as pd
from motion import H,reduced

def metric(d):
    x=reduced(d['qpos'],d['qvel']);phase=np.unwrap(d['phase']*2*np.pi/152)*152/(2*np.pi);curves=[]
    for cycle in range(int(np.ceil(phase[0]/152)),int(np.floor(phase[-1]/152))):
        grid=(cycle+np.arange(64)/64)*152;curve=np.stack([np.interp(grid,phase,x[:,j]) for j in range(17)],axis=-1);curves.append(curve)
    curves=np.array(curves);var=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    c=float(np.mean(np.sum((curves-curves.mean(0))**2,axis=-1))/max(var,1e-30))
    a,b=np.array_split(curves,2);drift=float(np.mean(np.sum((a.mean(0)-b.mean(0))**2,axis=-1))/max(var,1e-30))
    return dict(C_cycle=c,cycles=len(curves),mean_curve_drift=drift,valid=len(curves)>=8)

def run():
    records=[]
    for name,pattern in [('research_walker_phase','reset74*'),('research_walker_phase_wide','reset76*')]:
        folder=H.parent/name
        for cp in sorted(folder.glob('seed*/step1048576')):
            for p in sorted(cp.glob(pattern+'/metrics.json')):
                m=json.loads(p.read_text())
                if not m['eligible']:continue
                v=metric(np.load(p.parent/'trajectory.npz'));(p.parent/'whole_cycle.json').write_text(json.dumps(v,indent=2));records.append(dict(folder=name,label=cp.parent.name,reset=m['reset'],**v))
    pd.DataFrame(records).to_csv(H/'whole_cycle_records.csv',index=False);print('WHOLE CYCLE',len(records))

if __name__=='__main__':run()

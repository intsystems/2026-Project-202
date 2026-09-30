import numpy as np,pandas as pd
from scipy.signal import find_peaks
from motion import reduced
from features import measure

def period_info(data):
    x=reduced(data['qpos'],data['qvel'])
    def select(x):
        var=np.mean(np.sum((x-x.mean(0))**2,axis=1))
        e=np.array([np.mean(np.sum((x[p:]-x[:-p])**2,axis=1))/(2*max(var,1e-30)) for p in range(20,251)])
        candidates=find_peaks(-e)[0];good=candidates[e[candidates]<=1.1*e.min()+.005]
        p=int(good[0]+20) if len(good) else int(e.argmin()+20);return p
    p=select(x);halves=[select(z) for z in np.array_split(x,2)]
    return dict(period=p,halves=halves,stable=max(abs(v/p-1) for v in halves)<=.2)

def mg_settings(data):
    p=period_info(data);rows=[]
    for mode in ['fixed','cycles','left','tau4','tau16']:
        tau={'fixed':8,'cycles':max(1,round(p['period']/19)),'left':8,'tau4':4,'tau16':16}[mode]
        w=14*p['period'] if mode=='cycles' else 2048;x=data['qpos'][:,7 if mode=='left' else 4]
        for end in np.unique(np.linspace(w,len(x),3).astype(int)):
            rows.append(dict(mode=mode,window=w,tau=tau,period=p['period'],period_stable=p['stable'],end=int(end),**measure(x[end-w:end],w,tau)))
    return pd.DataFrame(rows)

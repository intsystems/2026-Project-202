"""Frozen scalar-feature comparisons. See EXPERIMENT_PROTOCOL.md."""
from pathlib import Path
import itertools, json, sys, time, math, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import detrend
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
ROOT=H.parent
sys.path.insert(0,str(ROOT/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau=1,k_neighbors=20,theiler="embedding")
W,S,EVENT,WARM,HORIZON=1000,500,4000,1500,5000
CAL_EVENTS=("lr_step","freeze","prune")
CAL_NULL=("base","batch_up")
TEST_NULL=("base","batch_up","scale","smooth")
STRONG=("lr10","lr100","freeze_head","freeze_bias","prune50","prune80","prune95")
STATS=['MG','abs_diff','norm_diff','perm_entropy','sample_entropy','spectral_entropy','crossings','lag1','det_std','level']

def permutation_entropy(x,order=5):
    if np.std(x)==0:return 0.
    y=np.lib.stride_tricks.sliding_window_view(x,order)
    pat=np.argsort(y,axis=1,kind='stable')
    _,counts=np.unique(pat,axis=0,return_counts=True)
    p=counts/counts.sum()
    return float(-np.sum(p*np.log(p))/math.lgamma(order+1))

def sample_entropy(x,order=2,frac=.2):
    tol=frac*np.std(x)
    if tol==0:return 0.
    long=np.lib.stride_tricks.sliding_window_view(x,order+1)
    short=long[:,:order]
    counts=[]
    for y in [short,long]:
        tree=cKDTree(y)
        counts.append(int(np.sum(tree.query_ball_point(y,tol,p=np.inf,return_length=True))-len(y)))
    return float(-np.log(counts[1]/counts[0])) if counts[0]>0 and counts[1]>0 else np.nan

def scalar_feature(seg,stat):
    if stat=='MG':return float(estimate(seg,CFG).MG)
    if stat=='level':return float(np.mean(seg))
    if stat=='abs_diff':return float(np.mean(np.abs(np.diff(seg))))
    if stat=='norm_diff':return float(np.mean(np.abs(np.diff(seg)))/(np.std(seg)+1e-12))
    if stat=='perm_entropy':return permutation_entropy(seg)
    if stat=='sample_entropy':return sample_entropy(seg)
    r=detrend(seg)
    if stat=='crossings':return float(np.count_nonzero(np.diff(np.signbit(r))))
    if stat=='lag1':return float(np.corrcoef(r[:-1],r[1:])[0,1])
    if stat=='det_std':return float(r.std()/(abs(seg.mean())+1e-12))
    if stat=='spectral_entropy':
        power=abs(np.fft.rfft(r))[1:]**2
        if power.sum()==0:return 0.
        p=power/power.sum();p=p[p>0]
        return float(-np.sum(p*np.log(p))/np.log(len(power)))
    raise KeyError(stat)

def window_features(seg):return {s:scalar_feature(seg,s) for s in STATS}

def variants(x,arm,event=EVENT):
    result={arm:x}
    if arm=='base':
        sc=x.copy();sc[event:]*=10
        sm=x.copy();sm[event:]=np.convolve(x,np.ones(16)/16,mode='full')[:len(x)][event:]
        result.update(scale=sc,smooth=sm)
    return result

def collect(folder,cache,legacy=None):
    if cache.exists():return pd.read_csv(cache)
    rows=[]
    old=pd.read_csv(legacy) if legacy else None
    for p in sorted(folder.glob('logs_*_s*.npz')):
        arm,seed=p.stem[5:].rsplit('_s',1)
        for name,x in variants(np.load(p)['param_norm'],arm).items():
            for a in range(0,len(x)-W+1,S):
                row={s:scalar_feature(x[a:a+W],s) for s in STATS if s not in ['MG','crossings','lag1','det_std'] or old is None}
                if old is not None:
                    g=old[(old.arm==name)&(old.seed==int(seed))&(old.start==a)]
                    if len(g)==1:
                        row.update({s:float(g.iloc[0][s]) for s in ['MG','crossings','lag1','det_std']})
                    else:row.update({s:scalar_feature(x[a:a+W],s) for s in ['MG','crossings','lag1','det_std']})
                row.update(arm=name,seed=int(seed),start=a,end=a+W);rows.append(row)
        print('features',p.name,flush=True)
    d=pd.DataFrame(rows);d.to_csv(cache,index=False);return d

def score_series(g,stat,rule):
    g=g.sort_values('start');v=g[stat].to_numpy(float);ends=g.end.to_numpy()
    ok=ends>WARM;v,ends=v[ok],ends[ok];out=[]
    if rule['detector']=='block':
        M,B=rule['M'],rule['B']
        for k in range(M+B-1,len(v)):
            recent=v[k-M+1:k+1];prior=v[k-M-B+1:k-M+1]
            if not np.isfinite(np.r_[recent,prior]).all():continue
            ref=np.median(prior)
            if abs(ref)>1e-12:out.append((int(ends[k]),rule['sign']*(1-np.median(recent)/ref)))
    else:
        # Past-only warm reference, then Page's one-sided cumulative score.
        warm=ends<=3500
        if warm.sum()<4 or not np.isfinite(v[warm]).all():return []
        mu=np.mean(v[warm]);sd=max(np.std(v[warm]),.01*abs(mu),1e-10);cum=0.
        for end,value in zip(ends[~warm],v[~warm]):
            if not np.isfinite(value):continue
            cum=max(0.,cum+rule['sign']*(mu-value)/sd-rule['drift'])
            out.append((int(end),cum))
    return out

def first_alarm(g,stat,rule):
    for end,value in score_series(g,stat,rule):
        if value>rule['delta']:return end
    return None

def evaluate(frame,stat,rule,event=EVENT,nulls=TEST_NULL):
    rows=[]
    for (arm,seed),g in frame.groupby(['arm','seed']):
        alarm=first_alarm(g,stat,rule);is_event=arm not in nulls
        fa=alarm is not None and (not is_event or alarm<=event)
        hit=is_event and alarm is not None and event<alarm<=event+HORIZON
        rows.append(dict(stat=stat,detector=rule['detector'],arm=arm,seed=seed,event=is_event,event_step=event,alarm=alarm,false_alarm=fa,hit=hit,delay=(alarm-event if hit else np.nan)))
    return pd.DataFrame(rows)

def calibrate(frame,stat,kind):
    choices=[dict(detector='block',M=M,B=B,sign=sign) for M,B,sign in itertools.product((2,3,4),(3,4,6),(1,-1))] if kind=='block' else [dict(detector='cusum',drift=k,sign=s) for k,s in itertools.product((.25,.5,1.),(1,-1))]
    if stat=='MG' and kind=='block':choices=[c for c in choices if c['sign']==1]
    best=None
    for rule in choices:
        maxima=[]
        for _,g in frame[frame.arm.isin(CAL_NULL)].groupby(['arm','seed']):
            vals=[v for _,v in score_series(g,stat,rule)]
            maxima.append(max(vals,default=0.))
        rule['delta']=max(0.,max(maxima))+(0.02 if kind=='block' else .5)
        ev=evaluate(frame[frame.arm.isin(CAL_EVENTS)],stat,rule)
        key=(int(ev.hit.sum()),-float(ev.loc[ev.hit,'delay'].median()) if ev.hit.any() else -1e9)
        if best is None or key>best[0]:best=(key,dict(rule,stat=stat,calibration_hits=int(ev.hit.sum()),calibration_n=len(ev)))
    return best[1]

def main():
    out=H/'new_results';out.mkdir(exist_ok=True)
    r=ROOT/'research_trajectory_reference'
    cal=collect(r/'results_cifar',out/'cal_features.csv',r/'results_detector/cal_windows.csv')
    # Freeze all rules before reading the historical test set or any fresh results.
    rules={s+'_'+kind:calibrate(cal,s,kind) for s in STATS for kind in ['block','cusum']}
    (out/'rules.json').write_text(json.dumps(rules,indent=2))
    test=collect(r/'results_graded',out/'test_features.csv',r/'results_detector/test_windows.csv')
    results=pd.concat([evaluate(test,rule['stat'],rule) for rule in rules.values()],ignore_index=True)
    results.to_csv(out/'baseline_records.csv',index=False)
    rows=[]
    for (stat,kind),g in results.groupby(['stat','detector']):
        a=g[g.arm.isin(STRONG)];row=dict(stat=stat,detector=kind,hits=int(a.hit.sum()),n=len(a),delay=float(a.delay.median()))
        for arm in TEST_NULL:row['alarms_'+arm]=int(g[g.arm==arm].false_alarm.sum())
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(out/'baseline_summary.csv',index=False)
    results.groupby(['stat','detector','arm']).agg(n=('hit','size'),hits=('hit','sum'),alarms=('false_alarm','sum')).to_csv(out/'baseline_by_arm.csv')
    # Timing includes each feature's own preprocessing, at equal thread count.
    x=np.load(r/'results_graded/logs_base_s10.npz')['param_norm'][2000:3000];tim=[]
    for stat in STATS:
        scalar_feature(x,stat)
        for rep in range(15):
            t=time.perf_counter();value=scalar_feature(x,stat);tim.append(dict(stat=stat,rep=rep,seconds=time.perf_counter()-t,value=value))
    pd.DataFrame(tim).to_csv(out/'baseline_timing.csv',index=False)
    legacy=pd.read_csv(r/'results_detector/test_per_run.csv');mg=results[(results.stat=='MG')&(results.detector=='block')]
    m=mg.merge(legacy[legacy.stat=='MG'],on=['arm','seed'],suffixes=('_new','_old'))
    assert (m.hit_new==m.hit_old).all() and (m.false_alarm_new==m.false_alarm_old).all()
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()

"""Choose compact harmonic versus multi-frequency forecast model from one scalar trace."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits
from run import H,R,ARMS,loss,features

def harmonic_design(t,p,k=4):
    ph=2*np.pi*np.outer(t,np.arange(1,k+1))/p
    return np.column_stack([np.ones(len(t)),np.sin(ph),np.cos(ph)])
def base_period(x):
    x=x-x.mean();v=x.var();scores=[np.mean((x[p:]-x[:-p])**2)/(2*v) for p in range(30,min(601,len(x)//2))]
    p=30+int(np.argmin(scores));origin=x[:-601]
    def error(lag):return np.mean((np.interp(np.arange(len(origin))+lag,np.arange(len(x)),x)-origin)**2)/v
    return float(minimize_scalar(error,bounds=(max(30,p-1),min(600,p+1)),method='bounded').x)
def simple_predict(x,h,k=4):
    p=base_period(x);t=np.arange(len(x));A=harmonic_design(t,p,k);w=np.linalg.lstsq(A,x,rcond=None)[0]
    return harmonic_design(np.arange(len(x),len(x)+h),p,k)@w,p
def complex_predict(x,h,k=4):
    # select four nonzero FFT peaks and refit their amplitudes/phases on all context samples
    n=len(x);z=x-x.mean();power=np.abs(np.fft.rfft(z*np.hanning(n)))[1:]**2
    freqs=np.fft.rfftfreq(n)[1:];peaks,_=find_peaks(power,distance=3)
    ids=peaks[np.argsort(power[peaks])[-k:]] if len(peaks) else np.array([power.argmax()])
    refined=[]
    for idx in ids:
        def error(f):
            tt=np.arange(n);a=np.column_stack([np.ones(n),np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt)])
            return np.mean((a@np.linalg.lstsq(a,x,rcond=None)[0]-x)**2)
        refined.append(minimize_scalar(error,bounds=(max(1e-9,freqs[idx]-1/n),min(.5,freqs[idx]+1/n)),method='bounded',options={'xatol':1e-9}).x)
    freqs=np.array(refined)
    A=np.column_stack([np.ones(n)]+[q for f in freqs for q in (np.sin(2*np.pi*f*np.arange(n)),np.cos(2*np.pi*f*np.arange(n)))])
    w=np.linalg.lstsq(A,x,rcond=None)[0];tt=np.arange(n,n+h)
    B=np.column_stack([np.ones(h)]+[q for f in freqs for q in (np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt))])
    return B@w,freqs
def collect(seed,arm,horizon=512,starts=(4096,8192,16384)):
    raw=np.load(R/f'research_generator/results/obs_{arm}_s{seed}.npz')['obs'][:,0];rows=[]
    for start in starts:
        x=raw[start:start+4096];y=raw[start+4096:start+4096+horizon]
        if len(y)<horizon:continue
        mu,sd=x.mean(),x.std();x=(x-mu)/(sd+1e-12);y=(y-mu)/(sd+1e-12)
        tic=time.perf_counter();f=features(x);f['feature_seconds']=time.perf_counter()-tic
        tic=time.perf_counter();a,p=simple_predict(x,horizon);f['simple']=loss(a,y);f['simple_seconds']=time.perf_counter()-tic
        tic=time.perf_counter();b,fr=complex_predict(x,horizon);f['complex']=loss(b,y);f['complex_seconds']=time.perf_counter()-tic
        pow=abs(np.fft.rfft(x*np.hanning(len(x))))**2;peaks,_=find_peaks(pow,distance=3,height=.01*pow.max())
        f.update(seed=seed,arm=arm,start=start,horizon=horizon,peak_count=len(peaks))
        f['holdout_simple']=loss(simple_predict(x[:-horizon],horizon)[0],x[-horizon:])
        f['holdout_complex']=loss(complex_predict(x[:-horizon],horizon)[0],x[-horizon:]);rows.append(f)
    return rows
def main():
    rows=[r for seed in [1,2,3,4,5] for arm in ARMS for r in collect(seed,arm)]
    d=pd.DataFrame(rows);d.to_csv(H/'model_order_records.csv',index=False)
    pilot=d[d.seed.isin([1,2])];test=d[d.seed.isin([3,4,5])]
    rules={}
    for feature in ['MG','entropy','increments','recurrence','peak_count']:
        vals=np.sort(pilot[feature].dropna().unique());best=None
        for sign in [1,-1]:
            for t in np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf]:
                use=(pilot[feature]<=t) if sign==1 else (pilot[feature]>t);pred=np.where(use,pilot.simple,pilot.complex);score=pred.mean()
                if best is None or score<best[0]:best=(score,dict(feature=feature,sign=sign,threshold=float(t)))
        rules[feature]=best[1]
    (H/'model_order_rules.json').write_text(json.dumps(rules,indent=2))
    out=[]
    for name,rule in rules.items():
        use=(test[rule['feature']]<=rule['threshold']) if rule['sign']==1 else (test[rule['feature']]>rule['threshold'])
        pred=np.where(use,test.simple,test.complex);out.append(dict(method=name,mse=float(pred.mean()),simple=float(test.simple.mean()),complex=float(test.complex.mean()),wins=int((pred<=np.minimum(test.simple,test.complex)+1e-12).sum()),n=len(test)))
    out += [dict(method='holdout',mse=np.where(test.holdout_simple<test.holdout_complex,test.simple,test.complex).mean(),n=len(test)),dict(method='fixed_simple',mse=test.simple.mean(),simple=test.simple.mean(),complex=test.complex.mean(),wins=0,n=len(test)),dict(method='fixed_complex',mse=test.complex.mean(),simple=test.simple.mean(),complex=test.complex.mean(),wins=0,n=len(test)),dict(method='oracle',mse=np.minimum(test.simple,test.complex).mean(),simple=test.simple.mean(),complex=test.complex.mean(),wins=0,n=len(test))]
    s=pd.DataFrame(out);s.to_csv(H/'model_order_summary.csv',index=False);print(s.round(4).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

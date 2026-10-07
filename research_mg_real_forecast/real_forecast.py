from pathlib import Path
import urllib.request,gzip,hashlib,json,zipfile,io
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import sys
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau='acorr',k_neighbors=20,theiler='autocorr',theiler_cap=320)
URLS={'traffic':'https://archive.ics.uci.edu/static/public/492/metro%2Binterstate%2Btraffic%2Bvolume.zip','energy':'https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip'}
def load(name):
 p=H/f'{name}.raw'
 if not p.exists():urllib.request.urlretrieve(URLS[name],p)
 raw=p.read_bytes();(H/f'{name}.sha256').write_text(hashlib.sha256(raw).hexdigest())
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:
  if name=='traffic':
   d=pd.read_csv(io.BytesIO(gzip.decompress(archive.read('Metro_Interstate_Traffic_Volume.csv.gz'))))
   d['date_time']=pd.to_datetime(d.date_time)
   # Duplicate weather entries are one timestamp, not successive measurements.
   series=d.groupby('date_time').traffic_volume.mean().sort_index().asfreq('h')
   # Long gaps remain missing; no interpolation through future values.
   return series.ffill(limit=3).to_numpy(float)
  return pd.read_csv(io.BytesIO(archive.read('energydata_complete.csv'))).Appliances.to_numpy(float)
def clean(x):return np.asarray(x,dtype=float)
def seasonal(x,h,p):return np.resize(x[-p:],h)
def ar(x,h,lags):
 z=(x-x.mean())/(x.std()+1e-12);w=np.lib.stride_tricks.sliding_window_view(z,lags+1);X=w[:,:lags];y=w[:,lags];m=Ridge(alpha=1.).fit(X,y);out=list(z)
 for _ in range(h):out.append(float(m.predict(np.asarray(out[-lags:]).reshape(1,-1))[0]))
 return np.asarray(out[-h:])*x.std()+x.mean()
def features(x,p):
 r=estimate(x,CFG,seed=123);y=x-x.mean();power=np.abs(np.fft.rfft(y))[1:]**2;power/=max(power.sum(),1e-30);v=np.var(x)
 return dict(MG=float(r.MG) if not r.degenerate else np.nan,entropy=float(-np.sum(power[power>0]*np.log(power[power>0]))/np.log(len(power))),increments=float(np.mean(np.diff(x)**2)/(2*v)),recurrence=float(min(np.mean((x[l:]-x[:-l])**2)/(2*v) for l in range(max(2,p//4),min(len(x)-1,2*p)+1))),peak=float(power.max()/power.sum()))
def collect(name):
 x=clean(load(name));p=24 if name=='traffic' else 144;context=14*p;h=p;starts=np.arange(context,len(x)-h,max(1,p//2));rows=[]
 for i,start in enumerate(starts):
  c=x[start-context:start];y=x[start:start+h]
  if not np.isfinite(np.r_[c,y]).all() or c.std()<1e-10:continue
  row=dict(dataset=name,origin=int(start),block=int(i),period=p,**features(c,p));row['seasonal_mse']=float(np.mean((seasonal(c,h,p)-y)**2));row['ar_mse']=float(np.mean((ar(c,h,2*p)-y)**2));rows.append(row)
  if i%100==0:print(name,i,'/',len(starts),flush=True)
 d=pd.DataFrame(rows);d.to_csv(H/f'{name}_windows.csv',index=False);return d
def fit(d):
 rules={};cal=d.iloc[:int(.4*len(d))]
 for f in ['MG','entropy','increments','recurrence','peak']:
  vals=np.sort(cal[f].dropna().unique());best=None
  for sign in [1,-1]:
   for t in np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf]:
    use=(cal[f]<=t) if sign==1 else (cal[f]>t);use=use|cal[f].isna();score=np.where(use,cal.seasonal_mse,cal.ar_mse).mean()
    if best is None or score<best[0]:best=(score,dict(feature=f,sign=sign,threshold=float(t)))
  rules[f]=best[1]
 return rules
def score(d,rules):
 t=d.iloc[int(.4*len(d)):];rows=[]
 for f,r in rules.items():
  use=(t[f]<=r['threshold']) if r['sign']==1 else (t[f]>r['threshold']);use=use|t[f].isna();p=np.where(use,t.seasonal_mse,t.ar_mse);rows.append(dict(feature=f,mse=p.mean(),regret=np.mean(p-np.minimum(t.seasonal_mse,t.ar_mse))))
 rows += [dict(feature='fixed_seasonal',mse=t.seasonal_mse.mean(),regret=np.mean(t.seasonal_mse-np.minimum(t.seasonal_mse,t.ar_mse)),),dict(feature='fixed_ar',mse=t.ar_mse.mean(),regret=np.mean(t.ar_mse-np.minimum(t.seasonal_mse,t.ar_mse)),),dict(feature='oracle',mse=np.minimum(t.seasonal_mse,t.ar_mse).mean(),regret=0.)]
 return pd.DataFrame(rows)
def main():
 out=[]
 for name in ['traffic','energy']:
  d=collect(name);r=fit(d);(H/f'{name}_rules.json').write_text(json.dumps(r,indent=2));s=score(d,r);s.insert(0,'dataset',name);s.to_csv(H/f'{name}_summary.csv',index=False);out.append(s)
 pd.concat(out).to_csv(H/'summary.csv',index=False);print(pd.concat(out).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

from pathlib import Path
import urllib.request,gzip,hashlib,json,time,warnings
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import sys
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau='acorr',k_neighbors=20,theiler='autocorr',theiler_cap=320)
URLS={
 'traffic':'https://archive.ics.uci.edu/static/public/492/Metro_Interstate_Traffic_Volume.csv.gz',
 'energy':'https://archive.ics.uci.edu/static/public/374/energydata_complete.csv'}
def download(name):
 p=H/f'{name}.raw';
 if not p.exists(): urllib.request.urlretrieve(URLS[name],p)
 return p
def load(name):
 p=download(name);raw=p.read_bytes();(H/f'{name}.sha256').write_text(hashlib.sha256(raw).hexdigest())
 if name=='traffic':
  import io
  d=pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(raw)))
  return d.sort_values('date_time').traffic_volume.to_numpy(float)
 d=pd.read_csv(p);return d.Appliances.to_numpy(float)
def clean(x):
 x=np.asarray(x,float);x=pd.Series(x).interpolate().bfill().ffill().to_numpy();return x
def normalize_context(x):return (x-x.mean())/(x.std()+1e-12)
def seasonal(x,h,p):return np.resize(x[-p:],h)
def ar(x,h,lags):
 z=normalize_context(x);w=np.lib.stride_tricks.sliding_window_view(z,lags+1);X=w[:,:lags];y=w[:,lags]
 model=Ridge(alpha=1.).fit(X,y);out=list(z)
 for _ in range(h):out.append(float(model.predict(np.asarray(out[-lags:]).reshape(1,-1))[0]))
 return np.asarray(out[-h:])*x.std()+x.mean()
def features(x,p):
 r=estimate(x,CFG,seed=123);y=x-x.mean();power=np.abs(np.fft.rfft(y))[1:]**2;power/=max(power.sum(),1e-30)
 ent=float(-np.sum(power[power>0]*np.log(power[power>0]))/np.log(len(power)))
 inc=float(np.mean(np.diff(x)**2)/(2*np.var(x)))
 rec=float(min(np.mean((x[l:]-x[:-l])**2)/(2*np.var(x)) for l in range(max(2,p//4),min(len(x)-1,2*p)+1)))
 peak=float(power.max()/power.sum())
 return dict(MG=float(r.MG) if not r.degenerate else np.nan,entropy=ent,increments=inc,recurrence=rec,peak=peak,mg_degenerate=bool(r.degenerate))
def collect(name):
 x=clean(load(name));p=24 if name=='traffic' else 144;context=14*p;h=p
 starts=np.arange(context,len(x)-h,max(1,p//2));rows=[]
 for i,start in enumerate(starts):
  c=x[start-context:start];y=x[start:start+h]
  row=dict(dataset=name,origin=int(start),block=int(i),period=p,**features(c,p))
  preds={'seasonal':seasonal(c,h,p),'ar':ar(c,h,2*p)}
  row.update(seasonal_mse=float(np.mean((preds['seasonal']-y)**2)),ar_mse=float(np.mean((preds['ar']-y)**2)))
  rows.append(row)
  if i%100==0:print(name,i,'/',len(starts),flush=True)
 d=pd.DataFrame(rows);d.to_csv(H/f'{name}_windows.csv',index=False);return d
def fit_rules(d):
 calibration=d.iloc[:int(len(d)*.4)];rules={}
 for feature in ['MG','entropy','increments','recurrence','peak']:
  vals=np.sort(calibration[feature].dropna().unique());thr=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf];best=None
  for sign in [1,-1]:
   for t in thr:
    choose=(calibration[feature]<=t) if sign==1 else (calibration[feature]>t)
    pred=np.where(choose,calibration.seasonal_mse,calibration.ar_mse)
    score=np.nanmean(pred)
    if best is None or score<best[0]:best=(score,dict(feature=feature,sign=sign,threshold=float(t)))
  rules[feature]=best[1]
 return rules
def score(d,rules):
 test=d.iloc[int(len(d)*.4):].copy();rows=[]
 for feat,rule in rules.items():
  choose=(test[feat]<=rule['threshold']) if rule['sign']==1 else (test[feat]>rule['threshold'])
  pred=np.where(choose,test.seasonal_mse,test.ar_mse)
  rows.append(dict(feature=feat,mse=float(np.nanmean(pred)),regret=float(np.nanmean(pred-np.minimum(test.seasonal_mse,test.ar_mse))),
                  wins=int(np.sum(pred<np.minimum(test.seasonal_mse,test.ar_mse)+1e-12)),n=len(pred)))
 rows += [dict(feature='fixed_seasonal',mse=float(test.seasonal_mse.mean()),regret=float(np.mean(test.seasonal_mse-np.minimum(test.seasonal_mse,test.ar_mse))),wins=0,n=len(test)),
          dict(feature='fixed_ar',mse=float(test.ar_mse.mean()),regret=float(np.mean(test.ar_mse-np.minimum(test.seasonal_mse,test.ar_mse))),wins=0,n=len(test)),
          dict(feature='oracle',mse=float(np.minimum(test.seasonal_mse,test.ar_mse).mean()),regret=0.,wins=0,n=len(test))]
 return pd.DataFrame(rows)
def main():
 allsum=[]
 for name in ['traffic','energy']:
  d=collect(name);rules=fit_rules(d);(H/f'{name}_rules.json').write_text(json.dumps(rules,indent=2));s=score(d,rules);s.insert(0,'dataset',name);s.to_csv(H/f'{name}_summary.csv',index=False);allsum.append(s)
 out=pd.concat(allsum);out.to_csv(H/'summary.csv',index=False);print(out.to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

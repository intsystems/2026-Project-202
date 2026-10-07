import sys,time,json
import numpy as np,pandas as pd
from scipy.ndimage import correlate1d
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from threadpoolctl import threadpool_limits
H=__import__('pathlib').Path(__file__).resolve().parent
def dataset(name):
 folder=H/name;z=__import__('zipfile').ZipFile(folder/'data.zip');out={}
 for split in ['TRAIN','TEST']:
  key=next(k for k in z.namelist() if k.lower().endswith('_'+split.lower()+'.ts'));lines=z.read(key).decode('utf-8-sig').splitlines();on=False;X=[];y=[]
  for line in lines:
   if line.lower().strip()=='@data':on=True;continue
   if not on or not line.strip():continue
   val,label=line.strip().rsplit(':',1);X.append(np.array([float(s) if s!='?' else np.nan for s in val.split(',')]));y.append(label)
  out[split]=(np.array(X),np.array(y))
 return out
def norm(x):return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
def rocket(X):
 rng=np.random.default_rng(90210);feats=[]
 for k in range(1000):
  size=int(rng.choice([7,9,11]));dilation=int(2**rng.uniform(0,np.log2(max(1,(X.shape[1]-1)/(size-1)))));weights=rng.normal(size=size);weights-=weights.mean();bias=rng.uniform(-1,1);pad=rng.random()<.5
  kernel=np.zeros((size-1)*dilation+1);kernel[::dilation]=weights
  z=correlate1d(X,kernel,axis=1,mode='constant',cval=0)+bias
  if not pad:
   left=len(kernel)//2;z=z[:,left:X.shape[1]-left]
  feats.extend([z.max(1),(z>0).mean(1)])
 return np.column_stack(feats)
def dtw(a,b,fraction):
 n=a.shape[1];radius=max(1,int(n*fraction));D=np.full((len(a),len(b)),np.inf)
 for start in range(0,len(a),64):
  x=a[start:start+64];prev=np.full((len(x),len(b),n+1),np.inf);prev[:,:,0]=0
  for i in range(n):
   curr=np.full_like(prev,np.inf)
   for j in range(max(0,i-radius),min(n,i+radius+1)):
    cost=(x[:,i,None]-b[None,:,j])**2
    curr[:,:,j+1]=cost+np.minimum(np.minimum(prev[:,:,j+1],prev[:,:,j]),curr[:,:,j])
   prev=curr
  D[start:start+len(x)]=prev[:,:,-1]
 return D
def one(name):
 raw=dataset(name);x,y=raw['TRAIN'];xt,yt=raw['TEST'];x=norm(x);xt=norm(xt);folder=H/name
 cv=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=90210).split(x,y));rows=[];preds=[]
 scores=[]
 for radius in [.1,.2]:
  scores.append(np.mean([balanced_accuracy_score(y[va],y[tr][dtw(x[va],x[tr],radius).argmin(1)]) for tr,va in cv]))
 radius=[.1,.2][int(np.argmax(scores))];pred=y[dtw(xt,x,radius).argmin(1)]
 rows.append(dict(method='DTW',balanced_accuracy=balanced_accuracy_score(yt,pred),accuracy=accuracy_score(yt,pred),config=radius))
 preds.extend(dict(method='DTW',row=i,y=label,pred=p) for i,(label,p) in enumerate(zip(yt,pred)))
 tic=time.perf_counter();z=rocket(x);zz=rocket(xt);seconds=time.perf_counter()-tic
 f=pd.read_csv(folder/'features_train.csv');ft=pd.read_csv(folder/'features_test.csv');mg=[c for c in f if c.startswith('MG_')]
 for method,a,b in [('rocket',z,zz),('rocket_MG',np.column_stack([z,f[mg]]),np.column_stack([zz,ft[mg]]))]:
  make=lambda alpha:make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),RidgeClassifier(alpha=alpha,class_weight='balanced'))
  alphas=[.01,.1,1,10,100];scores=[np.mean(cross_val_score(make(alpha),a,y,cv=cv,scoring='balanced_accuracy')) for alpha in alphas];alpha=alphas[int(np.argmax(scores))];m=make(alpha).fit(a,y);pred=m.predict(b)
  rows.append(dict(method=method,balanced_accuracy=balanced_accuracy_score(yt,pred),accuracy=accuracy_score(yt,pred),config=alpha))
  preds.extend(dict(method=method,row=i,y=label,pred=p) for i,(label,p) in enumerate(zip(yt,pred)))
 pd.DataFrame(rows).to_csv(folder/'strong_summary.csv',index=False);pd.DataFrame(preds).to_csv(folder/'strong_predictions.csv',index=False);(folder/'rocket_time.json').write_text(json.dumps(dict(all_extraction_seconds=seconds,kernels=1000)))
 print(name,pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  for name in (sys.argv[1:] or ['TwoLeadECG','ECG200']):one(name)

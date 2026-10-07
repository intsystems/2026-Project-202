from pathlib import Path
import sys,urllib.request,zipfile,io,json,time,hashlib,itertools
import numpy as np,pandas as pd
from scipy.stats import skew,kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
URL='https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip'
def archive():
 p=H/'uci_har.zip'
 if not p.exists():urllib.request.urlretrieve(URL,p)
 (H/'data_source.json').write_text(json.dumps(dict(url=URL,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
 outer=zipfile.ZipFile(p)
 return zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) if 'UCI HAR Dataset.zip' in outer.namelist() else outer
def data(split):
 z=archive();prefix='UCI HAR Dataset/'+split+'/'
 def read(path):return np.loadtxt(io.BytesIO(z.read(prefix+path)))
 s=read('subject_'+split+'.txt').astype(int);y=read('y_'+split+'.txt').astype(int);full=read('X_'+split+'.txt')
 signals=[]
 for base in ['total_acc','body_gyro']:
  a=np.stack([read('Inertial Signals/'+base+'_'+axis+'_'+split+'.txt') for axis in 'xyz'],axis=-1);signals.append(np.linalg.norm(a,axis=-1))
 return np.stack(signals,axis=1),s,y,full
def single(x):
 v=max(np.var(x),1e-30);z=(x-x.mean())/np.sqrt(v);p=abs(np.fft.rfft(z))[1:]**2;p=p/max(p.sum(),1e-30)
 patterns=np.argsort(np.lib.stride_tricks.sliding_window_view(z,4),axis=1,kind='stable');_,counts=np.unique(patterns,axis=0,return_counts=True);pr=counts/counts.sum()
 row=dict(mean=x.mean(),std=x.std(),min=x.min(),max=x.max(),median=np.median(x),iqr=np.quantile(x,.75)-np.quantile(x,.25),rms=np.sqrt(np.mean(x*x)),skew=float(skew(x)),kurt=float(kurtosis(x)),
   increments=np.mean(np.diff(z)**2)/2,entropy=-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p)),peak=p.max(),frequency=float(p.argmax()+1)/len(x),recurrence=min(np.mean((z[l:]-z[:-l])**2)/2 for l in range(4,65)),perm_entropy=-np.sum(pr*np.log(pr))/np.log(24),crossings=float(np.sum(np.diff(np.signbit(z)))),trend=np.polyfit(np.linspace(-1,1,len(z)),z,1)[0])
 for l in [1,4,8]:row['acf'+str(l)]=np.corrcoef(z[:-l],z[l:])[0,1]
 return row
def extract(signals,split):
 p=H/f'features_{split}.csv'
 if p.exists():return pd.read_csv(p)
 rows=[];times=dict(cheap=0.,MG=0.,PR=0.)
 for i,record in enumerate(signals):
  row={}
  for c,x in enumerate(record):
   tic=time.perf_counter();row.update({f'c{c}_{k}':v for k,v in single(x).items()});times['cheap']+=time.perf_counter()-tic
   for E,tau in itertools.product([6,10],[1,2]):
    tic=time.perf_counter();r=estimate(x,EstimatorConfig(max_E=E,tau=tau,k_neighbors=10,theiler=10,theiler_cap=10),seed=123);row[f'c{c}_MG_{E}_{tau}']=r.MG if not r.degenerate else np.nan;times['MG']+=time.perf_counter()-tic
    tic=time.perf_counter();n=len(x)-(E-1)*tau;a=np.column_stack([x[j*tau:j*tau+n] for j in range(E)]);eig=np.linalg.eigvalsh(np.cov(a.T));row[f'c{c}_PR_{E}_{tau}']=eig.sum()**2/max(np.sum(eig**2),1e-30);times['PR']+=time.perf_counter()-tic
  rows.append(row)
  if i%1000==0:print(split,'features',i,len(signals),flush=True)
 d=pd.DataFrame(rows);d.to_csv(p,index=False);(H/f'timing_{split}.json').write_text(json.dumps(dict(n=len(d),times=times),indent=2));return d
def learner(i):
 if i<2:return make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),SVC(C=[1,10][i],gamma='scale'))
 return make_pipeline(SimpleImputer(add_indicator=True),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=[15,31][i-2],learning_rate=.1,l2_regularization=1,random_state=42))
def subject_score(y,pred,s):return float(pd.DataFrame(dict(s=s,ok=y==pred)).groupby('s').ok.mean().mean())
def main():
 signals,s,y,full=data('train');x=extract(signals,'train');cheap=[c for c in x if '_MG_' not in c and '_PR_' not in c];mg=[c for c in x if '_MG_' in c];pr=[c for c in x if '_PR_' in c]
 sets={'MG':mg,'cheap':cheap,'cheap_MG':cheap+mg,'cheap_PR':cheap+pr,'cheap_PR_MG':cheap+pr+mg,'full561':None};fit=s%3!=0;valid=~fit;choices={};development=[]
 for name,cols in sets.items():
  xx=full if cols is None else x[cols].to_numpy()
  for i in range(4):
   model=learner(i);model.fit(xx[fit],y[fit]);pred=model.predict(xx[valid]);score=subject_score(y[valid],pred,s[valid]);development.append(dict(method=name,classifier=i,subject_accuracy=score))
  choices[name]=max([r for r in development if r['method']==name],key=lambda r:(r['subject_accuracy'],-r['classifier']))['classifier']
 pd.DataFrame(development).to_csv(H/'development.csv',index=False);(H/'frozen_choices.json').write_text(json.dumps(choices,indent=2));print('frozen',choices,flush=True)
 sig,ss,yy,fulltest=data('test');xt=extract(sig,'test');results=[];predictions=[]
 for name,cols in sets.items():
  xx=full if cols is None else x[cols].to_numpy();tt=fulltest if cols is None else xt[cols].to_numpy();m=learner(choices[name]);tic=time.perf_counter();m.fit(xx,y);training=time.perf_counter()-tic;tic=time.perf_counter();pred=m.predict(tt);inference=time.perf_counter()-tic
  results.append(dict(method=name,classifier=choices[name],accuracy=accuracy_score(yy,pred),macro_f1=f1_score(yy,pred,average='macro'),subject_accuracy=subject_score(yy,pred,ss),train_seconds=training,inference_seconds=inference,n_features=xx.shape[1]));predictions.extend(dict(method=name,subject=int(sid),row=j,y=int(label),pred=int(p)) for j,(sid,label,p) in enumerate(zip(ss,yy,pred)));print(results[-1],flush=True)
 pd.DataFrame(results).to_csv(H/'summary.csv',index=False);pd.DataFrame(predictions).to_csv(H/'predictions.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

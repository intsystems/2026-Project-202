from pathlib import Path
import sys,urllib.request,zipfile,io,json,time,hashlib,itertools
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'research_mg_har'))
import importlib.util
_spec=importlib.util.spec_from_file_location('har_feature_library',R/'research_mg_har/run.py')
_har=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_har)
single=_har.single
sys.path.insert(0,str(R/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
NAMES=['ECG200','ECG5000','TwoLeadECG','Wafer','FordA','ItalyPowerDemand','ChlorineConcentration','Earthquakes']
def dataset(name):
 folder=H/name;folder.mkdir(exist_ok=True);p=folder/'data.zip';url=f'https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip'
 if not p.exists():urllib.request.urlretrieve(url,p)
 (folder/'source.json').write_text(json.dumps(dict(url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
 z=zipfile.ZipFile(p);out={}
 for split in ['TRAIN','TEST']:
  key=next(k for k in z.namelist() if k.lower().endswith('_'+split.lower()+'.ts'));lines=z.read(key).decode('utf-8-sig').splitlines();active=False;X=[];y=[]
  for line in lines:
   if line.lower().strip()=='@data':active=True;continue
   if not active or not line.strip() or line.startswith('#'):continue
   values,label=line.strip().rsplit(':',1);X.append(np.array([float(s) if s!='?' else np.nan for s in values.split(',')]));y.append(label)
  a=np.array(X);assert a.ndim==2 and np.isfinite(a).all();out[split]=(a,np.array(y))
 return out
def features(X,path):
 if path.exists():return pd.read_csv(path)
 rows=[];cost={'cheap':0.,'PR':0.,'MG':0.}
 for i,x in enumerate(X):
  tic=time.perf_counter();row=single(x);cost['cheap']+=time.perf_counter()-tic
  for E,tau in itertools.product([4,8,12,20],[1,2]):
   T=min(40,max(10,(E-1)*tau));tic=time.perf_counter();r=estimate(x,EstimatorConfig(max_E=E,tau=tau,k_neighbors=10,theiler=T,theiler_cap=T),seed=123);row[f'MG_{E}_{tau}']=r.MG if not r.degenerate else np.nan;cost['MG']+=time.perf_counter()-tic
   tic=time.perf_counter();n=len(x)-(E-1)*tau
   if n>1:
    a=np.column_stack([x[j*tau:j*tau+n] for j in range(E)]);ev=np.linalg.eigvalsh(np.cov(a.T));row[f'PR_{E}_{tau}']=ev.sum()**2/max(np.sum(ev**2),1e-30)
   else:row[f'PR_{E}_{tau}']=np.nan
   cost['PR']+=time.perf_counter()-tic
  rows.append(row)
  if i%1000==0:print(path.parent.name,path.stem,i,len(X),flush=True)
 d=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan);d.to_csv(path,index=False);path.with_suffix('.timing.json').write_text(json.dumps(cost,indent=2));return d
def model(i):
 clf=SVC(C=[1,10][i],class_weight='balanced') if i<2 else ExtraTreesClassifier(n_estimators=200,min_samples_leaf=[1,3][i-2],class_weight='balanced',random_state=90210,n_jobs=1)
 return make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),clf)
def one(name):
 folder=H/name
 if (folder/'summary.csv').exists():return
 raw=dataset(name);X,y=raw['TRAIN'];xt,yt=raw['TEST'];f=features(X,folder/'features_train.csv');mg=[c for c in f if c.startswith('MG_')];pr=[c for c in f if c.startswith('PR_')];cheap=[c for c in f if c not in mg+pr]
 sets={'MG':mg,'cheap':cheap,'cheap_PR':cheap+pr,'cheap_PR_MG':cheap+pr+mg}
 cv=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=90210).split(X,y));choices={};dev=[]
 for feature,cols in sets.items():
  vals=[]
  for i in range(4):
   score=float(cross_val_score(model(i),f[cols].to_numpy(),y,cv=cv,scoring='balanced_accuracy').mean());vals.append(score);dev.append(dict(method=feature,classifier=i,score=score))
  choices[feature]=int(np.argmax(vals))
 norm=lambda a:(a-a.mean(axis=1,keepdims=True))/np.maximum(a.std(axis=1,keepdims=True),1e-12)
 Xn=norm(X);tn=norm(xt)
 rawscores=[float(cross_val_score(model(i),Xn,y,cv=cv,scoring='balanced_accuracy').mean()) for i in [0,1]];choices['raw_SVC']=int(np.argmax(rawscores));
 (folder/'frozen_choices.json').write_text(json.dumps(choices));pd.DataFrame(dev).to_csv(folder/'development.csv',index=False)
 ft=features(xt,folder/'features_test.csv');rows=[];preds=[]
 methods=[(key,model(choices[key]),f[cols].to_numpy(),ft[cols].to_numpy()) for key,cols in sets.items()]
 # Matched classifier ablation: select MG model with training CV, use same classifier without MG.
 methods +=[('matched_no_MG',model(choices['cheap_PR_MG']),f[cheap+pr].to_numpy(),ft[cheap+pr].to_numpy()),('raw_SVC',model(choices['raw_SVC']),Xn,tn),('raw_1NN',KNeighborsClassifier(n_neighbors=1),Xn,tn)]
 for name2,m,a,b in methods:
  tic=time.perf_counter();m.fit(a,y);train=time.perf_counter()-tic;tic=time.perf_counter();pred=m.predict(b);infer=time.perf_counter()-tic
  rows.append(dict(dataset=name,method=name2,balanced_accuracy=balanced_accuracy_score(yt,pred),accuracy=accuracy_score(yt,pred),macro_f1=f1_score(yt,pred,average='macro'),train_seconds=train,inference_seconds=infer,features=a.shape[1],n=len(yt)));preds.extend(dict(method=name2,row=i,y=label,pred=yp) for i,(label,yp) in enumerate(zip(yt,pred)))
 pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False);pd.DataFrame(preds).to_csv(folder/'predictions.csv',index=False);print(pd.DataFrame(rows)[['method','balanced_accuracy','accuracy']].to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  for name in (sys.argv[1:] or NAMES):
   try:one(name)
   except Exception as e:
    (H/f'{name}_failure.txt').write_text(repr(e));print('FAILED',name,repr(e),flush=True)

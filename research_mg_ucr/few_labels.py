from strong import H,dataset,norm,rocket
import numpy as np,pandas as pd,time,json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,recall_score
from threadpoolctl import threadpool_limits
def main():
 folder=H/'Wafer';raw=dataset('Wafer');x,y=raw['TRAIN'];xt,yt=raw['TEST'];x=norm(x);xt=norm(xt)
 f=pd.read_csv(folder/'features_train.csv');ft=pd.read_csv(folder/'features_test.csv');mg=[c for c in f if c.startswith('MG_')];pr=[c for c in f if c.startswith('PR_')];cheap=[c for c in f if c not in mg+pr]
 file=folder/'rocket_features.npz'
 if file.exists():cache=np.load(file);rx,rt=cache['train'],cache['test']
 else:
  tic=time.perf_counter();rx=rocket(x);rt=rocket(xt);np.savez_compressed(file,train=rx,test=rt);(folder/'rocket_extraction.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tic,n=len(x)+len(xt))))
 sets={'MG':mg,'cheap':cheap,'cheap_PR':cheap+pr,'cheap_PR_MG':cheap+pr+mg};rows=[]
 for size in [20,50,100]:
  for seed in range(10):
   ids,_=train_test_split(np.arange(len(y)),train_size=size,stratify=y,random_state=seed)
   for name in list(sets)+['raw_SVC','rocket','raw_1NN']:
    if name in sets:a=f[sets[name]].to_numpy();b=ft[sets[name]].to_numpy();cls=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,class_weight='balanced',random_state=42,n_jobs=1)
    elif name=='rocket':a,b=rx,rt;cls=RidgeClassifier(alpha=1,class_weight='balanced')
    elif name=='raw_SVC':a,b=x,xt;cls=SVC(C=1,class_weight='balanced')
    else:a,b=x,xt;cls=KNeighborsClassifier(n_neighbors=1)
    model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),cls) if name!='raw_1NN' else cls
    tic=time.perf_counter();model.fit(a[ids],y[ids]);train=time.perf_counter()-tic;tic=time.perf_counter();pred=model.predict(b);infer=time.perf_counter()-tic
    rows.append(dict(size=size,seed=seed,method=name,balanced_accuracy=balanced_accuracy_score(yt,pred),accuracy=accuracy_score(yt,pred),train_seconds=train,inference_seconds=infer))
   pd.DataFrame(rows).to_csv(folder/'few_label_records.csv',index=False)
  print(size,flush=True)
 d=pd.DataFrame(rows);s=d.groupby(['size','method']).balanced_accuracy.agg(['mean','std']);s.to_csv(folder/'few_label_summary.csv');print(s.round(4).to_string())
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

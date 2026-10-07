from strong import H,dataset,norm
import numpy as np,pandas as pd,json,time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
def cheap_ties(x):
 return np.array([[len(np.unique(v))/len(v),np.mean(np.diff(v)==0),max(np.unique(v,return_counts=True)[1])/len(v)] for v in x])
def main():
 folder=H/'Wafer';raw=dataset('Wafer');x,y=raw['TRAIN'];xt,yt=raw['TEST'];f=pd.read_csv(folder/'features_train.csv');ft=pd.read_csv(folder/'features_test.csv');mg=[c for c in f if c.startswith('MG_')]
 # Values versus missingness: no imputation indicator for values-only; missing fill can still carry information, explicitly retain caveat.
 sets={'ties_only':(cheap_ties(x),cheap_ties(xt)),'MG_flags_only':(f[mg].isna().to_numpy(),ft[mg].isna().to_numpy()),'MG_values':(f[mg].to_numpy(),ft[mg].to_numpy())}
 rows=[]
 for size in [20,50,100,1000]:
  for seed in range(10 if size<1000 else 1):
   ids=train_test_split(np.arange(len(y)),train_size=size,stratify=y,random_state=seed)[0] if size<1000 else np.arange(len(y))
   for name,(a,b) in sets.items():
    m=make_pipeline(SimpleImputer(keep_empty_features=True),ExtraTreesClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=1));m.fit(a[ids],y[ids]);pred=m.predict(b);rows.append(dict(size=size,seed=seed,method=name,balanced_accuracy=balanced_accuracy_score(yt,pred)))
 d=pd.DataFrame(rows);d.to_csv(folder/'tie_audit.csv',index=False);s=d.groupby(['size','method']).balanced_accuracy.mean();s.to_csv(folder/'tie_summary.csv');print(s)
 (folder/'artifact.json').write_text(json.dumps(dict(finding='MG missingness perfectly separates training labels; tied/quantized plateaus require a simple ties baseline.',
 train_cross_tab=pd.crosstab(y,f[mg].isna().sum(1)).to_dict()),indent=2,default=int))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()

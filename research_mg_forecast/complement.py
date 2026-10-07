from run import H,R,ARMS,features,periodic,ar,loss
import json,time
import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeRegressor,export_text
from threadpoolctl import threadpool_limits
SETS={'tree_MG':['MG'],'tree_recurrence':['recurrence'],'tree_MG_recurrence':['MG','recurrence'],
      'tree_cheap':['entropy','increments','recurrence'],'tree_cheap_MG':['entropy','increments','recurrence','MG']}
def main():
    pilot=pd.read_csv(H/'pilot.csv');y=pilot.ar-pilot.periodic;trees={};texts={}
    for name,cols in SETS.items():
        x=pilot[cols];fill=x.median();m=DecisionTreeRegressor(max_depth=2,min_samples_leaf=3,random_state=0).fit(x.fillna(fill),y)
        trees[name]=(m,fill);texts[name]=export_text(m,feature_names=cols)
    (H/'complement_rules.json').write_text(json.dumps(texts,indent=2))
    rules=json.loads((H/'rules.json').read_text());rows=[];feats=[]
    for seed in [3,4,5]:
      for arm in ARMS:
        raw=np.load(R/f'research_generator/results/obs_{arm}_s{seed}.npz')['obs'][:,0]
        pref=raw[24576:28672];x=(pref-pref.mean())/pref.std();y=(raw[28672:29184]-pref.mean())/pref.std()
        row=dict(seed=seed,arm=arm,**features(x));choose={}
        for name,(tree,fill) in trees.items():
            inp=pd.DataFrame([row])[SETS[name]].fillna(fill);choose[name]='periodic' if tree.predict(inp)[0]>0 else 'ar'
        for name,r in rules.items():choose[name]='periodic' if (row[name]<=r['threshold'] if r['sign']==1 else row[name]>r['threshold']) else 'ar'
        for name,fn in [('periodic',periodic),('ar',ar)]:row[name]=loss(fn(x,512),y);row[name+'_holdout']=loss(fn(x[:-512],512),x[-512:])
        choose.update(holdout='periodic' if row['periodic_holdout']<row['ar_holdout'] else 'ar',fixed_ar='ar',fixed_periodic='periodic',oracle='periodic' if row['periodic']<row['ar'] else 'ar')
        feats.append(row)
        for method,model in choose.items():rows.append(dict(seed=seed,arm=arm,method=method,model=model,error=row[model]))
        print(seed,arm,flush=True)
    pd.DataFrame(feats).to_csv(H/'complement_features.csv',index=False)
    d=pd.DataFrame(rows);d.to_csv(H/'complement_decisions.csv',index=False)
    summary=d.groupby('method').error.agg(['mean','median']);summary.to_csv(H/'complement_summary.csv');print(summary.round(4))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()

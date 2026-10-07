from run import H,R,ARMS,features,periodic,ar,loss
import numpy as np,pandas as pd,json,time
from threadpoolctl import threadpool_limits
def main():
    rules=json.loads((H/'rules.json').read_text());rows=[];records=[]
    for seed in [3,4,5]:
      for arm in ARMS:
        raw=np.load(R/f'research_generator/results/obs_{arm}_s{seed}.npz')['obs'][:,0]
        for start in [8192,16384]:
            prefix=raw[start:start+4096];sd=prefix.std();x=(prefix-prefix.mean())/sd;y=(raw[start+4096:start+4608]-prefix.mean())/sd
            row=dict(seed=seed,arm=arm,start=start,**features(x))
            for name,fn in [('periodic',periodic),('ar',ar)]:
                row[name]=loss(fn(x,512),y);row[name+'_holdout']=loss(fn(x[:-512],512),x[-512:])
            records.append(row)
            choose={key:('periodic' if ((row[key]<=r['threshold']) if r['sign']==1 else (row[key]>r['threshold'])) else 'ar') for key,r in rules.items()}
            choose.update(fixed_ar='ar',fixed_periodic='periodic',holdout='periodic' if row['periodic_holdout']<row['ar_holdout'] else 'ar',oracle='periodic' if row['periodic']<row['ar'] else 'ar')
            for method,model in choose.items():rows.append(dict(seed=seed,arm=arm,start=start,method=method,model=model,error=row[model]))
        print(seed,arm,flush=True)
    pd.DataFrame(records).to_csv(H/'transfer_features.csv',index=False)
    d=pd.DataFrame(rows);d.to_csv(H/'transfer_decisions.csv',index=False)
    d.groupby('method').error.agg(['mean','median']).to_csv(H/'transfer_summary.csv');print(d.groupby('method').error.mean())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()

"""Declared secondary channel and descriptive observer controls, without refitting."""
import argparse,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import H,measure,change
from actdim.estimator.surrogates import iaaft

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=type(H),default=H/'pilot_seed0');a=p.parse_args()
    rows=[];controls=[];ratios={}
    measure(np.sin(np.arange(512)/11)+np.cos(np.arange(512)/7),512)
    with threadpool_limits(limits=1):
        for arm in ['base','regularized']:
            log=pd.read_csv(a.root/arm/'logs.csv')
            for end in range(512,len(log)+1,128):
                x=log.train_nll.to_numpy()[end-512:end]
                rows.append(dict(arm=arm,end=end,**measure(x,512)))
            for end in [1024,3072]:
                x=log.probe_nll.to_numpy()[end-512:end];m=measure(x,512)['MG'];scaled=measure(10*x,512)['MG']
                assert np.isclose(m,scaled,rtol=1e-7)
                sur=[measure(iaaft(x,rng=np.random.default_rng(s),match=False),512)['MG'] for s in [8,9,10]]
                controls.append(dict(arm=arm,end=end,MG=m,scaled=scaled,surrogate_median=float(np.median(sur)),ratio=float(m/np.median(sur))))
    df=pd.DataFrame(rows);df.to_csv(a.root/'minibatch_windows.csv',index=False)
    pd.DataFrame(controls).to_csv(a.root/'observer_controls.csv',index=False)
    for arm in ['base','regularized']:ratios[arm]=change(df[df.arm==arm],'MG')
    ratios['paired']=ratios['regularized']['ratio']/ratios['base']['ratio']
    (a.root/'secondary_summary.json').write_text(json.dumps(ratios,indent=2))
    print(a.root.name,'exploratory minibatch MG paired ratio',ratios['paired'])

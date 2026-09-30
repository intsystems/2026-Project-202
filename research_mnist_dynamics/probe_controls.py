import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import H,mg,cheap,iaaft

if __name__=='__main__':
    root=H/'confirmation';rows=[];controls=[]
    with threadpool_limits(limits=1):
        for d in sorted(root.glob('seed_*/*')):
            if not (d/'probe.csv').exists():continue
            x=pd.read_csv(d/'probe.csv').test_probe.to_numpy();tag=str(d.relative_to(root)).replace('\\','/')
            for w,tau in [(256,1),(1024,1),(512,4)]:
                for end in list(range(1024,2049,256))+list(range(3072,4097,256)):
                    rows.append(dict(run=tag,window=w,tau=tau,end=end,**mg(x[end-w:end],tau)))
            for end in [2048,4096]:
                seg=x[end-512:end];a=mg(seg)['MG'];scaled=mg(10*seg)['MG']
                assert np.isclose(a,scaled,rtol=1e-7)
                vals=[mg(iaaft(seg,rng=np.random.default_rng(s),match=False))['MG'] for s in [8,9,10]]
                controls.append(dict(run=tag,end=end,MG=a,scale=scaled,smooth=mg(np.convolve(seg,np.ones(8)/8,mode='valid'))['MG'],
                    surrogate=np.median(vals),surrogate_ratio=a/np.median(vals)))
            print('Fixed-probe controls',tag,flush=True)
    pd.DataFrame(rows).to_csv(root/'probe_sensitivity.csv',index=False)
    pd.DataFrame(controls).to_csv(root/'probe_controls.csv',index=False)

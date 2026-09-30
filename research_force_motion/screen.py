"""Independent chaos-conditioned confirmation; no MG values used by screening."""
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run import HERE,init,rollout,top_lyap,train

if __name__=='__main__':
    out=HERE/'results_chaotic';out.mkdir(exist_ok=True)
    rows=[];accepted=[]
    with threadpool_limits(limits=1):
        for seed in range(6,31):
            j,u,x=init(seed,256,1.5)
            states,z,t,_=rollout(j,np.zeros((256,2)),x,0,length=16384,burn=2000)
            whole,_=top_lyap(j,states)
            tail,_=top_lyap(j,states[8192:])
            ok=whole>.01 and tail>.01
            rows.append(dict(seed=seed,lambda_whole=whole,lambda_tail=tail,accepted=ok))
            pd.DataFrame(rows).to_csv(out/'screening.csv',index=False)
            print('Screen',rows[-1],flush=True)
            if ok:accepted.append(seed)
            if len(accepted)==5:break
        (out/'selection.json').write_text(json.dumps(dict(seeds=accepted,criterion='whole and tail top exponent > .01; first five, seeds 6..30',MG_used=False),indent=2))
        for seed in accepted:train(seed,256,out,[0,2000,8000,20000,40000],1.5)

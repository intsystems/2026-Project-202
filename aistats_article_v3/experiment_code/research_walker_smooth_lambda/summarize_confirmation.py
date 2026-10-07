from pathlib import Path
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
COEFS=[0,0.25,1,4]
SEEDS=[231,232,233,234,235]

def lab(seed,coef): return H/f'seed{seed}_lambda{coef:g}'

def main():
    rows=[];traces=[]
    for seed in SEEDS:
        dfs={c:pd.read_csv(lab(seed,c)/'test.csv').set_index('reset') for c in COEFS}
        base=dfs[0]
        for c in COEFS:
            d=dfs[c]; common=base.eligible & d.eligible
            def ratio(col):
                x=d.loc[common,col]/base.loc[common,col]
                return float(x.median()) if len(x) else np.nan
            row=dict(seed=seed,coef=c,n=int(common.sum()),
                healthy=int((d.complete&(d.mean_speed>=.5)).sum()),
                reward_ratio=float(d.padded_reward.mean()/base.padded_reward.mean()),
                J1_ratio=ratio('J1'),J2_ratio=ratio('J2'),R_ratio=ratio('recurrence'),D_ratio=ratio('section_dispersion'))
            rows.append(row)
            for reset in common.index[common]:
                traces += [dict(seed=seed,coef=c,reset=int(reset),metric=k,ratio=float(d.loc[reset,k]/base.loc[reset,k])) for k in ['J1','J2','recurrence','section_dispersion']]
    out=pd.DataFrame(rows);out.to_csv(H/'confirmation_summary.csv',index=False);pd.DataFrame(traces).to_csv(H/'confirmation_traces.csv',index=False)
    agg=out.groupby('coef').agg(seeds=('seed','count'),median_n=('n','median'),healthy=('healthy','median'),reward_ratio=('reward_ratio','median'),J1_ratio=('J1_ratio','median'),J2_ratio=('J2_ratio','median'),R_ratio=('R_ratio','median'),D_ratio=('D_ratio','median')).reset_index()
    agg.to_csv(H/'confirmation_aggregate.csv',index=False)
    print(agg.to_string(index=False))

if __name__=='__main__': main()

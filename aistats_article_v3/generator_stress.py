"""Post-hoc phase-structure stress test on stored, previously trained generators."""
from pathlib import Path
import sys,time,json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks,periodogram,detrend
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from extra_baselines import permutation_entropy,sample_entropy
sys.path.insert(0,str(H.parent/'code'))
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.mle import estimate
CFG=EstimatorConfig(max_E=20,tau='acorr',k_neighbors=20,theiler='autocorr',theiler_cap=320)
OUT=H/'new_results';OUT.mkdir(exist_ok=True)

def harmonic_groups(x):
    # A fixed spectral baseline. It groups peaks, not all integer combinations.
    f,p=periodogram(x,window='hann',detrend='linear')
    peaks,_=find_peaks(p,height=.05*p.max(),distance=2)
    frequencies=sorted(f[peaks]);bases=[];resolution=1/len(x)
    for frequency in frequencies:
        belongs=any(any(abs(frequency-k*b)<=2*resolution for k in range(1,9)) for b in bases)
        if not belongs:bases.append(frequency)
    return len(bases),len(peaks)

def main():
    p=OUT/'generator_stress.csv'
    rows=pd.read_csv(p).to_dict('records') if p.exists() else []
    done={(r['arm'],int(r['seed']),int(r['neuron']),int(r['window']),float(r['noise'])) for r in rows}
    for seed in range(1,6):
        for arm in ['H4','M4','T4']:
            d=np.load(H.parent/f'research_generator/results/obs_{arm}_s{seed}.npz')
            for neuron in range(3):
                for window in [4096,8192]:
                    for noise in [0.,.01,.05]:
                        key=(arm,seed,neuron,window,noise)
                        if key in done:continue
                        x=d['obs'][:window,neuron].astype(float)
                        rng=np.random.default_rng(seed*100000+neuron*1000+window+int(noise*10000))
                        x=x+noise*np.std(x)*rng.normal(size=len(x))
                        t=time.perf_counter();est=estimate(x,CFG);tmg=time.perf_counter()-t
                        t=time.perf_counter();groups,peaks=harmonic_groups(x);th=time.perf_counter()-t
                        row=dict(arm=arm,seed=seed,neuron=neuron,window=window,noise=noise,
                          phases={'H4':1,'M4':2,'T4':4}[arm],MG=est.MG,degenerate=est.degenerate,
                          tau=est.tau,MG_seconds=tmg,harmonic_groups=groups,peaks=peaks,harmonic_seconds=th,
                          permutation_entropy=permutation_entropy(x),sample_entropy=sample_entropy(x),
                          normalized_increments=float(np.mean(np.abs(np.diff(x)))/np.std(x)))
                        rows.append(row);pd.DataFrame(rows).to_csv(p,index=False)
            print('generator stress',arm,seed,'rows',len(rows),flush=True)
    d=pd.DataFrame(rows);summ=[]
    for (neuron,window,noise),g in d.groupby(['neuron','window','noise']):
        for stat in ['MG','harmonic_groups','peaks','permutation_entropy','sample_entropy','normalized_increments']:
            piv=g.pivot(index='seed',columns='arm',values=stat)
            summ.append(dict(neuron=neuron,window=window,noise=noise,stat=stat,
                ordered_seeds=int(((piv.H4<piv.M4)&(piv.M4<piv.T4)).sum()),
                rho=float(spearmanr(g.phases,g[stat]).statistic),
                H4=float(piv.H4.median()),M4=float(piv.M4.median()),T4=float(piv.T4.median())))
    pd.DataFrame(summ).to_csv(OUT/'generator_stress_summary.csv',index=False)
    print(pd.DataFrame(summ).query('neuron==0 and window==8192 and noise==0').to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()

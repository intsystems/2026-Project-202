"""Audit causal replay, labels, raw runs and reference recomputation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from run import TextVAE,load_data,reference
from train import beta_at
from monitor import reference_events,policy,score,METHODS,BUDGETS,periodic

def temporal_tests():
    # A premature crossing cannot receive credit; a repeated check cannot hit twice.
    step=np.arange(0,7169,64);mi=np.full(len(step),5.);response=np.ones(len(step))
    low=((step>=1152)&(step<2048))|(step>=6784)
    mi[low]=1.;response[low]=.1
    truth=reference_events(pd.DataFrame(dict(step=step,MI=mi,shuffle_symkl=response)))
    assert [e['step'] for e in truth['events']]==[1216,2112,6848]
    assert [e['scored'] for e in truth['events']]==[True,True,False]
    result=score([1152,1216,1280,2048,2176,6912],truth)
    assert result['hits']==2 and result['censored_hits']==1
    assert result['records'][0]['event_id'] is None and result['records'][2]['event_id'] is None
    assert result['delays']==[0,64]
    for b in [0,6,12,24,96]:
        p=periodic(b);assert len(p)==len(set(p))==b
        assert all(1024<t<=7168 and t%64==0 for t in p)
    return True

def main():
    temporal_tests();selected=json.loads((H/'selection.json').read_text());selhash=hashlib.sha256((H/'selection.json').read_bytes()).hexdigest()
    assert selected['protocol_sha256']==hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()
    for fname,key in [('features.csv','feature_sha256'),('reference.csv','reference_sha256')]:
        assert selected[key]==hashlib.sha256((H/'seed100'/fname).read_bytes()).hexdigest()
    _,val,vocab=load_data();eps=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
    seen=set();results=[]
    for seed in range(100,110):
        out=H/f'seed{seed}';meta=json.loads((out/'meta.json').read_text())
        assert meta['initial_sha256'] not in seen;seen.add(meta['initial_sha256'])
        assert meta['seed']==seed and meta['steps']==7168 and meta['parameters']==276016
        assert meta['protocol_sha256']==selected['protocol_sha256']
        log=pd.read_csv(out/'logs.csv');ref=pd.read_csv(out/'reference.csv');feat=pd.read_csv(out/'features.csv')
        assert log.step.tolist()==list(range(7168));assert ref.step.tolist()==list(range(0,7169,64))
        np.testing.assert_allclose(log.beta,[beta_at(t) for t in log.step],rtol=0,atol=0)
        assert np.isfinite(log.select_dtypes('number')).all().all()
        assert np.isfinite(ref.select_dtypes('number')).all().all()
        assert feat.end.tolist()==list(range(512,7169,64))
        for row in feat.itertuples():
            x=log.probe_nll.iloc[row.end-512:row.end].to_numpy()
            np.testing.assert_allclose(row.std,x.std(),rtol=1e-10)
            np.testing.assert_allclose(row.KL,log.train_KL.iloc[row.end-64:row.end].mean(),rtol=1e-10)
            assert row.beta==max(.01,log.beta.iloc[row.end-1])
        evaluation=json.loads((out/'evaluation.json').read_text());truth=reference_events(ref)
        assert evaluation['selection_sha256']==selhash and evaluation['truth']==truth
        for d in evaluation['policies']:
            if d['method']=='periodic':checks=periodic(d['budget'])
            else:
                threshold=selected['settings'][str(d['budget'])][d['method']]['threshold']
                assert d['threshold']==threshold
                checks,decisions=policy(feat,d['method'],threshold,d['budget']);assert decisions==d['decisions']
                assert len(checks)<=d['budget'] and all(np.diff([1024]+checks)>=128)
                for end in [1536,3072,4096,6144]:
                    prefix,_=policy(feat[feat.end<=end],d['method'],threshold,d['budget'])
                    assert prefix==[t for t in checks if t<=end],f'Lookahead: {seed} {d["method"]}'
            assert checks==d['checks'];assert score(checks,truth)==d['score']
        net=TextVAE(vocab);net.load_state_dict(torch.load(out/'final.pt',weights_only=True))
        check=reference(net,val,eps)
        for k,v in check.items():
            if not k.endswith('_seconds'):np.testing.assert_allclose(v,ref.iloc[-1][k],rtol=1e-5,atol=1e-6)
        results.append(dict(seed=seed,passed=True,events=len(truth['events']),initial_suitable=truth['suitable']))
        print('AUDIT',seed,'passed',flush=True)
    (H/'audit.json').write_text(json.dumps(dict(temporal_tests=True,all_passed=True,seeds=results),indent=2))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):main()

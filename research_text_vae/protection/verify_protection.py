"""Audit saved continuations and recompute reference diagnostics from final weights."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from run import TextVAE,load_data,reference

def source(seed):
    return H.parent/('pilot_seed0' if seed==0 else f'confirmation_seed{seed}')

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)

def main():
    selected=json.loads((H/'selection.json').read_text())
    assert selected['reference_passed'] and selected['selection_did_not_use_MG']
    mode=selected['mode'];assert selected['confirmation_seeds']==list(range(1,10))
    cohort=pd.read_csv(H/'all_seeds.csv');assert cohort.seed.tolist()==list(range(10))
    _,val,vocab=load_data()
    eps=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
    audits=[]
    for method,seed in [('encoder5',0)]+[(mode,s) for s in range(10)]:
        out=H/method/f'seed{seed}';src=source(seed)
        meta=json.loads((out/'meta.json').read_text())
        assert meta['source_checkpoint_sha256']==hashlib.sha256((src/'branch.pt').read_bytes()).hexdigest()
        assert meta['outer_stream_matches_original']
        assert meta['mode']==method and meta['seed']==seed
        log=pd.read_csv(out/'logs.csv');ref=pd.read_csv(out/'reference.csv')
        assert log.step.tolist()==list(range(3072))
        assert ref.step.tolist()==list(range(0,3073,128))
        assert np.isfinite(log[['probe_nll','train_nll','train_KL']]).all().all()
        assert np.isfinite(ref.select_dtypes('number')).all().all()
        original=pd.read_csv(src/'regularized/logs.csv')
        close(log.loc[:1024,'probe_nll'],original.loc[:1024,'probe_nll'])
        assert (log.loc[1024:,'beta']==1).all()
        if method=='encoder5':
            assert meta['extra_encoder_updates']==10240 and meta['frozen_decoder_update_audited']
        else:assert meta['extra_encoder_updates']==0
        net=TextVAE(vocab);net.load_state_dict(torch.load(out/'final.pt',weights_only=True))
        final=reference(net,val,eps)
        for key,value in final.items():
            if not key.endswith('_seconds'):close(value,ref.iloc[-1][key])
        before=pd.read_csv(src/'base/reference.csv').query('512<=step<=1024').median(numeric_only=True)
        reg=pd.read_csv(src/'regularized/reference.csv').query('2048<=step<=3072').median(numeric_only=True)
        late=ref.query('2048<=step<=3072').median(numeric_only=True)
        valid=all(late[k]>=.5*before[k] and late[k]>=2*reg[k] for k in ['MI','shuffle_symkl'])
        saved=json.loads((out/'retention.json').read_text())
        assert bool(valid)==saved['protection_valid']
        for key in ['MI','shuffle_symkl']:
            for label,val1 in [('before',before[key]),('regularized',reg[key]),('protected',late[key])]:
                close(saved['reference'][key][label],val1)
        if method==mode:
            comparison=json.loads((out/'comparison.json').read_text())
            windows=pd.read_csv(out/'three_arm_windows.csv')
            assert not windows.duplicated(['arm','window','tau','end']).any()
            assert np.isfinite(windows.MG).all()
            for setting in comparison['scalar']:
                sub=windows[(windows.window==setting['window'])&(windows.tau==setting['tau'])]
                med=sub.query('2048<=end<=3072').groupby('arm').MG.median()
                close(setting['q_protected'],med.protected/med.base)
                close(setting['q_regularized'],med.regularized/med.base)
                close(setting['protected_vs_regularized'],med.protected/med.regularized)
            row=cohort[cohort.seed==seed].iloc[0]
            close(row.R,comparison['scalar'][0]['protected_vs_regularized'])
            assert bool(row.protection_valid)==bool(valid)
        audits.append(dict(seed=seed,mode=method,final_reference_recomputed=True,retention_valid=bool(valid),passed=True))
        print(f'PASS {method} seed{seed}: checkpoint, shared prefix, final reference, retention and ratios',flush=True)
    result=dict(all_passed=True,branches=audits)
    (H/'audit.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):main()

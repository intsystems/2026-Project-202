"""Consistency audit of scientific claims against saved raw tables and spectra."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
for name in ['results','results_chaotic']:
    r=H/name;m=pd.read_csv(r/'mg.csv');ref=pd.read_csv(r/'independent_reference.csv')
    assert len(m)==150 and len(ref)==15
    assert m.groupby(['seed','training_step','neuron']).size().eq(2).all()
    assert np.isfinite(m[['MG','MG40','ident']]).all().all()
    assert not m.degenerate.any()
    for row in ref.itertuples():
        v=np.load(r/f'seed_{row.seed}'/f'lyapunov_{row.training_step:05d}.npz')['spectrum']
        assert v.shape==(256,) and np.isfinite(v).all()
        np.testing.assert_allclose(v[:3],[row.lambda1,row.lambda2,row.lambda3],atol=1e-14)
    audit=json.loads((r/'control_audit.json').read_text())
    assert audit['maximum_difference']==0
    assert pd.read_csv(r/'scale_control.csv').delta.abs().max()<1e-12
r=H/'results_chaotic'
s=pd.read_csv(r/'screening.csv');assert len(s)==13 and list(s[s.accepted].seed)==[7,9,15,16,18]
m=pd.read_csv(r/'mg_summary.csv')
b=m[m.training_step==0].set_index(['seed','neuron']);a=m[m.training_step==40000].set_index(['seed','neuron'])
assert (a.MG<b.MG).sum()==15
s=pd.read_csv(r/'sensitivity.csv')
b=s[s.training_step==0].set_index(['seed','window','tau']);a=s[s.training_step==40000].set_index(['seed','window','tau'])
assert (a.MG<b.MG).sum()==20
ref=pd.read_csv(r/'independent_reference.csv');assert ref[ref.training_step==40000].endpoint_pass.sum()==4
p=pd.read_csv(r/'perturbations.csv');assert (p.aligned_nrmse<.1).sum()==13
assert ((p.aligned_nrmse<.1)&(p.return_error<.1)).sum()==12
t=pd.read_csv(r/'benchmark.csv');assert len(t)==6 and (t.drop(columns=['training_step','repeat'])>0).all().all()
print('Scientific-result audit passed: raw spectra, 300 windows, all sensitivities, failures and controls included.')

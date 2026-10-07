"""Verify wrapper semantics on recurrence, scale changes and unusable observations."""
import json
import numpy as np
from mg_pipeline import H,measure,estimate,EstimatorConfig

def main():
    x=np.sin(np.arange(1024)*np.sqrt(2)/10)
    settings=dict(window=512,stride=256,E=10,tau=2,k=10,theiler=20,min_std=1e-12)
    rows=list(measure(x,**settings))
    direct=estimate(x[:512],EstimatorConfig(max_E=10,tau=2,k_neighbors=10,theiler=20,theiler_cap=20))
    assert rows[0]['MG']==direct.MG
    assert rows[0]['n_points']==512-9*2 and [r['end'] for r in rows]==[512,768,1024]
    rescaled=list(measure(3*x+7,**settings))
    assert np.allclose([r['MG'] for r in rows],[r['MG'] for r in rescaled],rtol=1e-6)
    assert all(r['status']=='change_statistic' for r in rows)
    for bad in [np.ones(1024),np.full(1024,np.nan),1e-14*x]:
        assert all(r['status']=='unusable' and np.isnan(r['MG']) for r in measure(bad,**settings))
    assert all(r['status']=='unusable' for r in measure(x,**dict(settings,theiler=300)))
    (H/'pipeline_validation.json').write_text(json.dumps(dict(kernel_agreement=True,causal_window_ends=True,
        valid_vector_number=True,affine_rescaling=True,flat_nonfinite_noise_floor_gates=True,
        insufficient_neighbors=True,automatic_dimension_certification=False),indent=2))
    print('Measurement wrapper checks passed.')
if __name__=='__main__':main()

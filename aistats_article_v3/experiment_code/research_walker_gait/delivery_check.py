import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from pypdf import PdfReader
H=Path(__file__).resolve().parent

out=H/'seed200'
assert len(list(out.glob('step*/policy.zip')))==33
assert len(list(out.glob('step*/reset*/metrics.json')))==99
for f,sha in json.loads((H/'estimator_source_hashes.json').read_text()).items():
    assert hashlib.sha256((H.parent/f).read_bytes()).hexdigest()==sha
root=out/'step0655360/reset31001'
p=json.loads((root/'perturb_4.json').read_text());r=pd.read_csv(root/'perturb_4.csv')
d=np.load(root/'perturb_4_curves.npz')['distances'];d=d.reshape(-1,d.shape[-1])
assert len(r)==p['probes']==136
np.testing.assert_allclose(r.amplification.median(),p['median_amplification'])
for i,row in enumerate(r.itertuples()):
    if not row.fell and not row.nearly_tangent:
        np.testing.assert_allclose(np.median(d[i,-p['period']:])/d[i,0],row.amplification,rtol=1e-10)
reader=PdfReader(H/'report_ru.pdf');assert len(reader.pages)==2
text='\n'.join(x.extract_text() for x in reader.pages)
assert 'MG пока не оценивался' in text and '4 194 304' in text
(H/'delivery_checks.json').write_text(json.dumps(dict(checkpoints=33,evaluations=99,estimator_unchanged=True,perturbation_rows_verified=136,pdf_pages=2),indent=2))
print('PASS: 33 checkpoints, 99 evaluations, 136 perturbations, unchanged estimator, two-page PDF.')

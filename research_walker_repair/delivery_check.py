from pathlib import Path
import json
import pandas as pd
from pypdf import PdfReader
H=Path(__file__).resolve().parent
s=json.loads((H/'summary.json').read_text());a=json.loads((H/'audit.json').read_text())
assert a['passed'] and len(a['seeds'])==6
assert s['gate_passes']==4 and s['heldout_walks']==47 and s['heldout_total']==50
assert s['reference_events']==0 and s['MG_decreases_total']==0
assert s['old_final_control']['walking']==1
assert len(list(H.glob('seed21*/step*/policy.zip')))==54
assert len(list(H.glob('seed21*/step*/reset*/metrics.json')))==390
assert len(list(H.glob('seed21*/step*/reset*/MG_windows.csv')))==117
sens=pd.read_csv(H/'sensitivity.csv');p=sens[(~sens.pilot)&(sens.sensor=='right_knee')&(sens.window==1024)&(sens.tau==8)]
assert len(p)==5 and ((p.ratio<1)&(p.valid_pairs>=8)).sum()==3
pdf=PdfReader(H/'report_ru.pdf');assert len(pdf.pages)==3
text='\n'.join(p.extract_text() for p in pdf.pages)
assert '47/50' in text and '1/10' in text and '1,8 с' in text
result=dict(passed=True,pdf_pages=3,training_runs=6,checkpoints=54,evaluations=390,MG_traces=117,
    heldout_walks=47,confirmation_gate_passes=4,independent_simplification_events=0,short_window_sign_changes=3)
(H/'delivery_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

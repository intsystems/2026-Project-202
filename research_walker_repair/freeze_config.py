import hashlib,json,math
import numpy as np
import pandas as pd
from motion import H

def select(label):
    out=H/label;gate=json.loads((out/'gate.json').read_text());assert gate['passed']
    assert not list(H.glob('seed*/step*/reset*/MG_windows.csv'))
    frame=pd.read_csv(out/'validation.csv');base=frame[(frame.step==0)&frame.eligible];late=frame[(frame.step==1048576)&frame.eligible]
    common=sorted(set(base.reset)&set(late.reset));assert len(common)>=2,'Walking repaired but selected scalar unavailable'
    periods=list(base[base.reset.isin(common)].period)+list(late[late.reset.isin(common)].period)
    tau=int(np.clip(math.ceil(np.median(periods)/19),1,12))
    result=dict(pilot_label=label,horizon=1048576,confirmation_seeds=list(range(211,216)),
        conservative=bool(json.loads((out/'train.json').read_text())['conservative']),tau=tau,anchors=[0,1024],
        pilot_validation_periods=periods,selected_without_MG=True,
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (H/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys;select(sys.argv[1] if len(sys.argv)>1 else 'seed210')

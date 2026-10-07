from pathlib import Path
import hashlib,json,math
import numpy as np
from motion import H

if __name__=='__main__':
    pair=json.loads((H/'seed200/pair.json').read_text());assert pair['usable'],'No usable pilot pair: report feasibility failure.'
    assert not any((H/f'seed{s}').exists() for s in range(201,206))
    assert not list((H/'seed200').glob('step*/reset*/MG_windows.csv'))
    first=H/'seed200'/f"step{pair['early']:07d}"/f"reset{pair['common_resets'][0]}"
    probe=json.loads((first/'perturb_4.json').read_text())
    anchors=[0,1024] if probe['cpu_seconds']>60 else [0,512,1024,1536]
    periods=[]
    for step in [pair['early'],pair['late']]:
        for reset in pair['common_resets']:
            metrics=json.loads((H/'seed200'/f'step{step:07d}'/f'reset{reset}'/'metrics.json').read_text())
            periods.append(metrics['period'])
    tau=int(np.clip(math.ceil(float(np.median(periods))/19),1,12))
    selection=dict(pilot=200,horizon=pair['horizon'],confirmation_seeds=list(range(201,206)),
        pilot_pair=pair,pilot_periods=periods,tau=tau,anchors=anchors,resource_pilot_seconds=probe['seconds'],resource_pilot_cpu_seconds=probe['cpu_seconds'],
        selected_without_MG=True,confirmation_workers=5,
        resource_amendment_sha256=hashlib.sha256((H/'RESOURCE_AMENDMENT.md').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
    (H/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))

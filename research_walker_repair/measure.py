import argparse,json,shutil
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits
from motion import H,Actor
from perturb import probe
from features import analyze

def run(label):
    selection=json.loads((H/'selection.json').read_text());pair=json.loads((H/label/'pair.json').read_text())
    if pair['common_resets']:
        reset=pair['common_resets'][0]
        for step in [0,1048576]:
            checkpoint=H/label/f'step{step:07d}';root=checkpoint/f'reset{reset}'
            shared=H/'shared_anchor_probes'/f'reset{reset}'
            if step==0:
                reference=Actor(H/'anchor').model.policy.state_dict();current=Actor(checkpoint).model.policy.state_dict()
                for k in reference:torch.testing.assert_close(current[k],reference[k],rtol=0,atol=0)
                shared.parent.mkdir(exist_ok=True)
                with FileLock(str(shared)+'.lock',timeout=600):
                    if (shared/'complete.json').exists():
                        for p in shared.glob('perturb_2*'):shutil.copy2(p,root/p.name)
                    else:
                        probe(checkpoint,reset,selection['anchors']);shared.mkdir(parents=True,exist_ok=True)
                        for p in root.glob('perturb_2*'):shutil.copy2(p,shared/p.name)
                        (shared/'complete.json').write_text(json.dumps(dict(source=label,reset=reset)))
            else:probe(checkpoint,reset,selection['anchors'])
    analyze(label[4:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(a.label)

"""MNIST download and independent evaluator training, never using generated data."""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import torch
from torchvision.datasets import MNIST
from threadpoolctl import threadpool_limits
from model import Classifier,configure

H=Path(__file__).resolve().parent

def data():
    a=MNIST(H/'data',train=True,download=True);b=MNIST(H/'data',train=False,download=True)
    return a,b

@torch.no_grad()
def evaluate(c,x,y):
    c.eval();ps=[]
    for v in x.split(512):ps.append(c(v).softmax(1))
    p=torch.cat(ps);conf,label=p.max(1);ok=conf>=.9
    return dict(accuracy=float((label==y).float().mean()),acceptance=float(ok.float().mean()),
        accepted_accuracy=float((label[ok]==y[ok]).float().mean()))

if __name__=='__main__':
    configure();a,b=data();x=a.data[:,None].float()/127.5-1;y=a.targets
    tx=b.data[:,None].float()/127.5-1;ty=b.targets
    torch.manual_seed(20260929);c=Classifier();opt=torch.optim.Adam(c.parameters(),lr=.001)
    rows=[];t=time.perf_counter()
    with threadpool_limits(limits=1):
        for epoch in range(5):
            c.train();ids=torch.randperm(len(x))
            for ix in ids.split(128):
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(c(x[ix]),y[ix]).backward();opt.step()
            row=dict(epoch=epoch+1,**evaluate(c,tx,ty));rows.append(row);print(row,flush=True)
        torch.save(c.state_dict(),H/'classifier.pt')
        out=dict(test=rows[-1],epochs=rows,seconds=time.perf_counter()-t,
            training_seed=20260929,training_samples=len(x),test_samples=len(tx),
            raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'data/MNIST/raw').glob('*') if p.is_file()})
        (H/'classifier_metrics.json').write_text(json.dumps(out,indent=2))

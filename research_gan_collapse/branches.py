"""Exploratory matched branches from the same extended MNIST checkpoint."""
from pathlib import Path
import subprocess
import sys
H=Path(__file__).resolve().parent
for arm in ['base','g_fast','d_slow','frozen']:
    subprocess.run([sys.executable,str(H/'run.py'),'--channels','1','--seed','0','--arm',arm,
        '--steps','16384','--switch','12288','--every','256',
        '--resume',str(H/'mnist_pilot/base_long/checkpoint_12288.pt'),
        '--out',str(H/'mnist_branches'/arm)],check=True)

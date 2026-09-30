from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
for seed in [1,2,3,4,5]:
    subprocess.run([sys.executable,str(H/'run.py'),'--seed',str(seed),'--lr','.1',
        '--batch','64','--momentum','.9','--out',str(H/'confirmation'/f'seed_{seed}')],check=True)
subprocess.run([sys.executable,str(H/'reference.py'),'--root',str(H/'confirmation')],check=True)
subprocess.run([sys.executable,str(H/'probe.py'),'--root',str(H/'confirmation')],check=True)
subprocess.run([sys.executable,str(H/'analyze.py'),'--root',str(H/'confirmation')],check=True)

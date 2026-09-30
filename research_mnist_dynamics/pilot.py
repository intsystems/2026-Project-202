from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
for lr,batch,mom,name in [(1,0,0,'full1'),(3,0,0,'full3'),(.1,64,.9,'sgd')]:
    subprocess.run([sys.executable,str(H/'run.py'),'--lr',str(lr),'--batch',str(batch),
        '--momentum',str(mom),'--out',str(H/'pilot'/name)],check=True)
subprocess.run([sys.executable,str(H/'reference.py'),'--root',str(H/'pilot')],check=True)

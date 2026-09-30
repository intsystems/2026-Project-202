from pathlib import Path
import shutil

R=Path(__file__).resolve().parent
H=R/'research_walker_smooth_lambda'
S=R/'research_walker_smooth'
H.mkdir(exist_ok=True)
for name in ['train.py','smooth_ppo.py','motion.py','evaluate.py','diagnostic.py','requirements.txt','build_report.py']:
    shutil.copy2(S/name,H/name)
shutil.copytree(R/'research_walker_repair/anchor',H/'anchor',dirs_exist_ok=True)
print(H)

from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
ARCHIVE_EXPERIMENT = CODE.parent / "archive" / 'research_mg_deployment'


if __name__ == "__main__":
    if not (ARCHIVE_EXPERIMENT / 'checkpoint_selection.py').is_file():
        raise SystemExit("The local archive is required for this historical collection pipeline.")
    env = os.environ.copy()
    earlier = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(CODE)] + ([earlier] if earlier else []))
    subprocess.run([sys.executable, 'checkpoint_selection.py', *sys.argv[1:]],
                   cwd=ARCHIVE_EXPERIMENT, env=env, check=True)

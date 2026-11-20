from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]


def main():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(CODE)] + ([existing] if existing else []))
    for command in [['run.py', '--n', '256', '--gain', '1.5', '--seeds', '7', '--out', 'generated']]:
        subprocess.run([sys.executable, *command, *sys.argv[1:]], cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()

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
    for command in [['run.py', '--seeds', '10', '11', '12', '13', '14', '15', '16', '17']]:
        subprocess.run([sys.executable, *command, *sys.argv[1:]], cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()

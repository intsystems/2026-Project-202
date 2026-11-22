"""Collect baseline, regularized and free-bits logs for one Penn Treebank seed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    seed = str(args.seed)
    output = HERE / "generated" / f"seed{seed}"
    if (output / "freebits" / "meta.json").is_file():
        parser.error(f"Complete run already exists: {output}")
    env = os.environ.copy()
    earlier = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(CODE)] + ([earlier] if earlier else []))
    subprocess.run([sys.executable, "run.py", "--seed", seed, "--out", str(output)],
                   cwd=HERE, env=env, check=True)
    subprocess.run([sys.executable, "run_protection.py", "--seed", seed, "--mode", "freebits"],
                   cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()

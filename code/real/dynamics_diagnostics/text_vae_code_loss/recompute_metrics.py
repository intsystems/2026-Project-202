"""Score baseline, suppression and free-bits logs after a fresh collection."""
import argparse
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from analyze import analyze as analyze_base
from analyze_protection import analyze as analyze_protected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        analyze_base(HERE / "generated" / f"seed{args.seed}")
        analyze_protected(args.seed, "freebits")


if __name__ == "__main__":
    main()

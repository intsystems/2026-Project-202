"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from experiment_runner.plotting import draw


if __name__ == "__main__":
    draw(HERE, summary=('heldout_summary.csv', 'truth', ('MG', 'LB', 'PRdelay'), 'Reference dimension', 'Median estimate', 'line'))

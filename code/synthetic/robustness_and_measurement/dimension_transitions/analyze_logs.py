"""Rebuild the experiment figures from the included CSV files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from experiment_runner.plotting import draw


if __name__ == "__main__":
    draw(HERE, summary=('transitions_raw.csv', 'truth1', ('level1', 'level0', 'level2'), 'Low reference dimension', 'Median MG by segment', 'line'))

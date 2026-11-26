# Transformer grokking

Track scalar logs and detrended trajectory rank for modular addition and S5 composition, with no-decay controls.

## In the article

- [Appendix E · Grokking](../../../../article/manuscript/ReadMe.md#appendix-e--grokking). The modular arithmetic and permutation tasks, configuration grid, and controls in the grokking section.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Grokking and optimization stability](../ReadMe.md)

- [rank_milestones.csv](logs/rank_milestones.csv). Recorded generalization and memorization steps.
- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.
- [compare_accuracy_rise.py](compare_accuracy_rise.py). Compare median MG@weight norm before, during, and after the rise in validation accuracy on the saved transformer and perceptron logs.
- [recompute_metrics.py](recompute_metrics.py). Recompute the article's metrics after collecting the upstream runs.

- [Training loop](train.py)
- [Trajectory analysis](trajectory_metrics.py)
- [Training configurations](run_configs.py)
- [Trajectory and scalar diagnostics](../analyze_training.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python compare_accuracy_rise.py
python collect_logs.py --device cuda:0
python recompute_metrics.py --device cpu
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

The accuracy-rise comparison writes one row per run and pair of thresholds to `code/runs/redrawn/grokking_accuracy_rise.csv`. An accuracy threshold is reached at the first of five consecutive logged readings at or above it. The comparison uses complete windows within the rise and equally long adjacent intervals, restricted to the available log after training accuracy first reaches 95%. Each interval must contain at least three valid windows. The script also reports runs with too few windows, overlapping training fit, or an incomplete rise.

The article's scalar estimates fail its dimension checks. The four-run trajectory result is based on detrended participation ratio. Fresh trajectory analyses require new training and may differ from the archived article outputs.

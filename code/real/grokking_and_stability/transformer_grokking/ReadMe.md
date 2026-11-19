# Transformer grokking

Track scalar logs and detrended trajectory rank for modular addition and S5 composition, with no-decay controls.

Reference in the publication materials — Article, Section 7 and Appendices F–H/J.

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Grokking and optimization stability](../ReadMe.md)

- [rank_milestones.csv](logs/rank_milestones.csv). Recorded generalization and memorization steps.
- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.
- [recompute_metrics.py](recompute_metrics.py). Recompute the article's metrics after collecting the upstream runs.

- [Training loop](train.py)
- [Trajectory analysis](trajectory_metrics.py)
- [Training configurations](run_configs.py)
- [Trajectory and scalar diagnostics](../analyze_training.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --device cuda:0
python recompute_metrics.py --device cpu
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

The article's scalar estimates fail its dimension checks. The four-run trajectory result is based on detrended participation ratio. Fresh trajectory analyses require new training and may differ from the archived article outputs.

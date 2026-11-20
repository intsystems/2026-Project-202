# Transformer grokking

Track scalar logs and detrended trajectory rank for modular addition and S5 composition, with no-decay controls.

## In the article

- [Section 7.1 · The diagnostics reject both settings](../../../../article/manuscript/ReadMe.md#section-71--the-diagnostics-reject-both-settings). Scalar diagnostics for the transformer logs.
- [Section 7.2 · The trajectory simplifies at generalisation](../../../../article/manuscript/ReadMe.md#section-72--the-trajectory-simplifies-at-generalisation). The effective-rank dip and recovery in Figure 3.
- [Section 7.3 · The log estimate falls where the trajectory does](../../../../article/manuscript/ReadMe.md#section-73--the-log-estimate-falls-where-the-trajectory-does). The aligned scalar-log estimate in Figure 3.
- [Appendix G.2 · The effective-rank collapse, run by run](../../../../article/manuscript/ReadMe.md#appendix-g2--the-effective-rank-collapse-run-by-run). Timing and depth of the dip in Figure 10 and Table 16.
- [Appendix H.2 · Declines of the estimate that are not generalisation](../../../../article/manuscript/ReadMe.md#appendix-h2--declines-of-the-estimate-that-are-not-generalisation). Long-window estimates that decline without generalisation.
- [Appendix H.3 · Resolution of a windowed estimate on a training log](../../../../article/manuscript/ReadMe.md#appendix-h3--resolution-of-a-windowed-estimate-on-a-training-log). Resolution and surrogate controls in Table 18 and Figures 11–12.

## Files and commands

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

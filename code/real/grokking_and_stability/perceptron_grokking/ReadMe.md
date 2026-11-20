# Full-batch perceptron grokking

Compare modular arithmetic and polynomial targets, including label-matched controls and the window-length sweep.

## In the article

- [Section 7.1 · The diagnostics reject both settings](../../../../article/manuscript/ReadMe.md#section-71--the-diagnostics-reject-both-settings). Scalar diagnostics for the full-batch setting.
- [Appendix F · Diagnostics and figures for the grokking application](../../../../article/manuscript/ReadMe.md#appendix-f--diagnostics-and-figures-for-the-grokking-application). Diagnostic map and label-matched tasks in Figures 8–9.
- [Appendix H.1 · The full-batch setting](../../../../article/manuscript/ReadMe.md#appendix-h1--the-full-batch-setting). The effective-rank window-length comparison in Table 17.
- [Appendix J · The training runs](../../../../article/manuscript/ReadMe.md#appendix-j--the-training-runs). The perceptron run inventory in Table 21.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Grokking and optimization stability](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.
- [recompute_metrics.py](recompute_metrics.py). Recompute the article's metrics after collecting the upstream runs.

- [Training loop](train.py)
- [Training configurations](run_configs.py)
- [Grokking and representation analysis](../analyze_training.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --device cuda:0
python recompute_metrics.py --device cpu
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

The diagnostic map uses both perceptron and transformer estimates. The effective-rank collapse did not reproduce in the full-batch setting.

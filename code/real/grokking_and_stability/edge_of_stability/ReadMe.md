# Edge of stability

Track Hessian sharpness, oscillation and recurrence in full-batch descent on a quadratic perceptron.

## In the article

- [Appendix K · Descent at the edge of stability](../../../../article/manuscript/ReadMe.md#appendix-k--descent-at-the-edge-of-stability). Hessian sharpness, oscillation, and recurrence in Table 22 and Figure 15.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Grokking and optimization stability](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.
- [recompute_metrics.py](recompute_metrics.py). Recompute the article's metrics after collecting the upstream runs.

- [Full-batch training and sharpness](train.py)
- [Recurrence and stability analysis](../analyze_training.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --device cuda:0
python recompute_metrics.py --device cpu
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

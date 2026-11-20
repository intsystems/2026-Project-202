# Text VAE code loss

Compare baseline, code suppression and free bits on Penn Treebank. Plot scalar reconstruction loss, rolling MG and independently measured mutual information.

## On the poster

[Block 06 · Diagnose code loss and cycle formation](../../../../article/poster/ReadMe.md#block-06--diagnose-code-loss-and-cycle-formation) · Text VAE on Penn Treebank. This experiment belongs to the poster.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Diagnostics of model dynamics](../ReadMe.md)

- [run.py](run.py). Original source. Baseline training, free-bits branch or analysis.
- [run_protection.py](run_protection.py). Original source. Baseline training, free-bits branch or analysis.
- [analyze_protection.py](analyze_protection.py). Original source. Baseline training, free-bits branch or analysis.
- [collect_logs.py](collect_logs.py). Collect the paired VAE branches.
- [recompute_metrics.py](recompute_metrics.py). Score the three branches from a fresh run.
- [analyze.py](analyze.py). Analyze the fresh baseline and suppression branches.
- [analyze_logs.py](analyze_logs.py). Rebuild the poster panel from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --seed 1
python recompute_metrics.py --seed 1
```

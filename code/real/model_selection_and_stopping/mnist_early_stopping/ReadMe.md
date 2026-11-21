# Noisy MNIST early stopping

Use rules fixed on development runs to select stopping checkpoints in eight confirmation runs with 40% label noise.

## In the article

- [Appendix H · Comparing intervention rules](../../../../article/manuscript/ReadMe.md#appendix-h--comparing-intervention-rules). Noisy-label MNIST stopping against fixed schedules and scalar rules. The folder covers the 40% noise condition.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [frozen_rules.csv](logs/frozen_rules.csv). Fixed checkpoint rules used for evaluation.
- [run.py](run.py). Original source. Train and record features and classification metrics.
- [control.py](control.py). Original source. Score the frozen checkpoint decisions.
- [collect_logs.py](collect_logs.py). Collect MNIST training and feature logs.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python collect_logs.py --help
python analyze_logs.py
```

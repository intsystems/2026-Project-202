# Digits function subspace

A function-space control on a digits model. The effective excitation did not meet the full protocol.

## In the article

- [Appendix C.1 · Analytic and driven-learning evaluation](../../../../article/manuscript/ReadMe.md#appendix-c1--analytic-and-driven-learning-evaluation). Related parameter-space classifier. The function-space construction is an earlier experiment.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Handwritten digits and estimator calibration](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Function subspace model](system.py)
- [Collection and scoring](../../controlled_systems/run_systems.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

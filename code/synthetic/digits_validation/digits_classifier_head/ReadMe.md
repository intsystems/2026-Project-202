# Digits parameter subspace

A constrained head on a frozen model trained on handwritten digits. The number of controlled forcing phases supplies the reference.

## In the article

- [Section 3.1 · Analytic oscillators and learning](../../../../article/manuscript/ReadMe.md#section-31--analytic-oscillators-and-learning). Held-out ranks for the handwritten digits classifier with frozen features.
- [Appendix C.1 · Analytic and driven-learning evaluation](../../../../article/manuscript/ReadMe.md#appendix-c1--analytic-and-driven-learning-evaluation). Forcing, reference effective rank, and errors by observable.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Handwritten digits and estimator calibration](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Controlled training and observations](system.py)
- [Parameter subspace validation](../run_validation.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

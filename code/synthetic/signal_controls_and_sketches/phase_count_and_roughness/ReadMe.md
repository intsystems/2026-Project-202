# Roughness and phase controls

Match scalar roughness across different phase counts and vary observer roughness at fixed phase count.

## In the article

- [Section 3.2 · Learned generators distinguish phases from harmonics](../../../../article/manuscript/ReadMe.md#section-32--learned-generators-distinguish-phases-from-harmonics). Related control for distinguishing independent phases from spectral peaks. This folder contains the earlier synthetic roughness controls.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Robustness and trajectory measurement](../ReadMe.md)

- [verdict.csv](logs/verdict.csv). Recorded verdict and metrics for the controls.
- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Collection and scoring for this control](../../digits_validation/run_validation.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

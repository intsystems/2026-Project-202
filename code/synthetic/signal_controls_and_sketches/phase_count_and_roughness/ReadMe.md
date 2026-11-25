# Roughness and phase controls

Match scalar roughness across different phase counts and vary observer roughness at fixed phase count.

## In the article

- [Section 3.1 · Independent oscillations and harmonics](../../../../article/manuscript/ReadMe.md#section-31--independent-oscillations-and-harmonics). The direct check uses scalar sums of four tones with matched roughness. The appendix reports the median across the eight paired records in [geometry_levels.csv](logs/geometry_levels.csv).

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

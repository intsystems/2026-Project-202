# Observer and scale controls

Compare estimator shifts under changes in observer gain, amplitude, forcing and coordinate rotation.

## In the article

- [Section 2.3 · Pooled local-dimension estimation](../../../../article/manuscript/ReadMe.md#section-23--pooled-local-dimension-estimation). Related discussion of affine rescaling and sensitivity to the observed signal. This folder contains separate control sweeps.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Robustness and trajectory measurement](../ReadMe.md)

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

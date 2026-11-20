# Estimator calibration

Configuration selection on the development seeds and ranks for the eight- and twenty-direction protocols.

## In the article

- [Section 4 · Validation protocol](../../../../article/manuscript/ReadMe.md#section-4--validation-protocol). Rules for holding out ranks and freezing estimator settings.
- [Appendix B.3 · Frozen configurations](../../../../article/manuscript/ReadMe.md#appendix-b3--frozen-configurations). The selected eight- and twenty-direction configurations in Table 4.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Handwritten digits and estimator calibration](../ReadMe.md)

- [settings.py](settings.py). Fixed calibration settings and selection data.
- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Calibration runs and scoring](calibrate.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

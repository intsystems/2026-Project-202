# Estimator calibration

Configuration selection on the development seeds and ranks for the eight- and twenty-direction protocols.

## In the article

- [Section 3.1 · Analytic oscillators and learning](../../../../article/manuscript/ReadMe.md#section-31--analytic-oscillators-and-learning). Separate seeds and ranks for selecting and evaluating the estimator settings.
- [Appendix C.1 · Analytic and driven-learning evaluation](../../../../article/manuscript/ReadMe.md#appendix-c1--analytic-and-driven-learning-evaluation). The frozen handwritten digits configuration. The wider calibration sweep belongs to the earlier study.

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

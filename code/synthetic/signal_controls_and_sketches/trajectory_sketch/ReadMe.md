# Trajectory sketch validation

Check CountSketch on known ranks, compare logs with and without recording, and measure recording cost.

## In the article

- [Appendix E · Grokking](../../../../article/manuscript/ReadMe.md#appendix-e--grokking). The grokking section describes how the parameter and logit trajectories are projected. This folder contains separate sketch validation runs.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Robustness and trajectory measurement](../ReadMe.md)

- [sketch_cost.csv](logs/sketch_cost.csv). Timing and storage cost.
- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Sketch accuracy, recording and timing experiments](measure_sketch.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

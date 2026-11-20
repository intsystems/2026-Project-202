# Oscillating diagonal matrix

Known independent phases in a diagonal matrix trajectory.

## In the article

- [Section 5 · Evaluation on systems with a known active dimension](../../../../article/manuscript/ReadMe.md#section-5--evaluation-on-systems-with-a-known-active-dimension). The diagonal-matrix example.
- [Appendix C · Per-observer results and alternative statistics](../../../../article/manuscript/ReadMe.md#appendix-c--per-observer-results-and-alternative-statistics). The oscillating matrix row in Table 6.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Controlled systems with known dimension](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Oscillating matrix model](system.py)
- [Collection and scoring](../run_systems.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

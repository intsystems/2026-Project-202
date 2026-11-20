# Delay coordinate reconstruction

Example scalar observations and delay clouds for recurrent, transient and mini-batch cases.

## In the article

- [Section 3.3 · Observation model and estimator](../../../../article/manuscript/ReadMe.md#section-33--observation-model-and-estimator). Figure 1 shows a scalar signal, its delay coordinates, and admissible neighbors.
- [Appendix A · The results behind the estimator](../../../../article/manuscript/ReadMe.md#appendix-a--the-results-behind-the-estimator). Figure 4 compares delay clouds across regimes.
- [Appendix C · Per-observer results and alternative statistics](../../../../article/manuscript/ReadMe.md#appendix-c--per-observer-results-and-alternative-statistics). The traces and window estimates also supply Figure 5.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Delay embedding and estimator limits](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Log and delay cloud generation](collect_curves.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

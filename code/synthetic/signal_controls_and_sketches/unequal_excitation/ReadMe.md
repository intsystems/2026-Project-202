# Unequal excitation

Hold the number of phases fixed and vary their relative amplitudes to compare MG with the participation ratio.

## In the article

- [Section 6.4 · Nuisance factors and nulls](../../../../article/manuscript/ReadMe.md#section-64--nuisance-factors-and-nulls). When weakly excited phases become hard to resolve.
- [Appendix D · Conditions of validity in detail](../../../../article/manuscript/ReadMe.md#appendix-d--conditions-of-validity-in-detail). The amplitude-ratio sweep in Figure 7.

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

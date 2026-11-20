# Delay lag sensitivity

Sweep the delay span relative to the recurrent period while keeping the dynamics fixed.

## In the article

- [Section 6.2 · The delay lag](../../../../article/manuscript/ReadMe.md#section-62--the-delay-lag). Lag expressed as a fraction of the recurrent period.
- [Appendix D · Conditions of validity in detail](../../../../article/manuscript/ReadMe.md#appendix-d--conditions-of-validity-in-detail). The lag sweep in Figure 6.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Delay embedding and estimator limits](../ReadMe.md)

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

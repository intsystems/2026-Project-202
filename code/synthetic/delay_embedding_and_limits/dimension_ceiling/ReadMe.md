# Dimension ceiling

Sweep embedding dimension and record length to measure where MG stops tracking the reference.

## In the article

- [Section 5.1 · A constrained head on a network trained on image data](../../../../article/manuscript/ReadMe.md#section-51--a-constrained-head-on-a-network-trained-on-image-data). The paragraph headed “The ceiling.”
- [Appendix L · The ceiling: embedding dimension against record length](../../../../article/manuscript/ReadMe.md#appendix-l--the-ceiling-embedding-dimension-against-record-length). Embedding-dimension and record-length sweeps in Tables 23–24.

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

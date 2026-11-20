# Zero learning rate

Freeze the optimizer to check whether the scalar observation changes when its parameters are held fixed.

## In the article

- [Section 4 · Validation protocol](../../../../article/manuscript/ReadMe.md#section-4--validation-protocol). The requirement that the observed signal be constant when learning is switched off.
- [Section 5.1 · A constrained head on a network trained on image data](../../../../article/manuscript/ReadMe.md#section-51--a-constrained-head-on-a-network-trained-on-image-data). The zero-learning-rate test and the rejected mini-batch loss.
- [Appendix C · Per-observer results and alternative statistics](../../../../article/manuscript/ReadMe.md#appendix-c--per-observer-results-and-alternative-statistics). Table 6 marks the systems that fail the check.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Handwritten digits and estimator calibration](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Zero-learning-rate experiment](../run_validation.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.

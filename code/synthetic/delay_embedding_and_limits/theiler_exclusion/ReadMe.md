# Theiler exclusion and transients

Sweep temporal separation and compare the capped and full exclusion in the wider configuration.

## In the article

- [Section 6.1 · Recovery in each regime](../../../../article/manuscript/ReadMe.md#section-61--recovery-in-each-regime). Why exclusion changes the transient estimate.
- [Appendix I.1 · The exclusion cap at the twenty-component configuration](../../../../article/manuscript/ReadMe.md#appendix-i1--the-exclusion-cap-at-the-twenty-component-configuration). The capped and uncapped comparison in Table 19.
- [Appendix I.2 · The Theiler exclusion and the dimension of a transient](../../../../article/manuscript/ReadMe.md#appendix-i2--the-theiler-exclusion-and-the-dimension-of-a-transient). The exclusion sweep in Table 20 and Figures 13–14.

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

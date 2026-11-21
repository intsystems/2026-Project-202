# Theiler exclusion and transients

Sweep temporal separation and compare the capped and full exclusion in the wider configuration.

## In the article

- [Section 2.3 · Pooled local-dimension estimation](../../../../article/manuscript/ReadMe.md#section-23--pooled-local-dimension-estimation). The temporal exclusion used when selecting admissible neighbors.
- [Appendix B · Configurations and reproducibility](../../../../article/manuscript/ReadMe.md#appendix-b--configurations-and-reproducibility). The rule, its cap, and their interaction with the lag. The standalone transient sweep is from the earlier study.

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

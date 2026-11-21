# Zero learning rate

Freeze the optimizer to check whether the scalar observation changes when its parameters are held fixed.

## In the article

- [Section 3.1 · Analytic oscillators and learning](../../../../article/manuscript/ReadMe.md#section-31--analytic-oscillators-and-learning). The parameter norm and fixed-probe loss become constant at zero learning rate.
- [Appendix C.1 · Analytic and driven-learning evaluation](../../../../article/manuscript/ReadMe.md#appendix-c1--analytic-and-driven-learning-evaluation). The frozen-parameter test for the driven systems and classifier observables.

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

# Known modes in linear learning

In this linear regression, momentum makes the weights oscillate around the least-squares solution. Figure 2 of the article uses the training loss, half the mean squared prediction error over all training examples. The standalone plot uses the signed prediction error on one fixed test input.

## In the article

- [Section 3.1 · Independent oscillations and harmonics](../../../../article/manuscript/ReadMe.md#section-31--independent-oscillations-and-harmonics). Estimating the number of oscillating weight directions from the training loss.
- [Appendix C.2 · Dense regression with endogenous modes](../../../../article/manuscript/ReadMe.md#appendix-c2--dense-regression-with-endogenous-modes). The loss formula, oscillation frequencies and selective-damping comparison.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Linear learning and recurrent dynamics](../ReadMe.md)

- [research_known_modes.py](research_known_modes.py). Collection and scoring source. Runs the deterministic linear experiment and measurements.
- [collect_logs.py](collect_logs.py). Run the linear simulation and score its scalar records.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py
```

# Walker2d policy selection

Select policies for transfer to actuator lag and noise using nominal reward, MG, entropy or a fixed policy.

## In the article

- [Appendix F.2 · Selecting policies for actuator shifts](../../../../article/manuscript/ReadMe.md#appendix-f2--selecting-policies-for-actuator-shifts). Nominal acquisition, target evaluation, and comparisons with simple selection rules.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.
- [analyze.py](analyze.py). Replay all selectors and score their choices from the bundled episodes.
- [checkpoint_selection.py](checkpoint_selection.py) and [collect_logs.py](collect_logs.py). Generate new episodes from policy checkpoints.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python analyze.py
```

The CSVs cover the pilot seed 231 and evaluation seeds 232–235. The paper's averages use the four evaluation seeds.

The policy weights and normalization files required for new simulation rollouts are separate inputs. If you have them, install PyTorch, Gymnasium with MuJoCo, and Stable-Baselines3, then run:

```bash
python collect_logs.py 232 --checkpoint-root /path/to/checkpoints
```

The checkpoint root must contain directories such as `seed232_lambda0/step0262144/` with `policy.zip` and `normalize.pkl` for each combination of smoothing coefficient and training step.

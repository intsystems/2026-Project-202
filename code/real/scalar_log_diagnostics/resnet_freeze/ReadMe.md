# ResNet-18 backbone freeze

Use a weight-norm log to detect backbone freezing on CIFAR-10. Compare scratch and ImageNet initializations.

## In the article

The current manuscript does not report this experiment.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Diagnostics of model dynamics](../ReadMe.md)

- [run_resnet.py](run_resnet.py). Original source. Train ResNet, export the observers, or compute rolling statistics and detector outputs.
- [analyse_resnet.py](analyse_resnet.py). Original source. Train ResNet, export the observers, or compute rolling statistics and detector outputs.
- [export_observers.py](export_observers.py). Original source. Train ResNet, export the observers, or compute rolling statistics and detector outputs.
- [collect_logs.py](collect_logs.py). Collect ResNet training logs for the selected scenario.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --help
```

# ResNet-18 backbone freeze

Use a weight-norm log to detect backbone freezing on CIFAR-10. Compare scratch and ImageNet initializations.

## In the article

- [Section 4.1 · Detecting restricted parameter updates](../../../../article/manuscript/ReadMe.md#section-41--detecting-restricted-parameter-updates). Backbone freezing in scratch and ImageNet-initialized ResNet-18.
- [Appendix D.1 · CNN and ResNet interventions](../../../../article/manuscript/ReadMe.md#appendix-d1--cnn-and-resnet-interventions). Training conditions and the frozen detector and update-sketch reference.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Diagnostics of model dynamics](../ReadMe.md)

- [run_resnet.py](run_resnet.py). Train ResNet and record weight norms and compressed updates.
- [analyse_resnet.py](analyse_resnet.py). Compute rolling statistics and apply the [frozen detector](detector_rule.json).
- [export_observers.py](export_observers.py). Export scalar observations from the raw logs.
- [collect_logs.py](collect_logs.py). Collect ResNet training logs for the selected scenario.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python analyse_resnet.py --windows logs/scratch_windows.csv
python analyse_resnet.py --windows logs/finetune_windows.csv
```

To train a new set of models and score its logs, install PyTorch and torchvision, then run:

```bash
python collect_logs.py --scenario scratch
python analyse_resnet.py --root results_resnet/scratch
```

Use `--scenario finetune` and `--root results_resnet/finetune` for ImageNet initialization. Training can download CIFAR-10 and ImageNet initialization weights.

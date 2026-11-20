# ResNet-18 backbone freeze data

CIFAR-10 runs with random initialization (`scratch`) and ImageNet initialization (`finetune`). Compare the normal runs with runs where the backbone freezes.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`scratch_param_norm.csv`](scratch_param_norm.csv), [`finetune_param_norm.csv`](finetune_param_norm.csv) | Weight norm at each training step by intervention and seed. |
| [`scratch_windows.csv`](scratch_windows.csv), [`finetune_windows.csv`](finetune_windows.csv) | Window-level MG, update participation ratio, update size, and scalar diagnostics. |
| [`scratch_detector_per_run.csv`](scratch_detector_per_run.csv), [`finetune_detector_per_run.csv`](finetune_detector_per_run.csv) | Event detections, false alarms, and detection delay by statistic, intervention, and seed. |

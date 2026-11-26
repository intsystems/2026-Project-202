# Experiment code

| Folder | What you will find |
| --- | --- |
| [`synthetic/`](synthetic/ReadMe.md) | Controlled systems, calibration on handwritten digits, estimator limits, and supplementary linear-learning and RNN experiments. |
| [`real/`](real/ReadMe.md) | Grokking and stability, diagnostics of model dynamics, and selection and stopping experiments. |
| [`actdim/`](actdim/ReadMe.md) | The shared estimator, trajectory sketch, and rank calculations. |
| [`experiment_runner/`](experiment_runner/ReadMe.md) | Common code for running collections and saving plots. |

Within an experiment, `logs/ReadMe.md` describes the CSVs, `figures/ReadMe.md` lists the PDF plots, and `analyze_logs.py` redraws the plots. The experiment README gives the available commands and identifies any additional inputs for collecting new logs.

## Run the code

From this directory, install the analysis dependencies.

```bash
python -m pip install -r requirements.txt
```

From an experiment directory, run `python analyze_logs.py` to render figures using the bundled tables. Use `--output PATH` to choose an output directory. The default is `code/runs/redrawn/<experiment>`. The prebuilt exports are in each `figures/` directory. Some controlled simulations retained aggregate CSV results and configuration files. Use the collection command to generate a new run. Supplemental figures summarize the bundled tables or show selected results.

The [manuscript guide](../article/manuscript/ReadMe.md#reproduce-reported-measurements) maps reported values to their data. The [paper figure guide](../article/manuscript/figures/ReadMe.md) gives commands for the figures in the article.

For fresh neural network training, install PyTorch with `python -m pip install "torch>=1.13"` or use a build that supports your GPU. ResNet and MNIST also need torchvision. See each experiment page for datasets, checkpoints, and hardware. Use the bundled outputs for exact comparisons with the paper. You can replay Walker2d policy selection from saved episodes. New MuJoCo rollouts require the original policy weights and normalization files.

[Publication home](../ReadMe.md)

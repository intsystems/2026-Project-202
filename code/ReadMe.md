# Experiment code

| Folder | What you will find |
| --- | --- |
| [`synthetic/`](synthetic/ReadMe.md) | Controlled systems, calibration on handwritten digits, estimator limits, and synthetic poster examples. |
| [`real/`](real/ReadMe.md) | Grokking and stability, diagnostics of model dynamics, and selection and stopping experiments. |
| [`actdim/`](actdim/ReadMe.md) | The shared estimator, trajectory sketch, and rank calculations. |
| [`experiment_runner/`](experiment_runner/ReadMe.md) | Common code for running collections and saving plots. |

Within an experiment, `logs/ReadMe.md` describes the CSVs, `figures/ReadMe.md` lists the PDF plots, `analyze_logs.py` redraws those plots from CSVs, and `collect_logs.py` starts a fresh collection. The experiment README links to the implementation and describes any extra inputs.

## Run the code

From this directory, install the analysis dependencies.

```bash
python -m pip install -r requirements.txt
```

From an experiment directory, run `python analyze_logs.py` to render figures using the bundled tables. Use `--output PATH` to choose an output directory. The default is `code/runs/redrawn/<experiment>`. The prebuilt exports are in each `figures/` directory. Some controlled simulations retained aggregate CSV results and configuration files. Use the collection command to generate a new run. Supplemental figures summarize those tables or show individual poster panels.

For fresh training, install PyTorch with `python -m pip install "torch>=1.13"` or use a build that supports your GPU. See the experiment page for datasets, checkpoints and hardware. ResNet and MNIST also need torchvision. Walker2d needs Gymnasium with MuJoCo and Stable-Baselines3. The [experiment runner](experiment_runner/ReadMe.md) handles training schedules and writes fresh outputs to the local runs directory. These results can differ from the supplied values. The original project retained some upstream artifacts locally, and the local archive preserves the historical workflows that depend on them.

[Publication home](../ReadMe.md)

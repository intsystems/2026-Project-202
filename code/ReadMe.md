# Experiment code

- [Synthetic experiments](synthetic/ReadMe.md)
- [Experiments on recorded model runs](real/ReadMe.md)
- [Active dimension methodology](actdim/ReadMe.md)

Each experiment has a collection entry point, a CSV analysis and plotting entry point, and separate directories for its CSVs and PDF figures. Some entries include the original training, scoring or configuration files. The individual pages describe runtime requirements and inputs.

## Run the code

From this directory, install the analysis dependencies.

```bash
python -m pip install -r requirements.txt
```

From an experiment directory, run `python analyze_logs.py` to render figures using the bundled tables. Use `--output PATH` to choose an output directory. The default is `code/runs/redrawn/<experiment>`. The prebuilt exports are in each `figures/` directory. Some controlled simulations retained aggregate CSV results and configuration files. Use the collection command to generate a new run. Supplemental figures summarize those tables or show individual poster panels.

For fresh training, install PyTorch with `python -m pip install "torch>=1.13"` or use a build that supports your GPU. See the experiment page for datasets, checkpoints and hardware. ResNet and MNIST also need torchvision. Walker2d needs Gymnasium with MuJoCo and Stable-Baselines3. The [experiment runner](experiment_runner/ReadMe.md) handles training schedules and writes fresh outputs to the local runs directory. These results can differ from the supplied values. The original project retained some upstream artifacts locally, and the local archive preserves the historical workflows that depend on them.

[Publication home](../ReadMe.md)

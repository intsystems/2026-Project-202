# Text VAE code loss data

Penn Treebank runs with baseline training, latent-code suppression (`regularized`), and free-bits protection (`protected`).

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| `vae_*_log.csv` (3 files, [example](vae_base_log.csv)) | Reconstruction negative log-likelihood on the probe set at each logged training step. |
| `vae_*_mg.csv` (3 files, [example](vae_base_mg.csv)) | MG and companion statistics computed over windows of the probe loss. |
| `vae_*_mi.csv` (3 files, [example](vae_base_mi.csv)) | Independent mutual-information estimate by training step. |
| [`all_seeds.csv`](all_seeds.csv) | Per-seed summary of retention and MG across the three branches. |

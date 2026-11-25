# Sentence information in a text VAE

Compare KL penalties with and without free bits on Penn Treebank. Plot scalar reconstruction loss, rolling MG and mutual information between sentences and the latent variable.

## In the article

- [Section 4.2 · Tracking sentence information in a text VAE](../../../../article/manuscript/ReadMe.md#section-42--tracking-sentence-information-in-a-text-vae). Compare changes in reconstruction loss with mutual information and decoder sensitivity.
- [Appendix D.4 · Text VAE and preservation reference](../../../../article/manuscript/ReadMe.md#appendix-d4--text-vae-and-preservation-reference). Comparison with the independently measured retention of latent information.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Diagnostics of model dynamics](../ReadMe.md)

- [run.py](run.py). Original source. Baseline training, free-bits branch or analysis.
- [run_protection.py](run_protection.py). Original source. Baseline training, free-bits branch or analysis.
- [analyze_protection.py](analyze_protection.py). Original source. Baseline training, free-bits branch or analysis.
- [collect_logs.py](collect_logs.py). Collect the paired VAE branches.
- [recompute_metrics.py](recompute_metrics.py). Score the three branches from a fresh run.
- [analyze.py](analyze.py). Analyze the fresh baseline and suppression branches.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --seed 1
python recompute_metrics.py --seed 1
```

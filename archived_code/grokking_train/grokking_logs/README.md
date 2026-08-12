# This directory is empty by design

It is the default `--outdir` of `python train.py`, nothing more. Its `.gitignore` excludes
`*.csv`, so anything you train into it stays out of the repository. **No log the article uses
is stored here**, and a reader who follows a pointer to `grokking_train/grokking_logs/`
expecting the paper's data will find nothing.

The logs live in three places, by role:

| What | Where | Read by |
| --- | --- | --- |
| The six original runs of the run inventory — `mod_wd1`, `mod_wd1_s43`, `mod_wd1_s44`, `s5_wd1`, `mod_wd0`, `s5_wd0` | [`../../grokking_analysis/grokking_logs/`](../../grokking_analysis/grokking_logs/) (5 CSVs, canonical) and, per-run and already split out by name, [`../../active_rank/results_fine/*_train.csv`](../../active_rank/results_fine/) | `grokking_analysis/experiments.py`, `grokking_analysis/reproduce_figures.py`, `active_rank/dip.py`, `edm_validation/phase10_identifiability_audit.py` |
| The seven extended 120 000-step reruns — `grokpos_s0`, `lowdata15_s0/s1/s2`, `lowdata20_s0`, `wd0_s0/s1` | [`../../dimension_recovery/results/extended/`](../../dimension_recovery/results/extended/) | `active_dimension/e5_real_logs.py` |
| The Colab notebooks and sweep scripts that originally produced the first set | [`../../Grokking/`](../../Grokking/) | kept for history; nothing in `grokking_train/` depends on it |

The filenames in `grokking_analysis/grokking_logs/` are the historical ones and do not match
the run keys; `../runs.py` is the registry that maps each key to the configuration and the log
it produced, and the run table in [`../README.md`](../README.md) gives the key-to-filename
correspondence.

To refresh the canonical copies rather than writing here:

```bash
python train.py --article --outdir ../grokking_analysis/grokking_logs
```

# Grokking Training — the log producer

The other half of [`../grokking_analysis/`](../grokking_analysis/): this module *makes* the
CSV logs that one *reads*. Notebook-free, one importable package (`grok/`), one run registry
(`runs.py`) and one CLI (`train.py`).

Supersedes the fifteen near-duplicate `generator_logs_*.ipynb` notebooks in
[`../Grokking/`](../Grokking/) with a single training loop driven by a `RunConfig`. Adding a
task or an observable is a change in one place, not in fifteen. That directory is kept for its
history and for the raw exploratory logs, but nothing here depends on it.

```
train.py <run>  ->  grokking_logs/<run>.csv  ->  reproduce_figures.py <figure>  ->  figures/
```

## Quick start

```bash
pip install -r requirements.txt

python train.py --list                   # what can be trained
python train.py s5_wd1                   # one run -> ./grokking_logs/
python train.py --article                # every log the paper's figures need
python train.py s5_wd1 --dry-run         # resolve the config, train nothing
python test_train.py                     # sanity checks (also runs under pytest)
```

To feed the figure pipeline directly:

```bash
python train.py --article --outdir ../grokking_analysis/grokking_logs
cd ../grokking_analysis && python reproduce_figures.py
```

CPU cost on a laptop: ~7 min each for `s5_wd1` / `s5_wd0` / `mod_wd1` / `mod_wd0`, ~50 min
for `full_batch` (a full-batch forward on 4704 examples every step). A GPU makes the
mini-batch runs roughly interactive.

## Layout

| Path | Contents |
| --- | --- |
| `grok/groups.py` | Finite-group algebra: permutations by lexicographic rank, Lehmer-code ranking, `S_n` Cayley tables. NumPy only — no torch |
| `grok/tasks.py` | `a + b (mod p)` and composition in `S_n` as `[a, b, =] -> a*b` prompts, plus the train/val split |
| `grok/models.py` | `OmnigrokTransformer` (1L, LayerNorm-free — Figs. 1–3) and `EncoderTransformer` (stock `nn.TransformerEncoder` — App. B). **The name is historical**: this configuration is Nanda et al. (2023)'s, which Liu et al. (2022) adopt rather than originate. It is kept because the registry key and parameter order are load-bearing for bit-identical reproduction. |
| `grok/metrics.py` | The scalar observables: weight norm, accuracy, and the gradient probe |
| `grok/config.py` | `RunConfig` — one dataclass describing a run end to end |
| `grok/loop.py` | The training loop; the only place that touches the optimizer |
| `runs.py` | Registry mapping each published log to the configuration that produced it |
| `train.py` | CLI |
| `test_train.py` | Group-algebra, registry and end-to-end checks |
| `grokking_logs/` | Output |

## Runs

`*` marks the logs the article's figures are actually built from.

| Key | Task | Optimizer | Log | Grokking |
| --- | --- | --- | --- | --- |
| `mod_wd1` * | modular addition, $p=113$, 30 % train | AdamW, WD=1.0, batch 256 | `..._to_flat_grokking_with_stochastic.csv` | step 13810 |
| `mod_wd0` * | modular addition, $p=113$ | AdamW, WD=0.0 | `..._to_flat_grokking_with_stochastic_without_wight_decay.csv` | never |
| `s5_wd1` * | $S_5$ composition, 50 % train | AdamW, WD=0.2, batch 256 | `..._S_5_with_stochastic.csv` | step 6735 |
| `s5_wd0` * | $S_5$ composition | AdamW, WD=0.0 | `..._S_5_with_stochastic_without_wight_decay.csv` | never |
| `full_batch` * | modular addition, $p=97$, 50 % train | AdamW, WD=1.0, **full batch** | `grokking_modular_addition_logs.csv` | step 3270 |
| `sn` | $S_n$ composition — template | AdamW, WD=1e-3, batch 256 | `grokking_s{n}_logs.csv` | — |
| `sn_wd0` | $S_n$ composition — the WD=0 control | AdamW, WD=0.0 | `grokking_s{n}_logs_without_weight_decay.csv` | — |

`runs.RUNS[key]` is the same `RunConfig` the CLI uses, so the registry is importable:

```python
import runs
from grok import train

df, path = train(runs.get("s5_wd1"), outdir="grokking_logs")
```

## Log schema

One row per *logged* step. `log_every` therefore sets the sampling rate of the reconstructed
attractor and must stay constant within a run — `edm.sliding_dimension` slides its window over
the row grid, not over the step numbers.

| Column | Meaning |
| --- | --- |
| `step` | optimization step (not row index — rows are `log_every` apart) |
| `train_loss` / `train_acc` | on the current mini-batch, measured *before* its update |
| `val_loss` / `val_acc` | on a fixed slice of the validation split (`val_batch_size`, default 512) |
| `weight_norm` | $\lVert w \rVert_2$ over every parameter — the non-generic observable of Figs. 1–2 |
| `grad_norm`, `embed_grad_norm` | $\lVert g \rVert_2$ over all parameters / over the token embedding |
| `grad_cosine` | cosine between consecutive gradients; 0.0 on the first step |

`RunConfig.columns` picks the subset and the order. Three presets cover the published logs:
`BASE_COLUMNS` (modular addition), `FULL_COLUMNS` (adds the three gradient diagnostics, used
for $S_5$), `BASELINE_COLUMNS` (App. B: no weight norm, and `train_acc` before `val_loss`).
`step`, `train_acc` and `val_acc` are mandatory — `edm.load_logs` rejects a log without them.

The gradient probe costs a full parameter-sized `cat` and copy per step, so it only runs when
one of its three columns is requested.

## Reproduction fidelity

`mod_wd1`, `mod_wd0`, `s5_wd1` and `s5_wd0` reproduce the published CSVs **bit for bit** on
CPU — the residual is 1 ulp of float64, i.e. the decimal round-trip of the CSV itself:

```
s5_wd1, first 3 logged steps, max |reproduced - published| per column
  train_loss 8.9e-16   val_loss 8.9e-16   train_acc 0   val_acc 0
  weight_norm 0        grad_norm 5.6e-17  embed_grad_norm 2.8e-17  grad_cosine 3.6e-16
```

`full_batch` does **not** reproduce: the App. B notebook never called `torch.manual_seed`, so
its initialisation and its train/val split are unrecoverable. The registry entry sets
`seed=None` to record that. Everything else about it — architecture, optimizer, step budget,
column order, row count — is pinned, so a rerun is a statistically equivalent baseline, not the
identical series.

### The double-step bug

`generator_logs_to_S_5_with_stochastic*.ipynb` call `optimizer.step()` **twice** per batch: the
same gradient is applied twice and the AdamW state advances twice, so the run sees roughly
double the intended learning rate and weight decay. The published $S_5$ logs cannot be
reproduced without it, so it lives behind `RunConfig.double_step` and is set only on `s5_wd1`
and `s5_wd0`. It is off everywhere else, including the `sn` templates — **leave it off for new
runs.**

If you want to know what $S_5$ looks like without it:

```bash
python train.py s5_wd1 --set double_step=false --set csv=s5_wd1_single_step.csv
```

### RNG discipline

The task builders call `torch.manual_seed` themselves and then draw **exactly one**
`torch.randperm` for the split; the model is constructed afterwards and continues the same
stream. Inserting a torch RNG call into `grok/tasks.py` silently changes the initial weights of
every run. Sub-sampling (`max_pairs`) deliberately uses NumPy's generator for that reason.

Two more things the logs depend on: `dtype` (the mini-batch runs are `float64` — the notebooks
set it globally) and the device, since float64 matmuls differ in their last bits between CPU and
CUDA. The bit-exact comparison above is CPU-to-CPU.

## Growing the task: $S_n$ and beyond

`S_5` is not special-cased anywhere. `grok/groups.py` builds any `S_n` by ranking permutations
with their Lehmer code, so there is no `n!`-sized lookup table and nothing to change:

```bash
python train.py sn --set n=6 --set max_steps=200000     # -> grokking_s6_logs.csv
python train.py sn_wd0 --set n=6 --set max_steps=200000 # the control
```

`csv` is a `str.format` template over the config fields, which is why `{n}` resolves. Any
`RunConfig` field can be overridden the same way; `--set batch_size=none` means full batch and
`--set columns=step,train_acc,val_acc,val_loss` trims the log.

The product set grows as $(n!)^2$, so $S_6$ is 518 400 pairs and $S_7$ is 25.4 M — past both the
guard in `groups.py` and the point where the full Cayley table is a trainable dataset. Sample it
instead:

```bash
python train.py sn --set n=7 --set max_pairs=500000 --set max_steps=300000
```

Sampled pairs are composed directly, so the table is never materialised. `--set fraction=...`
still splits train/val inside whatever was sampled.

What to expect: the paper's claim (Sec. 4.2) is that the post-grokking dimension plateaus at the
smallest **faithful** irreducible representation of the group — 4 for $S_5$, and
`grok.minimal_faithful_dimension(n) == n - 1` in general, since the two 1-D representations are
unfaithful. That claim is estimator-dependent; read
[`../grokking_analysis/README.md`](../grokking_analysis/README.md#the-mackayghahramani-correction)
before treating a measured plateau as confirmation.

### A different task altogether

`grok/tasks.py` is a registry of builders returning a `Task` (prompt tensors, target tensors,
`num_classes`). Any binary operation on a finite set fits — add a builder, add it to `TASKS` and
name its extra `RunConfig` fields in `_TASK_ARGS`. Nothing in `models.py`, `metrics.py` or
`loop.py` knows what the operation is.

## Tests

`python test_train.py` (or `pytest test_train.py`). The group algebra and the registry checks
run **without torch**; the training checks skip themselves if it is missing.

Worth knowing what they pin:

* the vectorised Cayley table equals the notebooks' nested-loop construction element for element
  ($S_3$, $S_4$, $S_5$) — the element numbering is what makes the $S_5$ logs comparable;
* group axioms on $S_5$ (identity, Latin square, associativity) and non-commutativity;
* each registry entry's column names, column **order** and row count against the actual CSVs in
  `../grokking_analysis/grokking_logs/`, and that every figure's log has a run that produces it;
* `permute().reshape()` in `Attention` equals the `einops.rearrange` it replaced (skipped if
  einops is absent — it is no longer a dependency);
* a fresh log round-trips through `edm.load_logs` and `edm.sliding_dimension`.

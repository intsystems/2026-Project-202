# GAN diversity and scalar MG: executed exploratory pilot

Start with **report_ru.pdf** (Russian, four pages), then `summary.json`.
All experiments executed locally on CPU on 2026-09-29. This is **one GAN training
seed with matched continuations**, not a multi-seed confirmatory experiment.

## What actually happened

- Stacked MNIST, 6144 steps: only 6.05% of generated triples passed the independent
  confidence filter. Quality gate failed; no positive collapse example claimed.
- Ordinary single-digit MNIST, extended to12288 steps: all10 digit classes, 9.63
  effective classes among classifier-accepted images, 62.89% accepted. The
  predeclared70% quality gate also failed and was NOT relaxed.
- Exploratory matched branches to16384: unchanged; G learning rate x10; D rate
  /10; frozen generator including BatchNorm. None caused sustained class collapse.
- Primary MG late/early: unchanged1.005, G-fast1.140, D-slow1.531, frozen1.274.
  Frozen generator tensors, images and classifier predictions are identical.
  Hence a changed MG of GAN loss need not mean changed image diversity. This
  does NOT establish failure/success of a drop-only collapse detector because no
  qualified collapse event occurred. Nor does it refute MG for training dynamics.
- CIFAR-10, a GPU run and five held-out seeds were NOT performed. No claim of
  practical detector superiority follows merely from cheaper MG computation.

## Reproduce (from project root)

Dependencies and versions: `environment.json`. Python3.13/PyTorch2.10 CPU were used.
`torchvision.datasets.MNIST` downloads the real MNIST train/test split. Test images
are used only to evaluate the separate classifier. `code/actdim` supplies the
unchanged scalar estimator and is included in the compact archive.

```powershell
python research_gan_collapse/prepare.py
python research_gan_collapse/run.py --seed 0 --arm base --steps 6144 --switch 3072 --out research_gan_collapse/pilot/base
python research_gan_collapse/run.py --channels 1 --seed 0 --arm base --steps 6144 --switch 3072 --out research_gan_collapse/mnist_pilot/base
python research_gan_collapse/run.py --channels 1 --seed 0 --arm base --steps 12288 --switch 12288 --every 512 --resume research_gan_collapse/mnist_pilot/base/checkpoint_06144.pt --out research_gan_collapse/mnist_pilot/base_long
python research_gan_collapse/branches.py
python research_gan_collapse/analyze.py --root research_gan_collapse/mnist_branches
python research_gan_collapse/analyze.py --root research_gan_collapse/pilot
python research_gan_collapse/analyze.py --root research_gan_collapse/mnist_pilot
python research_gan_collapse/audit_benchmark.py
python research_gan_collapse/make_report.py
python research_gan_collapse/build_report.py
python research_gan_collapse/package_results.py
```

These commands regenerate/overwrite their output files. Use a separate copy of
the bundle to preserve the original run. XeLaTeX and Times New Roman are required
only for rebuilding the PDF; the Markdown report can be read independently.
The runner currently uses CPU tensors; GPU execution has not been implemented or
validated. Do not assume uploading it to Colab makes these timings GPU timings.

## Critical details

- G input64 independent normal coordinates, 28x28 outputs, non-saturating loss.
  G and D optimizers are both Adam(.5,.999), lr.0002, batch64; D then G per step.
  G-fast=.002; D-slow=.00002. Dataset/noise are identical before branch point.
- Evaluator: independent CNN trained on real MNIST, test accuracy98.91%; at
  confidence>=.9, accepted accuracy99.71% on REAL test digits. This does not prove
  precision on generated/OOD images. Contact sheets retained for inspection.
- Evaluate10000 fixed latent codes, and first512 for cheap reference. Confidence
  filtering is per digit; a stacked triple requires all3 pass. Effective modes
  means exp(label entropy), NOT active dimension. Counts are sample-dependent.
- MG E20/tau1/k20/Theiler39/W512; E40 uses the same exclusion. Primary g_loss,
  secondary d_loss. Right-endpoint timestamps only; no future data or best-channel
  selection. Other window/lag settings and all controls preserved.
- Late/early comparisons in report: median windows ending10240..12288 vs
 14336..16384, not a single selected window and not an independent-seed test.
- Training metadata reports4 effective Torch threads after inference started;
  independent matched benchmark explicitly recorded1. These regimes are not mixed
  in the report's speed table. CPU timing varied; each repeat is in benchmark.csv.
- Frozen control validates every G tensor including BN buffers and exact image
  equality. Loss still depends on the updating discriminator. Do not conflate a
  change in game dynamics with a change in generator diversity.
- No article sources or any of the six article PDFs were changed.

## Files

`PROTOCOL.md`: original settings plus every exploratory amendment, all made before
MG inspection. `EXPERIMENT_PROPOSAL_RU.md` is the earlier proposal, not a claim of
completion of the full CIFAR study. `pilot/base` is the unsuccessful stacked run;
`mnist_pilot` is the easier calibration; `mnist_branches` holds matched continuations.

Each run preserves logs.csv, reference.csv, metadata.json, checkpoint weights,
10k evaluator predictions and image sheets. `mnist_branches` also includes all
MG windows, sensitivity, surrogate controls, fresh-latent repeats, repeated timing,
exact frozen audit, and numerical summary.

`gan_collapse_pilot_results.zip` contains all CSV/JSON/prediction arrays/image
sheets, scripts, estimator code, classifier, report and final checkpoints for
each run. Full intermediate checkpoints and downloaded MNIST stay in the local
workspace; they are reproducible using the commands above. SHA256 file hashes
are in bundle_manifest.json; ZIP integrity is checked when packaging.

Source motivating the task: Lin et al., PacGAN, NeurIPS2018. This pilot is NOT a
reproduction of PacGAN's published architecture/results and uses no packing.

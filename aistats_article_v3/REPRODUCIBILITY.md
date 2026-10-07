# Reproducing this revision

The new forecasting and intervention tables and Figure 7 are rebuilt by `new_application_assets.py --bundled` using `evidence/practical_utility/`. Full MG and sampled-query MG are distinct implementations; the unchanged forecast choices are an empirical result on the 20-seed cohort, not equality of the estimates. The evidence contains per-record choices, errors, paired timings, original protocols and source snapshots. Repeating network training or evaluating a newly fitted selector requires the parent experimental workspace and its artifacts; the source archive does not claim a complete clean-room reproduction of every training run. The original studies and manuscript versions remain separate.

Run commands from `aistats_article_v3` using an environment with NumPy, pandas, SciPy, scikit-learn, matplotlib and threadpoolctl. Building PDFs additionally requires latexmk, pdfLaTeX, BibTeX, pdftotext and pdftohtml (or PyMuPDF for the color check).

```sh
python prepare_assets.py --bundled
python make_baseline_tables.py
python review_analysis.py
python new_application_assets.py --bundled
python validate_claims.py
python verify_pipeline.py
python build.py
python package.py
```

These commands read saved evidence, recompute tables including latency sensitivity and CUSUM, generate the main result figures, build black/blue PDFs and check the archive hashes. They do not run training. The input CSVs for the method illustration are in `evidence/method_figure/`; the ready-to-use vector figure is included. `update_method_figure.py` currently needs the parent project's figure module.

For measuring a new scalar log, use `mg_pipeline.py` and `MEASUREMENT.md`. Its estimator snapshot runs without the parent repository. `verify_pipeline.py` checks that the wrapper matches the kernel, that non-measurable windows remain marked, and that an affine rescaling preserves a clean periodic estimate.

Training/analysis sources are archived in `experiment_code/` with the parent project's relative layout. `source_snapshot_manifest.json` maps copies to original files and records SHA256 hashes. For example, from `experiment_code/`:

```sh
python research_known_modes.py
python research_generator/run.py --seeds 1 2 3 4 5
python research_trajectory_reference/cifar_graded.py --seeds 10 11 12 13
python research_text_vae/run.py --seed 1 --out research_text_vae/seed1
```

These are experiment entry points, not a claim that a clean-room training reproduction has been validated. Historical scripts use external CIFAR-10/Penn Treebank assets, sibling result folders, or initial checkpoints; these dependencies must be supplied. Walker continuations require the shared policy checkpoint and normalizer. The archive does not contain all datasets, weights, original run logs, or environment locks. Its primary verified reproduction level is numerical evidence to tables/figures/PDF, plus the standalone MG measurement kernel. The checklist therefore retains the corresponding `No` answers.

The archived digits implementation under `experiment_code/archived_code/active_dimension/` generated the historical scores. The refactored `code/actdim/systems/digits_parameter.py` uses different named random streams and is not asserted to reproduce every historical number bit for bit. Appendix C specifies the historical drive.

Hardware metadata recovered for timing experiments are in `evidence/compute_provenance/`; they do not retrospectively identify every historical machine. No new training or timing measurements were performed for the October 4 editorial revision. The detection-horizon table is a post-hoc re-scoring of stored first detections with frozen thresholds.

# Cyclical text-VAE monitoring: Russian report and reproducible experiment

Start with `report_ru.pdf` (Russian). The question is whether MG can allocate expensive reference checks effectively, after charging for scalar acquisition and comparison with cheap policies. This is a new experiment, not a replacement for the earlier two-arm or protected-arm findings.

## Scope and files

- `PROTOCOL.md`: fixed before pilot training, includes schedule, event definitions, monitoring rules, calibration grid and cost accounting.
- `seed100`: pilot only, excluded from confirmation results.
- `seed101` ... `seed109`: nine fresh initialization/batch seeds on the same fixed corpus. No omitted or replacement seeds.
- `selection.json`, `calibration.csv`: thresholds chosen using only pilot100, before confirmation training; hashes bind the choice to the pilot and protocol.
- Per-seed `logs.csv`: one observation before each update; `reference.csv`: after the indicated number of updates; `features.csv`: trailing windows with exclusive endpoint `end`.
- `events.json`: independent low/high conditions, persistent state and confirmed transitions. `evaluation.json`: every monitor decision, requested check, match and miss. Labels are not policy inputs.
- `scores.csv` / `all_scores.csv`: primary and secondary budgets, with all seeds kept. `events_by_seed.csv`: event counts and suitability. `summary.json`: cohort results, uncertainty, numerical flags and costs.
- `warmup.pt`: model/Adam/RNG after1024 updates; `final.pt`: final model. `meta.json`: initial model and random-stream hashes, CPU/threads/schedule.
- `benchmark.json`, `timings.csv`: warmed interleaved CPU operation timings. `costs.csv`: full cost and explicitly conditional already-logged-probe cost.
- `example.pdf`: seed101 chosen in advance, not the best seed. `cohort.pdf`: budgets and cost. `audit.json`: event edge cases, prefix-invariant causal policies, budget/cooldown, final reference recomputation.

The code imports the unchanged parent `run.py` and `code/actdim` estimator. Data provenance, exact vocabulary and row indices are in `../data/selection.json`. Corpus files are fetched by the existing loader if absent; the archive contains provenance but not the raw corpus.

## Reproduction

Run from the project root with the parent's `requirements.txt`. Existing completed training runs are protected against overwrites. A fresh full repetition needs an isolated copy/output tree, since calibration refuses to run after confirmation results already exist.

```powershell
python research_text_vae/cyclical/train.py --seed 100
python research_text_vae/cyclical/features.py --seed 100
python research_text_vae/cyclical/monitor.py --calibrate --seed 100
python research_text_vae/cyclical/confirm.py
python research_text_vae/cyclical/benchmark.py
python research_text_vae/cyclical/verify.py
python research_text_vae/cyclical/summarize.py
python research_text_vae/cyclical/build_report.py
python research_text_vae/cyclical/package.py
```

`confirm.py` uses at most two processes; each training process has two Torch threads. Cost benchmarking runs after this series completes, in one process, and does not use parallel training times. Other jobs can share the host; timings are local component estimates, not an isolated-machine end-to-end benchmark.

## Interpretation and accounting

Event labels require both information and decoder response to cross predeclared hysteresis thresholds for two consecutive reference checkpoints. The confirmation time is not backdated. Main events leave a full512-step detection horizon; later events are listed as censored. Recall and conditional delay must be read together. No exact active-dimensionality claim is made.

The monitor observes only previous scalar features, without labels or future rows, at grid spacing64. All feature policies use the same threshold-and-reset architecture; this does not establish that the selected KL policy is the best possible VAE-specific diagnostic. A fixed maximum budget can be exhausted early. Periodic comparison uses no phase search. Its matched-count version uses the final number of requested checks, not event times, and is explicitly an offline equal-count comparison.

One initial reference is charged to all policies. Feature acquisition charges7168 probe forwards, including warm-up; processing charges all105 feature endpoints. Ordinary training KL and beta are already computed by the training objective. Offline E40 diagnostics and dense labels are evaluation costs, not required by the deployed primary MG20 policy. Costs exclude ordinary training shared by all policies and the offline collection of truth used solely for evaluation.

The probe is reconstruction loss with an encoder that sees the sentence; it is not prior-generative perplexity. Cycles are repeated optimizer interventions, not proof of a recurrent invariant training trajectory. Same corpus and schedule across seeds limit transfer claims. A reduction in reference-check count is not automatically a speedup or equal-quality monitoring.

Method inspiration: Fu et al., *Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing*, NAACL2019, ACL Anthology N19-1021. The experiment uses a smaller model and its own fixed schedule; it is not a numerical reproduction.

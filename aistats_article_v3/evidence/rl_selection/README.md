# MG deployment selection experiment

Start with `report_ru.pdf` or `report_ru.md`. Neither the manuscript nor the stored training runs were modified.

Two hypotheses were evaluated on historical Walker2d policies:

* `run.py`: four final policies, actuator delays 1--3 steps after nominal warm-up; pilot230, test231--235. All five test decisions by MG coincide with fixed coefficient1. Most delayed episodes fall.
* `checkpoint_selection.py`: sixteen candidates, actuator lag0.05/0.1/0.2 and noise0/0.02, development231, confirmation232--235. Joint actuator shifts also apply during warm-up. `analyze.py` is authoritative for the common gate and all method/oracle summaries, including development re-scoring.

Commands from the parent project, using the existing `.venv_walker/Scripts/python.exe`:

```sh
python research_mg_deployment/run.py --seeds 230 231 232 233 234 235
python research_mg_deployment/checkpoint_selection.py 232
python research_mg_deployment/analyze.py
python research_mg_deployment/audit_and_time.py
python research_mg_deployment/make_report.py
```

Repeat checkpoint selection for each of233,234,235. Existing per-candidate caches avoid re-evaluation. Seed231 raw CSV is retained from the exploratory runner before caching was implemented; its incomplete-episode features are ignored by `analyze.py`. The development runner was executed twice due to an index-handling bug after data collection; repeat evaluations are not independent evidence. Type conversion/flag fixes precede confirmation. Full delay runtime also includes extra diagnostic delay0 episodes. Source and protocol files preserve these distinctions.

The archive contains scalar nominal traces for the first experiment and episode-level return/feature outputs for both. It does **not** contain every target state trajectory or the original trained models. `checkpoint_manifest.json` identifies80 required original checkpoint files and hashes. Replaying model evaluations and audit/timing needs the parent `research_walker_smooth_lambda` directory, its normalizers and `code/actdim` estimator. Recomputing selections and summary statistics from supplied CSV/JSON needs only NumPy/pandas. PDF building needs XeLaTeX and Times New Roman.

All candidate training cost is sunk and common; this experiment measures no training acceleration. Reset repetitions share a policy, fine-tuning seeds share one initial policy. No broad superiority or new curriculum result follows from the sample means. Confidence testing uses seed-paired differences (four units), not the many correlated evaluation episodes.

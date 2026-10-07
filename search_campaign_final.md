# Final campaign report: practical gains from MG

## The strongest verified result

The cleanest gain is a budgeted MG implementation for model routing in learned recurrent generators. The task is to choose a forecasting model from one scalar neuron trace. The frozen routing rule chooses between eight candidate forecasters on a 2,048-sample prefix and predicts the next 32 samples. The primary routing scenario uses neuron 2 and horizon 32.

Thirty new generator seeds (651--680), eight arms per seed including a chaos control, were evaluated after rules were frozen on seeds 601--618. Full MG and the query-budget estimator gave the same choice and the same forecast loss on all 160 forecasts. The 128-query implementation uses a 2,048-point delay cloud but estimates MG at 128 uniformly spaced query points.

On these 160 forecasts:

| Selector | Mean normalized MSE |
|---|---:|
| MG / 128-query MG | 0.03637 |
| Validation model selection | 0.04074 |
| TwoNN geometry | 0.04364 |
| Full geometric audit without MG | 0.03939 |
| LB intrinsic estimate | 0.04732 |
| Fixed / cheap scalar rules | 0.05--0.06 |
| Oracle | 0.00847 |

The gain over ordinary validation is about 10.7%; the paired bootstrap interval is reported in `query_intervals.csv` and includes zero, so this is a positive empirical result rather than a statistically definitive superiority claim.

The computational gain is stronger. On the same CPU benchmark, full MG routing takes about 0.199 s, 128-query MG about 0.012 s, and validation-based model selection about 0.045 s. The query approximation is approximately 16 times faster than full MG, while preserving all 160 routing decisions. The audit recomputes eight selected forecasts, checks full-query agreement, compares tree and blocked distance search, verifies affine rescaling, and rejects constant/nonfinite windows.

This result gives MG a practical role: a scalar-log geometry feature can route between downstream models while avoiding a full delay-cloud calculation. It also shows that the original full MG estimator can be approximated aggressively for a fixed decision task. The approximation is validated on this generator family; it is not a general theorem.

## Other settings

Walker2d checkpoint selection remains useful against nominal reward: target return 4.313 versus 4.005 and completion 90.8% versus 80.0%. Spectral entropy is better at 4.480 and 97.5%, so this is a qualified gain.

TwoLeadECG compact classification improves from 82.19% without MG to 91.31% with MG plus compact features, but ROCKET reaches 99.91%. The MG incremental result is real for the compact ablation and is weaker than a strong raw-series baseline. Wafer's apparent few-label perfect result was rejected after a ties-only control also reached 100%.

Real energy/traffic routing, HAR, physical sensor routing, certified stopping, and some UCR datasets did not show a robust MG advantage. The search therefore does not support “MG beats all analogues.” It supports a narrower claim: MG can be useful for geometry-aware routing when the decision concerns recurrent structure, and a query-budget approximation can make that routing cheap.

## Reproducibility

The final generator campaign is under `research_mg_real_forecast/`. Relevant files are `FINAL_CONFIRMATION.md`, `routing_v2_frozen.pkl`, `geometry_frozen.pkl`, `locked_decisions.csv`, `final_decisions.csv`, `final_audited_decisions.csv`, `final_audited_summary.csv`, `query_fresh_comparisons.csv`, `query_fresh_summary.csv`, `query_paired_summary.csv`, `query_audit.json`, and `query_freeze.json`. Old exploratory screens are retained separately. No paper edits were made from the negative or inconclusive settings.

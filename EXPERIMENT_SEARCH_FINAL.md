# Final search for practical MG gains

Date: 2026-10-05. This file records the result of a broad post-hoc/exploratory search. Only the Walker2d policy-selection result has been added to the AISTATS article. All other experiments are retained as validation, negative results, or hypotheses requiring independent confirmation.

## Most useful setting

In 30 new recurrent-generator seed groups (651--680, eight task arms each), MG routed between eight forecasters using neuron 2 and a 2,048-sample prefix to predict 32 future samples. Mean normalized MSE was 0.03139 for MG, 0.03224 for ordinary validation routing, 0.03540 for TwoNN, 0.04054 for a non-MG feature route, 0.04074 for validation in the frozen routing family, and 0.00759 for the unavailable oracle. At the seed-group level the MG versus validation interval includes zero. The result is a useful candidate routing signal, not proof of universal superiority.

The same MG decisions were reproduced by a 128-query approximation on 160 forecasts. Approximate MG routing reduced analysis time from about 0.199 seconds to 0.012 seconds in the paired timing benchmark. The approximation passed exact full-query, blocked-distance, affine-rescaling, constant-window and nonfinite-window audits. It is a new approximation and requires separate validation outside this generator family.

## Practical application already in the paper

For Walker2d checkpoint selection under hidden actuator lag/noise, MG gave target return 4.313 and 90.8% complete episodes, versus 4.005 and 80.0% for nominal reward. Spectral entropy was better (4.480 and 97.5%), while fixed pilot was essentially tied with MG. This is a qualified gain over reward-only selection.

## Broad screens

- FORCE certification stopping: MG did not beat output error or scalar recurrence in cost.
- UCI traffic and appliance energy: MG did not beat strong AR/recurrence routing.
- UCI HAR: adding MG to compact features improved subject accuracy only from 83.58% to 83.66%; the full 561-feature baseline reached 92.97%.
- UCR ECG and sensor tasks: MG improved some compact ablations, e.g. TwoLeadECG 82.19% to 91.31%, but ROCKET was 99.91%. ToeSegmentation1 improved 83.61% to 87.78%, but ROCKET was 96.11%.
- Wafer few-label success was an artifact: a ties-only feature reached 100% balanced accuracy, so the result is rejected as an MG advantage.
- Real building-sensor routing showed development gains that failed to transfer.
- A recurrent model-order screen and MG+recurrence tree screen did not establish superiority.

## Interpretation

The search supports a narrower claim: MG can provide a useful geometry-based routing signal from one scalar log, especially when the downstream choice concerns recurrent structure and when a query-budget approximation is acceptable. The search does not support the claim that MG is the best practical selector, the best early-stopping signal, or universally better than entropy, recurrence, validation, TwoNN, PR, or raw-series models.

## Additional checks from the new experiment ideas

The noisy-label MNIST intervention was rerun with a stratified fixed probe and discrete checkpoint rules. At 40% noise, MG early stopping reached mean clean-test accuracy 0.8475, close to the best simple level rule 0.8490 and fixed 512-step rule 0.8472. At 60% noise, MG and entropy/fixed-early stopping were about 0.7658, while clean-validation oracle was 0.7662. Freeze/prune branches on clean MNIST did not show an MG advantage over fixed intervention times. The first pilot replay used a bad checkpoint join and is excluded from claims; the corrected v2 results are retained under `research_mg_interventions/v2/`.

The Wafer few-label result was audited: MG values and missingness flags appeared almost perfectly predictive because normal/abnormal traces had different repeated-value structure. A ties-only baseline reached 100% balanced accuracy, so this apparent MG win is rejected. Strong ROCKET-style baselines also beat MG on ECG and motion tasks.

## Files

`research_mg_real_forecast/` contains forecasting screens, generator fresh/replication cohorts, frozen rules, query approximation audits, real UCI time-series tests and routing summaries. `research_mg_deployment/` contains Walker2d policy-selection results. `research_mg_certification/` contains the negative stopping result. `research_mg_ucr/` contains the UCR/HAR classifier screens and Wafer artifact audit. `mg_search_results.zip` is an earlier archive; the new `mg_search_campaign_final.zip` includes this final ledger and later artifacts.

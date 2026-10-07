# Extended MG practical search, October 5

The campaign tested decision-level uses of MG and preserved all negative results. The reference distinction remains essential: MG can be a useful geometry feature or change statistic without proving active dimension in stochastic/transient records.

## Strongest results

**Checkpoint selection for Walker2d under hidden actuator shifts.** Sixteen historical PPO policies are filtered by nominal performance and selected from one scalar action log. On four confirmation fine-tuning seeds, MG selection gives target return 4.313 and 90.8% complete episodes versus 4.005 and 80.0% for nominal reward. Spectral entropy is stronger at 4.480 and 97.5%; scalar recurrence is 4.398 and 95.0%. This is a practical gain over reward-only selection with a clear stronger competitor. The paper includes this qualified result.

**Model routing for learned recurrent generators.** A frozen selector chooses between compact harmonic and multi-frequency/AR-like forecasters from one neuron. On 30 final confirmation seed groups, MG reaches mean error 0.0314 in the primary neuron-2, horizon-32 scenario. The ordinary prefix validation winner gives 0.0322; the old MG rule gives 0.0322. TwoNN/other geometrical baselines are below the task-specific routing comparison in the extended audit, while the oracle is 0.0076. The paired bootstrap does not establish strong statistical superiority, and the selector is calibrated on earlier seeds. This is the strongest forecasting candidate but requires another clean protocol before entering the paper.

**Budgeted MG.** Evaluating MG at 64 or 128 query points instead of every delay vector keeps the routing error at 0.0330 versus 0.0314 for 256/full settings, while analysis time falls to about 0.011/0.015 seconds on the measured benchmark. Full MG was about 0.516 seconds in the same timing family. This is a concrete computational result for deployment, but the sampled estimator is a new approximation and needs its own controlled validation.

## Results that failed to beat strong baselines

- Certification stopping: MG success 87.5%, but output error and recurrence cost less; no saving was demonstrated.
- UCI energy/traffic forecasting: MG did not beat strong AR or recurrence routing.
- Smartphone HAR: adding MG gave only 83.66% versus 83.58% for cheap features; raw/full features reached 92.97%.
- UCR sensor classification: compact MG gains appeared on some datasets, but ROCKET/raw waveform baselines were stronger. Wafer's apparent 100% few-label result was an artifact of repeated-value/missingness structure; a ties-only baseline also reached 100%.
- Real building sensor routing: development gains did not transfer to later time blocks.
- Fixed model-order and MG+recurrence tree screens did not establish an additional robust advantage.

The final actionable direction is: **use MG as a geometry-based routing feature when the downstream decision is between models with different assumptions, and use a query-budget implementation when the scalar log is long.** The strongest current competitor remains scalar recurrence; claims should acknowledge this explicitly.

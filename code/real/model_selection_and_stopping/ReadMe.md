# Model selection and early stopping

These experiments use diagnostics to select a forecaster, deployment policy, or training checkpoint.

**In the article** · [Appendix F.2 · Selecting policies for actuator shifts](../../../article/manuscript/ReadMe.md#appendix-f2--selecting-policies-for-actuator-shifts) · [Section 4.8 · Forecast model selection from single-neuron recordings](../../../article/manuscript/ReadMe.md#section-48--forecast-model-selection-from-single-neuron-recordings) · [Appendix I · Comparing intervention rules](../../../article/manuscript/ReadMe.md#appendix-i--comparing-intervention-rules)

- [RNN forecast selection](rnn_forecast_selection/ReadMe.md). Compare reduced-query MG, full MG and held-out validation for choosing a model to forecast one neuron in trained recurrent generators.
- [Walker2d policy selection](walker2d_policy_selection/ReadMe.md). Select policies for transfer to actuator lag and noise using nominal reward, MG, entropy or a fixed policy.
- [Noisy MNIST early stopping](mnist_early_stopping/ReadMe.md). Use rules fixed on development runs to select stopping checkpoints in eight confirmation runs with 40% label noise.

[Recorded model runs](../ReadMe.md)

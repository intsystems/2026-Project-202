# Search for practical decision-level gains from MG

## Best supported settings

| Setting | Practical decision | MG result | Strongest competitor | Status |
|---|---|---|---|---|
| Walker2d checkpoint selection under hidden actuator lag/noise | Choose one stored PPO policy before deployment | Return 4.313 vs 4.005 for nominal-reward selection; 90.8% vs 80.0% complete target episodes | Spectral entropy: 4.480 and 97.5% | Useful relative to reward-only; no universal superiority |
| Learned recurrent generator forecasting | Choose harmonic or multi-frequency forecast model from one neuron | Confirmation MSE 0.569 vs 0.628 entropy and 0.661/0.698 fixed models | Scalar recurrence: 0.528 | Useful model routing; recurrence remains better |
| Smartphone HAR | Add MG features to compact sensor features | 83.66% subject accuracy vs 83.58% without MG | Full 561-feature baseline: 92.97% | Small incremental gain only |
| Wafer sensor classification, few labels | Use MG in a compact classifier | Apparent 99.96--99.98% balanced accuracy | Ties/missingness feature also 100% | Artifact, not a valid MG win |

## Negative or inconclusive settings

- MG-triggered expensive certification of FORCE training: MG maintained 87.5% success but was more expensive than recurrence, output error and fixed stopping.
- Real UCI traffic and appliance-energy forecasting: MG did not beat strong autoregression/recurrence baselines.
- Physical energy sensor routing: development gains did not transfer to later data.
- ECG/UCR classification: MG helped compact feature ablations on some tasks, but raw/ROCKET baselines were stronger. On TwoLeadECG, the compact MG result was beaten by ROCKET; on ECG200, ROCKET also beat MG.
- Fresh 256-unit generator replication: the gains from the old 1,000-unit generator screen did not survive as a broad superiority result. The selected neuron/horizon showed mixed behavior across new seeds and best alternatives.

## Practical conclusion

MG has a defensible applied role as a low-dimensional geometry feature for routing among downstream models or policies when only a scalar log is available. The strongest evidence is the Walker2d deployment selection and the recurrent-forecasting model choice. The experiments do not support claiming that MG is the best selector or that it universally improves training, stopping, curriculum, classification, or forecasting.

The main paper contains the audited Walker2d policy-selection result because it gives a concrete gain over nominal reward and explicitly reports the stronger spectral-entropy competitor. The other screens remain exploratory evidence and negative controls.

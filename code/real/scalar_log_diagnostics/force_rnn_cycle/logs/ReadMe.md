# FORCE RNN cycle data

Single-neuron activity before and after FORCE training. The rollouts use fixed weights at each checkpoint.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`force_checkpoint_00000_log.csv`](force_checkpoint_00000_log.csv), [`force_checkpoint_40000_log.csv`](force_checkpoint_40000_log.csv) | Activity of neuron 0 against evaluation time before and after training. |
| [`force_rolling_mg.csv`](force_rolling_mg.csv) | Windowed MG for the two recorded checkpoint traces. |
| [`force_training_traces.npz`](force_training_traces.npz) | Scalar rollouts used in the transition figure, sampled every 25 training iterations through iteration 1,000 and less frequently afterward. The arrays contain training iterations (`steps`), rollout times (`time`), and neuron activity (`activity`, one row per checkpoint). |
| [`force_all_seeds_endpoints.csv`](force_all_seeds_endpoints.csv) | MG and scalar diagnostics across evaluation seeds and sampled checkpoints. |
| [`force_independent_reference.csv`](force_independent_reference.csv) | Full-state reference metrics, including Lyapunov exponents and period estimates. |

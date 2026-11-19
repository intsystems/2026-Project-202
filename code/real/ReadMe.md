# Experiments on recorded model runs

These experiments apply MG and trajectory measurements to logs of trained networks and policies. Some tasks use generated targets or simulated environments.

| Folder | What you will find |
| --- | --- |
| [`grokking_and_stability/`](grokking_and_stability/ReadMe.md) | Transformer and perceptron grokking, with scalar diagnostics, trajectory measurements, and full-batch runs at the edge of stability. |
| [`dynamics_diagnostics/`](dynamics_diagnostics/ReadMe.md) | Detecting backbone freezing in ResNet, loss of latent information in a text VAE, and a learned oscillation in a FORCE RNN. |
| [`selection_and_stopping/`](selection_and_stopping/ReadMe.md) | Selecting an RNN forecaster, selecting a Walker2d policy under actuator lag and noise, and stopping training early on MNIST with noisy labels. |

[Code index](../ReadMe.md)

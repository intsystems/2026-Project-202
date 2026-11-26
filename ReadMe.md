# What Can a Scalar Log of Loss, Weight Norm, or Neuron Activation Reveal About a Neural Network?

What can a record of loss, weight norm, or one neuron's activation reveal about a network's dynamics? The [camera-ready article](article/artifacts_camera_ready.pdf) tests this question in recurrent generators, image classifiers, and a text variational autoencoder. It also evaluates whether scalar logs help select models or warn of training changes.

| Folder | What you will find |
| --- | --- |
| [`article/`](article/ReadMe.md) | The current paper, previous PDF editions, and the ICOMP poster. |
| [`code/synthetic/`](code/synthetic/ReadMe.md) | Controlled experiments with known dimensions, estimator checks, and learned recurrent generators. |
| [`code/real/`](code/real/ReadMe.md) | Recorded runs for grokking, training diagnostics, and model selection. |
| [`code/actdim/`](code/actdim/ReadMe.md) | The MG estimator and tools for compressing trajectories and calculating their effective rank. |

The [reproduction guide](code/ReadMe.md) explains how to regenerate figures from the bundled logs and identifies the experiments with fresh training commands.

[Setup and reproduction](code/ReadMe.md) · [MIT license](LICENSE)

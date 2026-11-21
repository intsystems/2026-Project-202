# Measuring dynamic simplification from scalar neural network logs

This repository accompanies the [article](article/ReadMe.md) and the ICOMP poster on estimating active dimension from scalar logs. The article tests dimension recovery on controlled systems and evaluates MG for monitoring training and selecting models.

| Folder | What you will find |
| --- | --- |
| [`article/`](article/ReadMe.md) | The AISTATS v3 manuscript in multiple styles, and the ICOMP poster. |
| [`code/synthetic/`](code/synthetic/ReadMe.md) | Controlled experiments with known dimensions, estimator checks, and learned recurrent generators. |
| [`code/real/`](code/real/ReadMe.md) | Recorded runs for grokking, training diagnostics, and model selection. |
| [`code/actdim/`](code/actdim/ReadMe.md) | The MG estimator and tools for compressing trajectories and calculating their effective rank. |
| [`archive/`](archive/ReadMe.md) | An index for the local archive of earlier project materials. |

Each experiment describes its CSVs, PDF figures and commands, and links to the article sections that discuss it.

[Setup and reproduction](code/ReadMe.md) · [MIT license](LICENSE)

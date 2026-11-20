# Measuring dynamic simplification from scalar neural network logs

This repository accompanies an article on estimating active dimension from a scalar log and an ICOMP poster on practical training diagnostics. The method uses delay coordinates to estimate local geometry from nearest-neighbor distances with temporal exclusion.

The controlled experiments test when the scalar estimate tracks a known number of active components. The scalar estimates on the grokking logs fail the dimension diagnostics. The stored trajectories show a drop and recovery in detrended effective rank near generalization in the regularized transformer runs.

| Folder | What you will find |
| --- | --- |
| [`article/`](article/ReadMe.md) | The article PDF and the ICOMP poster PDF. |
| [`code/synthetic/`](code/synthetic/ReadMe.md) | Controlled experiments with known dimensions, estimator checks, and the two synthetic poster examples. |
| [`code/real/`](code/real/ReadMe.md) | Recorded runs for grokking, training diagnostics, and model selection. |
| [`code/actdim/`](code/actdim/ReadMe.md) | The MG estimator and tools for compressing trajectories and calculating their effective rank. |
| [`archive/`](archive/ReadMe.md) | An index for the local archive of earlier project materials. |

Each experiment links to the CSVs, PDF figures, and commands for working with them. Start with the [handwritten digits experiment](code/synthetic/digits_validation/digits_classifier_head/ReadMe.md) for a test against a known reference, or with [transformer grokking](code/real/grokking_and_stability/transformer_grokking/ReadMe.md) for the article's main training result.

[Setup and reproduction](code/ReadMe.md) · [MIT license](LICENSE)

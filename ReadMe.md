# Measuring dynamic simplification from scalar neural network logs

This repository accompanies an article on estimating active dimension from a scalar log and an ASCOMP 2026 poster on practical training diagnostics. The method forms delay coordinates and estimates local geometry from nearest-neighbor distances with temporal exclusion.

The controlled experiments show when the scalar estimate tracks known active components. In the grokking experiments, scalar estimates fail the dimension diagnostics. The stored trajectories show a drop and recovery in detrended effective rank close to generalization in the regularized transformer runs studied.

- [Article and poster](article/ReadMe.md)
- [Experiment code and setup](code/ReadMe.md)
- [Local project archive](archive/ReadMe.md)
- [MIT license](LICENSE)

Start with the [synthetic experiment index](code/synthetic/ReadMe.md) for estimator validation or the [recorded model runs](code/real/ReadMe.md) for the grokking and poster applications.

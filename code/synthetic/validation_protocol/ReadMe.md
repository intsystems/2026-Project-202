# Digits and the validation protocol

The main controlled experiments on handwritten digits, the fixed estimator configurations, and the zero-learning-rate check applied across systems.

- [Digits parameter subspace](digits_parameter_subspace/ReadMe.md). A constrained head on a frozen model trained on handwritten digits. The number of controlled forcing phases supplies the reference.
- [Digits function subspace](digits_function_subspace/ReadMe.md). A function-space control on a digits model. The effective excitation did not meet the full protocol.
- [Estimator calibration](estimator_calibration/ReadMe.md). Configuration selection on the development seeds and ranks for the eight- and twenty-direction protocols.
- [Zero learning rate](zero_learning_rate/ReadMe.md). Freeze the optimizer to check whether the scalar observation changes when its parameters are held fixed.

[Synthetic experiment groups](../ReadMe.md)

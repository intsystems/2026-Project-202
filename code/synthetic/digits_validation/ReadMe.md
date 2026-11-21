# Handwritten digits and estimator calibration

Two experiments use models trained on handwritten digits to test whether MG tracks the number of active directions under controlled forcing. This folder also contains the fixed settings used in the comparison and the zero-learning-rate check applied across the controlled systems.

**In the article** · [Section 3.1 · Analytic oscillators and learning](../../../article/manuscript/ReadMe.md#section-31--analytic-oscillators-and-learning) · [Appendix C.1 · Analytic and driven-learning evaluation](../../../article/manuscript/ReadMe.md#appendix-c1--analytic-and-driven-learning-evaluation) · [Appendix B · Configurations and reproducibility](../../../article/manuscript/ReadMe.md#appendix-b--configurations-and-reproducibility)

| Folder | What you will find |
| --- | --- |
| [`digits_classifier_head/`](digits_classifier_head/ReadMe.md) | The main Digits experiment. A constrained classifier head is evaluated under recurrent forcing, noise, and transient dynamics. |
| [`digits_function_space/`](digits_function_space/ReadMe.md) | A control that drives directions in function space. Its tables show when the measured effective rank falls below the intended reference. |
| [`estimator_calibration/`](estimator_calibration/ReadMe.md) | Development sweeps, selected settings, and held-out calibration results. |
| [`zero_learning_rate/`](zero_learning_rate/ReadMe.md) | A check of whether the scalar signal and estimate can keep changing when training updates are switched off. |

[All synthetic experiments](../ReadMe.md)

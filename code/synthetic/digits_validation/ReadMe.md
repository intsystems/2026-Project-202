# Handwritten digits and estimator calibration

Two experiments use models trained on handwritten digits to test whether MG tracks the number of active directions under controlled forcing. This folder also contains the fixed settings used in the comparison and the zero-learning-rate check applied across the controlled systems.

**In the article** · [Section 4 · Validation protocol](../../../article/manuscript/ReadMe.md#section-4--validation-protocol) · [Section 5.1 · A constrained head on a network trained on image data](../../../article/manuscript/ReadMe.md#section-51--a-constrained-head-on-a-network-trained-on-image-data) · [Appendix B.3 · Frozen configurations](../../../article/manuscript/ReadMe.md#appendix-b3--frozen-configurations) · [Appendix E · System constructions](../../../article/manuscript/ReadMe.md#appendix-e--system-constructions)

| Folder | What you will find |
| --- | --- |
| [`digits_classifier_head/`](digits_classifier_head/ReadMe.md) | The main Digits experiment. A constrained classifier head is evaluated under recurrent forcing, noise, and transient dynamics. |
| [`digits_function_space/`](digits_function_space/ReadMe.md) | A control that drives directions in function space. Its tables show when the measured effective rank falls below the intended reference. |
| [`estimator_calibration/`](estimator_calibration/ReadMe.md) | Development sweeps, selected settings, and held-out calibration results. |
| [`zero_learning_rate/`](zero_learning_rate/ReadMe.md) | A check of whether the scalar signal and estimate can keep changing when training updates are switched off. |

[All synthetic experiments](../ReadMe.md)

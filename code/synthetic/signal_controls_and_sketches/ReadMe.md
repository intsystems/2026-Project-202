# Robustness and trajectory measurement

Controls for interpreting changes in MG, tests with known dimension transitions, and validation of the trajectory sketch used to measure effective rank in the grokking runs.

**In the article** · [Section 2.3 · Pooled local-dimension estimation](../../../article/manuscript/ReadMe.md#section-23--pooled-local-dimension-estimation) · [Section 3.2 · Learned generators distinguish phases from harmonics](../../../article/manuscript/ReadMe.md#section-32--learned-generators-distinguish-phases-from-harmonics) · [Appendix E · Grokking](../../../article/manuscript/ReadMe.md#appendix-e--grokking)

- [Observer and scale controls](observer_controls/ReadMe.md). Compare estimator shifts under changes in observer gain, amplitude, forcing and coordinate rotation.
- [Dimension transitions](dimension_transitions/ReadMe.md). Switch between higher and lower counts of excited phases and test whether the estimate tracks the change and recovery.
- [Unequal excitation](unequal_excitation/ReadMe.md). Hold the number of phases fixed and vary their relative amplitudes to compare MG with the participation ratio.
- [Roughness and phase controls](phase_count_and_roughness/ReadMe.md). Match scalar roughness across different phase counts and vary observer roughness at fixed phase count.
- [Trajectory sketch validation](trajectory_sketch/ReadMe.md). Check CountSketch on known ranks, compare logs with and without recording, and measure recording cost.

[Synthetic experiment groups](../ReadMe.md)

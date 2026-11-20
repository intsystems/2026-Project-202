# Known modes in linear learning data

Scalar logs and estimates from linear learning with a known number of oscillatory modes.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`stationary_summary.csv`](stationary_summary.csv) | Median and range of MG by mode count, window length, and scalar signal. Used by the poster. |
| [`measurements.csv`](measurements.csv) | Window estimates by run, signal, and regime, including the phase switch and damped controls. |
| [`audits.csv`](audits.csv) | Checks on data rank, gradients, and invariant drift. |
| [`amplitude_control.csv`](amplitude_control.csv) | MG when the scalar signal is rescaled. |
| `stationary_r*.csv` (12 files, [example](stationary_r1_s0.csv)), `switch_r*.csv` (3 files, [example](switch_r4_s0.csv)), `damped_r*.csv` (3 files, [example](damped_r4_s0.csv)) | Per-step training and probe losses, parameter and momentum norms, and test loss. |

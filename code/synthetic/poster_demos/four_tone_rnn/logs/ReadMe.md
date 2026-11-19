# Four-tone RNN learning data

Autonomous activity from recurrent generators trained on targets with four tones and one, two, or four independent rhythms.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`runs.csv`](runs.csv) | One row per run, with target settings, learning fidelity, MG, and state-dynamics statistics. |
| `obs_H4_*.csv` (5 files, [example](obs_H4_s1.csv)), `obs_M4_*.csv` (5 files, [example](obs_M4_s1.csv)), `obs_T4_*.csv` (5 files, [example](obs_T4_s1.csv)) | Activity of three recorded neurons after training. The filename identifies the target condition and seed. |

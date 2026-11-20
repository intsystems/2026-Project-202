# Walker2d policy selection data

Policy checkpoints evaluated under nominal conditions and with actuator lag and noise.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| `checkpoint_selection_s*.csv` (5 files, [example](checkpoint_selection_s231.csv)) | Rollouts by training seed, checkpoint, evaluation environment, and reset. Includes reward, completion, and available scalar features. |
| [`checkpoint_selections.csv`](checkpoint_selections.csv) | Checkpoint selected, evaluation reward, survival, and regret by method and training seed. |
| [`checkpoint_summary.csv`](checkpoint_summary.csv) | Mean reward, survival, and regret by selection method. |
| [`checkpoint_comparisons.csv`](checkpoint_comparisons.csv) | Paired differences relative to MG, win and tie counts, and sign-flip test results. |

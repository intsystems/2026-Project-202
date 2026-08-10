"""Print the participation-ratio-against-window-length table that is on the VM.

``pr_vs_window.py`` writes a CSV; a poll shows only that the file exists and how big it
is. This prints the table itself, so the result can be read without a download.
"""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
path = Path("/content/out/rank_fb_long/pr_vs_window.csv")
if not path.exists():
    raise SystemExit(f"{path} not written yet")

d = pd.read_csv(path)
cols = ["window_steps", "n_windows", "PR_pos_det_med", "PR_pos_det_min", "PR_pos_det_max",
        "PR_step_med", "fn_PR_pos_det_med", "fn_PR_step_med"]
extra = [c for c in ("PR_pos_det_at_gen", "fn_PR_pos_det_at_gen", "at_gen_centre")
         if c in d.columns]
for (run, ladder), g in d.groupby(["run", "ladder"]):
    print(f"\n=== {run} / {ladder}  (t_mem={g.t_mem.iloc[0]} t_gen={g.t_gen.iloc[0]}) ===")
    print(g[cols + extra].round(3).to_string(index=False))

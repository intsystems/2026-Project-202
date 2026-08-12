"""Run ``../active_rank/analyze_rank.py`` on the VM, where the sketches already are.

Each sketch is ~53 MB; the ``rank_windows.csv`` derived from a whole set is under a
megabyte.  Downloading hundreds of megabytes to compute a table the VM can compute in
seconds is the wrong way round, so the analysis runs here and only its output travels.

Window 60 rows at ``log_every = 10`` is a 600-step window -- the resolution
``active_rank``'s ``results_fine/`` used, and the one its reported numbers come from.

Every ``/content/out/rank*`` directory is processed, so the full-batch and mini-batch
arms are handled by one call and cannot end up analysed with different settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/content/edm/active_rank")
import analyze_rank  # noqa: E402

dirs = sorted(d for d in Path("/content/out").glob("rank*") if d.is_dir())
if not dirs:
    raise SystemExit("no /content/out/rank* directories")

for d in dirs:
    if not list(d.glob("*_rank.npz")):
        print(f"\n### {d.name}: no sketches, skipping", flush=True)
        continue
    print(f"\n{'#' * 70}\n### {d.name}\n{'#' * 70}", flush=True)
    analyze_rank.main(["--indir", str(d), "--outdir", str(d),
                       "--window", "60", "--stride", "5"])

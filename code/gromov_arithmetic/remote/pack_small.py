"""Tar everything under /content/out except the trajectory sketches.

The sketches are ~53 MB each and there are eight of them; every statistic anyone wants
from them is already in the ``rank_windows.csv`` the VM computed, and re-deriving one
costs a re-run rather than a lost result. So they stay on the VM and the archive holds
only what is small enough to keep in the repo.
"""

import os
import tarfile

OUT = "/content/out"
ARCHIVE = "/content/outputs_small.tar.gz"
SKIP = (".npz",)

if not os.path.isdir(OUT):
    raise SystemExit(f"no output directory at {OUT}")

total = 0
with tarfile.open(ARCHIVE, "w:gz") as tar:
    for root, _dirs, files in os.walk(OUT):
        for name in sorted(files):
            if name.endswith(SKIP):
                continue
            path = os.path.join(root, name)
            arc = os.path.relpath(path, OUT)
            tar.add(path, arcname=arc)
            size = os.path.getsize(path)
            total += size
            print(f"  {arc:<48} {size:>12,}")

print(f"\npacked {total:,} bytes -> {ARCHIVE} "
      f"({os.path.getsize(ARCHIVE):,} compressed)")

"""Tar the run outputs into one file. Invoked by ``colab_job.ps1 -Fetch``.

One archive beats N ``colab download`` calls: the output file names are derived from
each run's ``RunConfig.csv_name``, so the local side would otherwise have to know them.
"""

import os
import tarfile

OUT = "/content/out"
ARCHIVE = "/content/outputs.tar.gz"

if not os.path.isdir(OUT):
    raise SystemExit(f"no output directory at {OUT} -- has anything run yet?")

with tarfile.open(ARCHIVE, "w:gz") as tar:
    for name in sorted(os.listdir(OUT)):
        path = os.path.join(OUT, name)
        tar.add(path, arcname=name)
        print(f"  {name:<55} {os.path.getsize(path):>12,} bytes")

print(f"\npacked -> {ARCHIVE} ({os.path.getsize(ARCHIVE):,} bytes)")

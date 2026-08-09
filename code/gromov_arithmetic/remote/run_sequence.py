"""Run a list of argument vectors in order, on the VM, under one detached process.

``launch_detached.py`` starts exactly one process, but a campaign is several commands
(a sweep, then the arithmetic runs, then the polynomial runs).  Holding the exec
channel open for each in turn is what loses results when Colab reclaims the VM, so
the whole sequence goes into one detached process and the poller downloads finished
files as they appear.

Reads ``/content/job_seq.json``: a list of argv lists, each relative to /content/edm.
Keeps going after a failure and reports the exit codes at the end -- one bad config
should not cost the runs queued behind it.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SEQ = Path("/content/job_seq.json")
if not SEQ.exists():
    raise SystemExit(f"{SEQ} not found -- upload the sequence first")

jobs = json.loads(SEQ.read_text())
print(f"sequence of {len(jobs)} job(s)", flush=True)

results = []
for i, argv in enumerate(jobs, 1):
    print(f"\n{'=' * 70}\n[{i}/{len(jobs)}] {' '.join(argv)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd="/content/edm")
    results.append((argv[0], proc.returncode, round(time.time() - t0, 1)))
    print(f"[{i}/{len(jobs)}] exit={proc.returncode} in {results[-1][2]}s", flush=True)

print(f"\n{'=' * 70}\nSEQUENCE COMPLETE", flush=True)
for script, code, secs in results:
    print(f"  {code:>3}  {secs:>8.1f}s  {script}", flush=True)

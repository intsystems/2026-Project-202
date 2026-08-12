"""Cheap status poll: job liveness, finished files, and a usable slice of the log.

The shared ``prediction_improved/remote/list_outputs.py`` tails three lines, which is
enough to see that a single run is alive.  A sweep prints one summary line per rate
and a sequence prints one per job, so three lines regularly hides the very numbers
the poll exists to read.  Everything else is identical, and it stays fast enough to
call repeatedly while training runs detached.
"""

import os
from pathlib import Path

OUT = Path("/content/out")
LOG = Path("/content/job.log")
PIDFILE = Path("/content/job.pid")
TAIL = int(os.environ.get("POLL_TAIL", "25"))

alive = False
if PIDFILE.exists():
    pid = PIDFILE.read_text().strip()
    alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
print(f"JOB_ALIVE {alive}")

total = 0
if OUT.is_dir():
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            size = os.path.getsize(path)
            total += size
            print(f"FILE {path.relative_to(OUT).as_posix()} {size}")
print(f"TOTAL_BYTES {total}")

if LOG.exists():
    for line in LOG.read_text(errors="replace").splitlines()[-TAIL:]:
        print(f"| {line}")

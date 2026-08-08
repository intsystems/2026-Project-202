"""List finished output files and the tail of the job log. Pairs with launch_detached.py.

Kept deliberately cheap: a poller calls this every few minutes while training runs
detached, and each call must return in well under a second so it does not itself occupy
the exec channel.
"""

import os
from pathlib import Path

OUT = Path("/content/out")
LOG = Path("/content/job.log")
PIDFILE = Path("/content/job.pid")

alive = False
if PIDFILE.exists():
    pid = PIDFILE.read_text().strip()
    alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
print(f"JOB_ALIVE {alive}")

if OUT.is_dir():
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"FILE {path.relative_to(OUT).as_posix()} {os.path.getsize(path)}")

if LOG.exists():
    tail = LOG.read_text(errors="replace").splitlines()[-3:]
    for line in tail:
        print(f"LOG {line}")

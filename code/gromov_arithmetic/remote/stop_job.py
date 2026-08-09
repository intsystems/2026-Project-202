"""Terminate the detached sequence started by ``launch_detached.py``.

Needed when a queued job turns out to be misconfigured: the sequence keeps going after
a failure by design, so a bad entry would otherwise hold the GPU for its full budget
while the useful runs behind it wait.  Kills the process group, since
``run_sequence.py`` spawns the actual trainer as a child.
"""

import os
import signal
from pathlib import Path

PIDFILE = Path("/content/job.pid")
if not PIDFILE.exists():
    raise SystemExit("no pid file -- nothing to stop")

pid = PIDFILE.read_text().strip()
if not pid.isdigit():
    PIDFILE.unlink()
    raise SystemExit(f"pid file held '{pid}', removed")

pid = int(pid)
try:
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    print(f"sent SIGTERM to process group of {pid}")
except ProcessLookupError:
    print(f"pid {pid} already gone")
PIDFILE.unlink()
print("pid file removed; a new sequence can be launched")

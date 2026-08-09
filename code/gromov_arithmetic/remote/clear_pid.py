"""Drop a stale ``/content/job.pid`` so a new sequence is not mistaken for a running one.

``launch_detached.py`` already ignores a pid file whose process is gone, but it matches
on ``argv[0]`` -- and every sequence here has the same ``argv[0]``
(``run_sequence.py``), so a recycled pid could look alive.  Clearing the file before a
deliberate relaunch removes the ambiguity entirely.
"""

from pathlib import Path

pidfile = Path("/content/job.pid")
if pidfile.exists():
    print(f"clearing pid file ({pidfile.read_text().strip()})")
    pidfile.unlink()
else:
    print("no pid file")

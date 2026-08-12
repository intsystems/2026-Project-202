"""Run ``pr_vs_window.py`` on whatever sketches are already on the VM.

The sequence writes one ``*_rank.npz`` per run and only analyses at the end, so the
first run's answer would otherwise wait for the second run's training. The analysis is
CPU work and the training is GPU work, so it can be read early at almost no cost. The
sequence's own final call reprocesses everything and overwrites this output.
"""

import subprocess
import sys

sys.exit(subprocess.run(
    [sys.executable, "-u", "gromov_arithmetic/pr_vs_window.py",
     "--indir", "/content/out/rank_fb_long", "--check"],
    cwd="/content/edm").returncode)

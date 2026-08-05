# prediction_improved

An early-warning signal for grokking read off the model's **function**, not its weights.
The method and its rationale are in [`method.md`](method.md); this file is how to run it.

Training reuses [`../grokking_train/`](../grokking_train/) unchanged -- those configs
produced the published logs. The only addition is [`probe.py`](probe.py), an observer
that records normalized probe-logit projections. It is verified not to perturb training:
[`verify_noninvasive.py`](verify_noninvasive.py) trains the same config with and without
it and requires the logs to be **bit-identical**. That check matters because
`grok/tasks.py` seeds one global torch RNG stream that the train/val split, the weight
initialisation and the mini-batch order all continue -- a single stray draw changes the
initial weights and can destroy grokking.

Everything runs on Google Colab from a Windows terminal via the
[Colab CLI](https://github.com/googlecolab/google-colab-cli).

## One-time setup

The CLI needs **Python >= 3.12** and is officially **Linux/macOS only**. `uv` supplies
the interpreter; [`colab_shim/`](colab_shim/) supplies the Windows fixes. No admin rights
and no WSL required.

```powershell
# 1. uv (brings its own Python; installs to ~\.local\bin)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. the CLI. The version pin is required: google-colab-cli 0.6.0 declares
#    jupyter-kernel-client unpinned, and 1.0.0 renamed KernelClient ->
#    JupyterKernelClient, so every execution dies with an AttributeError.
uv tool install google-colab-cli --with "jupyter-kernel-client<1.0"

# 3. authenticate (opens a URL, asks for the code Google shows)
cd <repo>\code\prediction_improved
uv run --python 3.12 --with google-auth-oauthlib python colab_auth.py --start
uv run --python 3.12 --with google-auth-oauthlib python colab_auth.py --finish <CODE>

# 4. confirm
.\colab.ps1 sessions
```

Step 3 is split in two on purpose. The CLI's built-in flow needs one process to survive
from printing the URL, through the browser round-trip, to the disk write; if it dies in
between, the token is exchanged and then lost. Splitting it removes that failure mode.
Each Google account authenticates once; the token persists in
`~\.config\colab-cli\token.json`.

## Running experiments

```powershell
# create a T4 session, push the code, train both runs, pull results back, shut down
.\colab_job.ps1 -Sync -Run mod_wd1,mod_wd0 -Fetch .\results -Stop

# keep the session and iterate
.\colab_job.ps1 -Sync -Run s5_wd1
.\colab_job.ps1 -Fetch .\results

# a CPU session (free, no GPU quota) -- enough for verify_noninvasive.py
.\colab_job.ps1 -Session cpu -Gpu none -Sync

# raw CLI, any subcommand
.\colab.ps1 status
.\colab.ps1 stop -s gpu
```

`-Run` takes any key from [`../grokking_train/runs.py`](../grokking_train/runs.py)
(`mod_wd1`, `mod_wd0`, `s5_wd1`, `s5_wd0`, `full_batch`, ...). Config overrides pass
through: `-ExtraArgs "--set","max_steps=5000"`.

Each run writes three files into `--outdir`:

| file | contents |
| --- | --- |
| `<csv_name>.csv` | the standard training log, format unchanged |
| `<key>_probe.csv` | per-logged-step projections and velocities, both probe families |
| `<key>_probe_snapshots.npz` | periodic full normalized-logit matrices |

**Use a GPU.** Measured on `mod_wd1`: T4 97 steps/s (3.4 min for 20 000 steps) against
~14 steps/s on CPU (~24 min). The models are small enough to be launch-bound rather than
FLOP-bound, so float64 costs the T4 little. Probe overhead on GPU is negligible.

## Files

| path | role |
| --- | --- |
| [`method.md`](method.md) | the method: what it computes, why, what would falsify it |
| [`probe.py`](probe.py) | the observer: normalized probe-logit projections |
| [`run_probe.py`](run_probe.py) | train one registered run with the probe attached |
| [`verify_noninvasive.py`](verify_noninvasive.py) | proof the probe leaves training bit-identical |
| [`smoke_test.py`](smoke_test.py) | GPU/environment check for a fresh runtime |
| [`colab_job.ps1`](colab_job.ps1) | sync / run / fetch / stop |
| [`colab.ps1`](colab.ps1) | raw CLI passthrough, with the Windows shim on `PYTHONPATH` |
| [`colab_auth.py`](colab_auth.py) | two-step OAuth |
| [`colab_shim/`](colab_shim/) | the Windows fixes (below) |
| [`remote/`](remote/) | small scripts `colab_job.ps1` executes on the VM |

## What `colab_shim/` fixes

It goes on `PYTHONPATH` rather than patching site-packages, so `uv tool upgrade` cannot
silently revert it.

* **`termios.py`, `tty.py`** -- the only real obstacle to running the CLI on Windows is
  an unconditional `import termios` in `colab_cli/console.py`, imported at module load
  but used solely by the interactive `console`/`repl` TTY. Stubs make every other
  command work; the interactive TTY raises a clear error instead.
* **`sitecustomize.py`** -- two unrelated fixes, applied at interpreter startup:
  1. **Credential leak.** `setup_logging()` unconditionally sets the *root* logger to
     DEBUG with a file handler, and `requests_oauthlib` logs the whole token response --
     so every authentication wrote the access token, refresh token and id token in clear
     text to `~\.config\colab-cli\colab.log`. No flag disables it. The shim pins the
     loggers that emit token material to WARNING and attaches a redaction filter to
     every handler.
  2. **The stray console window.** `spawn_keep_alive` launches the VM keep-alive daemon
     with `DETACHED_PROCESS`; since `python.exe` is a console binary, Windows hands it a
     fresh console -- an empty black window for the life of the session.
     `CREATE_NO_WINDOW` is ignored alongside `DETACHED_PROCESS`, so the shim runs the
     daemon under `pythonw.exe` instead.

## Notes

* **Sessions cost quota.** Stop them when done (`-Stop`, or `.\colab.ps1 stop -s <name>`).
  `.\colab.ps1 sessions` lists what is still running.
* **The keep-alive daemon must stay running** while a job is in flight -- it stops Colab
  reclaiming an idle VM. `colab stop` tears it down.
* **`colab exec` defaults to a 30 s timeout** and aborts on *silence*, not on duration.
  Long jobs therefore have to emit output; `run_probe.py --progress-every` does.
* **The kernel is single-threaded.** A long-running `exec` blocks every later one on that
  session; `.\colab.ps1 restart-kernel -s <name>` clears a stuck one.

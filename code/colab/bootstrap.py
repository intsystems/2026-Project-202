"""Colab mode: set up a runtime, run a campaign, get the results back.

Nothing in `actdim` imports this file, and this file is the only place that knows Colab
exists. That separation is deliberate. The archived tree drove Colab from a PowerShell
wrapper around a third-party CLI, with a detached process, a single global job slot and a
pid file; it worked, and it also orphaned a live GPU twice when a network timeout made an
empty session listing look like an absent session. None of that is needed to run a
campaign: a notebook cell holding the process is enough, and a notebook that dies leaves
nothing behind but a stopped runtime.

Usage in a Colab cell:

    !git clone -q https://github.com/<user>/EDMGrokking.git
    %cd EDMGrokking/code
    !pip install -q -r requirements-train.txt
    from colab.bootstrap import setup, run, save
    setup()                       # check the GPU, pin threads, report the environment
    run("train.perceptron.eos")   # any actdim target
    save()                        # tar runs/ and download it, or copy it to Drive

The same package runs unchanged on a Linux GPU box, where none of this is needed:

    python -m actdim run train --device cuda:0
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

CODE_DIR = Path(__file__).resolve().parent.parent


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def setup(threads: Optional[int] = None, require_gpu: bool = True) -> dict:
    """Prepare the runtime and report what it is.

    Raises if a GPU was expected and none is attached, because the training campaigns
    are float64 and a Colab CPU runtime will take a day over what a T4 does in an hour.
    Pass ``require_gpu=False`` to run the analysis half, which needs no GPU at all.
    """
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)

    from actdim.runtime import device as device_mod
    from actdim.runtime.determinism import pin_blas_threads
    from actdim.runtime.provenance import library_versions

    pin_blas_threads(threads or 1)
    info = device_mod.describe("auto")
    info["libraries"] = library_versions()
    info["free_disk_gb"] = round(device_mod.free_disk_gb(str(CODE_DIR)), 1)
    info["in_colab"] = in_colab()

    print(f"device      {info['device']}")
    if "gpu" in info:
        print(f"gpu         {info['gpu']} ({info['gpu_memory_gb']} GB, CUDA {info.get('cuda')})")
    print(f"torch       {info.get('torch')}")
    print(f"free disk   {info['free_disk_gb']} GB")

    if require_gpu and info["device"] == "cpu":
        raise SystemExit(
            "No GPU is attached. In Colab: Runtime > Change runtime type > T4 GPU.\n"
            "To run the analysis half instead, call setup(require_gpu=False)."
        )
    return info


def run(*targets: str, device: str = "auto", jobs: int = 0, extra: Sequence[str] = ()) -> int:
    """Run actdim targets, streaming output into the notebook.

    Runs in the foreground on purpose. A detached job needs a pid file, a poller and a
    lifecycle, and gains nothing here: the cell holds the process, and closing the
    notebook stops the runtime either way.
    """
    if not targets:
        raise ValueError("name at least one target, e.g. run('train.perceptron.eos')")
    command = [sys.executable, "-u", "-m", "actdim", "run", *targets,
               "--device", device, "--jobs", str(jobs), *extra]
    print("$ " + " ".join(command[2:]))
    started = time.time()
    result = subprocess.run(command, cwd=str(CODE_DIR))
    print(f"\nexit {result.returncode} after {(time.time() - started) / 60:.1f} min")
    return result.returncode


def plan(*targets: str) -> None:
    """What a campaign would run, and what it would cost, before starting it."""
    subprocess.run([sys.executable, "-m", "actdim", "plan", *targets], cwd=str(CODE_DIR))


def save(archive: str = "actdim_runs.tar.gz", drive_dir: Optional[str] = None,
         only: Iterable[str] = ()) -> Path:
    """Pack ``runs/`` and hand it back.

    With ``drive_dir`` the archive is copied to a mounted Google Drive, which survives
    the runtime being recycled. Without it the archive is offered as a browser download.
    A campaign's raw output runs to hundreds of megabytes, so ``only`` restricts the
    archive to named run directories.
    """
    import tarfile

    runs = CODE_DIR / "runs"
    if not runs.exists():
        raise SystemExit("runs/ is empty -- nothing has been run in this session")

    members = sorted(p for p in runs.iterdir() if p.is_dir())
    if only:
        wanted = set(only)
        members = [p for p in members if p.name in wanted or
                   any(p.name.startswith(w.rstrip(".") + ".") for w in wanted)]
    if not members:
        raise SystemExit(f"no run directories matched {list(only)}")

    path = CODE_DIR / archive
    with tarfile.open(path, "w:gz") as tar:
        for member in members:
            tar.add(member, arcname=f"runs/{member.name}")
    size_mb = path.stat().st_size / 1024 ** 2
    print(f"{path.name}  {size_mb:.1f} MB  ({len(members)} run directories)")

    if drive_dir:
        import shutil

        target = Path(drive_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / path.name)
        print(f"copied to {target / path.name}")
        return target / path.name

    if in_colab():
        from google.colab import files  # type: ignore

        files.download(str(path))
    return path


def mount_drive(mountpoint: str = "/content/drive") -> str:
    """Mount Google Drive, for campaigns longer than a runtime's lifetime."""
    from google.colab import drive  # type: ignore

    drive.mount(mountpoint)
    return mountpoint


def unpack(archive: str = "actdim_runs.tar.gz", into: Optional[str] = None) -> None:
    """Unpack a fetched archive back into ``runs/`` on the machine that has the repo.

    Run this locally, not in Colab, after downloading the archive:

        python -c "import sys; sys.path.insert(0,'.'); \\
                   from colab.bootstrap import unpack; unpack('~/Downloads/actdim_runs.tar.gz')"
    """
    import tarfile

    source = Path(archive).expanduser()
    target = Path(into).expanduser() if into else CODE_DIR
    with tarfile.open(source, "r:gz") as tar:
        names = tar.getnames()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise SystemExit(f"refusing to unpack {source}: it contains absolute or parent paths")
        tar.extractall(target)
    print(f"unpacked {len(names)} entries into {target}")

"""Choosing a device, in one place.

The archived tree resolved ``"auto"`` in five separate functions and recorded the
unresolved string in one committed result file, so that measurement cannot say whether it
ran on a GPU. Resolution happens here, and the resolved value is what gets written down.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve(spec: str = "auto") -> str:
    """Turn ``auto`` into the device that will actually be used.

    Returns a concrete string: ``cpu``, ``cuda:0``, or ``mps``. Never returns ``auto``.
    """
    spec = (spec or "auto").strip().lower()
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe(spec: str = "auto") -> Dict[str, Any]:
    """The resolved device and what it is, for the provenance record."""
    device = resolve(spec)
    info: Dict[str, Any] = {"requested": spec, "device": device}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(index)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["cuda"] = torch.version.cuda
    return info


def require_gpu(spec: str = "auto") -> str:
    """Resolve a device for work that is not worth running on a CPU.

    The training campaigns are float64, which a CPU will run but slowly enough that a
    silent fallback wastes hours before anyone notices. Callers that can tolerate a CPU
    pass ``--device cpu`` explicitly.
    """
    device = resolve(spec)
    if device == "cpu":
        raise SystemExit(
            "This experiment trains a network and needs a GPU. No CUDA device was found.\n"
            "Run it on the GPU box or in Colab mode, or pass --device cpu to accept a\n"
            "much slower run (hours rather than minutes; the numbers are unaffected, but\n"
            "float64 CPU and CUDA results differ in their last bits)."
        )
    return device


def torch_device(spec: str = "auto") -> Any:
    """The resolved device as a ``torch.device``."""
    import torch

    return torch.device(resolve(spec))


def free_disk_gb(path: Optional[str] = None) -> float:
    """Free space where results are about to be written."""
    import shutil

    return shutil.disk_usage(path or ".").free / 1024 ** 3

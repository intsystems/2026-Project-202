"""Memory-light loader: the E4 CIFAR subset cached once as float32 tensors (~150 MB)."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cifar_subset.pt"


def load():
    if not CACHE.exists():
        sys.path.insert(0, str(HERE.parent / "research_trajectory_reference"))
        from cifar_events import load as full
        torch.save(tuple(full()), CACHE)
    return torch.load(CACHE)

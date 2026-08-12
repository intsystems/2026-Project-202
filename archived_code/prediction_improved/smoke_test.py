r"""Smoke test for the Colab runtime: is there a GPU, and is the environment usable?

Run remotely with:  .\colab.ps1 run --gpu T4 smoke_test.py
"""

import platform
import subprocess
import sys


def main():
    print("=== runtime ===")
    print("python  :", sys.version.split()[0], "on", platform.platform())

    try:
        import torch
    except ImportError:
        print("torch   : NOT INSTALLED")
        return 1

    print("torch   :", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("RESULT: no GPU visible -- provision with --gpu, e.g. `colab new --gpu T4`")
        return 1

    print("device  :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("memory  : %.1f GB" % (props.total_memory / 1024 ** 3))
    print("capability:", f"{props.major}.{props.minor}")

    # A real (if tiny) GPU computation, so this fails loudly if the driver is broken.
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print("matmul  : ok, checksum finite =", bool(c == c))

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        print("nvidia-smi:", out.stdout.strip() or out.stderr.strip())
    except Exception as exc:                                   # noqa: BLE001
        print("nvidia-smi: unavailable (%s)" % exc)

    print("RESULT: GPU OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unpack the uploaded code bundle on the VM.  Invoked by ``colab_rank.ps1 -Sync``.

PowerShell's ``Compress-Archive`` stores ``a\\b.py`` as the entry name, and a backslash is
a legal filename character on Linux -- so a plain ``extractall`` produces files literally
named ``active_rank\\run_rank.py`` and no importable package.  Normalising the separators
here keeps the Windows side free to use the built-in zip tooling.  Same job as
``../prediction_improved/remote/unpack.py``, kept separate only because the bundle name
differs.

This lives as a file rather than a here-string inside the .ps1 because the backslash
survives neither PowerShell's expansion nor a shell heredoc reliably.
"""

import os
import shutil
import sys
import zipfile

TARGET = "/content/edm"
BUNDLE = "/content/rank_bundle.zip"

shutil.rmtree(TARGET, ignore_errors=True)
with zipfile.ZipFile(BUNDLE) as archive:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.endswith("/"):
            continue
        destination = os.path.join(TARGET, name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with archive.open(info) as source, open(destination, "wb") as out:
            shutil.copyfileobj(source, out)

os.makedirs("/content/out", exist_ok=True)
count = sum(len(files) for _root, _dirs, files in os.walk(TARGET))
print(f"unpacked {count} files into {TARGET}")

import torch  # noqa: E402

accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | {accelerator}")

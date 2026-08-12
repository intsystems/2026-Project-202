"""Unpack the uploaded code bundle on the VM. Invoked by ``colab_job.ps1 -Sync``.

PowerShell's ``Compress-Archive`` stores ``a\\b.py`` as the entry name, and a backslash
is a legal filename character on Linux -- so a plain ``extractall`` produces files
literally named ``grokking_train\\grok\\loop.py`` and no package to import. Normalising
the separators here keeps the Windows side free to use the built-in zip tooling.
"""

import os
import shutil
import sys
import zipfile

TARGET = "/content/edm"

shutil.rmtree(TARGET, ignore_errors=True)
with zipfile.ZipFile("/content/edm_bundle.zip") as archive:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.endswith("/"):
            continue
        destination = os.path.join(TARGET, name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with archive.open(info) as source, open(destination, "wb") as out:
            shutil.copyfileobj(source, out)

count = sum(len(files) for _root, _dirs, files in os.walk(TARGET))
print(f"unpacked {count} files into {TARGET}")

try:
    import torch
    accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | {accelerator}")
except ImportError:
    print(f"python {sys.version.split()[0]} | torch NOT INSTALLED")

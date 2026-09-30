from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_llm_eos_reference_colab.zip"

required = [
    ROOT / "research_llm_eos" / "run_v2.py",
    ROOT / "research_llm_eos" / "transformer_mg_benchmark.py",
    ROOT / "research_llm_eos" / "analyze_v2.py",
    ROOT / "research_llm_eos" / "reference_spectrum.py",
    ROOT / "research_llm_eos" / "PROTOCOL.md",
    ROOT / "research_llm_eos" / "data" / "tinyshakespeare.txt",
]

missing = [path for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))

if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in required:
        archive.write(path, path.relative_to(ROOT).as_posix())

    package_root = ROOT / "code" / "actdim"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Missing package directory: {package_root}")
    for path in package_root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.write(path, path.relative_to(ROOT).as_posix())

    archive.writestr(
        "requirements_colab.txt",
        "numpy\npandas\nscipy\nscikit-learn\nmatplotlib\n",
    )

with zipfile.ZipFile(OUT) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt member in generated archive: {bad}")
    print(f"Created: {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"Files: {len(archive.namelist())}")

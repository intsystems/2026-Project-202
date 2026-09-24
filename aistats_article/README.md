# AISTATS 2027 edition

This folder is generated from `../icomp_v2/report.tex` by `make_aistats.py`. The official AISTATS 2027 style files are kept unchanged in `style/`.

Build it directly:

```powershell
python make_aistats.py
python make_aistats.py --claude blue
```

Or rebuild every edition and both review modes at once:

```powershell
python ../build_all.py
```

Outputs are `aistats2027.pdf`, `aistats2027_blue.pdf`, and the generated `aistats2027*.tex` files. Edit the shared ICOMP source and rerun `build_all.py` to synchronize all editions.

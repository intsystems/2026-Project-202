# Article versioning

The current reference snapshot is `artifacts_article/reference/icomp_artifacts.pdf`, copied from the requested latest Artifacts PDF. Its scientific content is represented in `icomp_v2/report.tex`; ICOMP, the Neural Network Artifacts edition, and AISTATS 2027 are generated from that synchronized source.

Run this from `2026-Project-202`:

```powershell
python build_all.py
```

The script also supports selective rebuilds:

```powershell
python build_all.py --mode blue
python build_all.py --edition icomp
python build_all.py --edition aistats --mode blue
```

Each build runs in a fresh LaTeX work directory, writes a log to `build_logs/`,
and fails if the expected PDF was not regenerated during the current run.

The command builds both modes for every edition:

| Edition | Ordinary PDF | Blue review PDF |
| --- | --- | --- |
| ICOMP | `icomp_v2/report.pdf` | `icomp_v2/report_blue.pdf` |
| Artifacts | `artifacts_article/icomp_artifacts.pdf` | `artifacts_article/icomp_artifacts_blue.pdf` |
| AISTATS 2027 | `aistats_article/aistats2027.pdf` | `aistats_article/aistats2027_blue.pdf` |

The blue mode defines `\\claudedraft`. Only text explicitly wrapped in `\\cl{...}` or `\\begin{claude}...\\end{claude}` is blue; the current article has no such text, so the ordinary and blue PDFs currently look the same.

`article_versions.json` records the source hash and the paths produced by the last complete build. The AISTATS style files are kept in `aistats_article/style/` and are the official 2027 paper pack.

# venue_common

`neurips2026.py` converts `icomp_v2/report.tex` from its ICOMP dress to the
NeurIPS 2026 workshop template. It is not run directly. Two venue folders
supply a `VENUE` dict and call its `main()`:

| | |
| --- | --- |
| [`newinml_article/make_newinml.py`](../newinml_article/) | NewInML, 2–8 pages |
| [`artifacts_article/make_artifacts.py`](../artifacts_article/) | Neural Network Artifacts, 8–12 pages |

Both workshops want the same paper on the same kit — `\documentclass{article}`
plus `\usepackage[dblblindworkshop]{neurips_2026}` — and differ in the name in
the first-page footer, the page limit, and nothing else that reaches the LaTeX.
So the conversion is here once and each folder holds only what is genuinely its
own: the venue's page limits, its copy of the style file, its author block, its
README.

The alternative was a copy of this file in each folder, which would have been
two copies to fix every time the paper changed shape. The paper is under review
and will change shape.

## What a venue profile has to supply

```python
VENUE = {
    "workshop":      "NewInML",              # for the banner in the generated .tex
    "workshoptitle": "NewInML",              # what \workshoptitle prints
    "slug":          "icomp_newinml",        # names the PDF and the zip
    "authors":       "newinml_authors.tex",  # read under --mode final
    "submit":        "https://openreview.net/...",
    "tracks": {                              # one entry per length the call takes
        "workshop": {
            "name":    "workshop paper, double-blind",
            "option":  "dblblindworkshop",   # the neurips_2026 package option
            "why":     "...",                # a comment written into the .tex
            "min": 2, "max": 8,              # the page limit, references excluded
            "default": True,
        },
    },
    "notes": ["printed after every build"],
}
```

With one track the venue script takes no `--track`; with more, the track names
the build directory, the PDF and the zip, so several lengths can sit side by
side.

## What it takes as input

One shape: a standalone LaTeX article, `\documentclass` through
`\end{document}`, with a `\bibliography` in it. `report.tex` is already that,
which is why nothing has to be assembled first. `--source-main` points at
another copy of it; assets are looked for beside it.

The page limit is measured, not guessed: a `\label` goes in just before
`\bibliography`, and `main.aux` then records the page the main text ends on.
Both calls exclude references, and this paper's appendix follows them.

## What it will not do

Write to `icomp_v2/`, and cut pages. Over-length is reported and left alone —
the cut belongs in `report.tex`, where both editions pick it up.

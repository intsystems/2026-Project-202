# Neural Network Artifacts as a New Data Modality @ NeurIPS 2026 edition

`make_artifacts.py` puts `icomp_v2/report.tex` on the NeurIPS 2026 workshop
template. It reads the article where it lies and writes nothing back to it.

```sh
cd artifacts_article
python make_artifacts.py                # full paper (default)
python make_artifacts.py --mode final   # camera-ready, author names shown
```

Each run writes `build/full/`, a compiled `icomp_artifacts.pdf`, and an
`icomp_artifacts.zip` that compiles on its own in an empty directory.

**Edit `report.tex`, rerun that one line, and this edition follows.**

## The venue

| | |
| --- | --- |
| Call | <https://artifactsasdata.org/cfp/> |
| Template | `\documentclass{article}` + `\usepackage[dblblindworkshop]{neurips_2026}` |
| Length | 8–12 pages, excluding references and supplementary material |
| Submission | OpenReview, <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/NeuralArtifacts> |
| Deadline | 1 September 2026, AoE |
| Self-contained | required: "reviewers will not be required to consult supplementary material" |

The call also takes extended abstracts of 4–6 pages. This paper goes as a full
paper, so that is the only track built here and the script takes no `--track`:
a track nobody builds is a page limit nobody checks. Adding it back is a
five-line entry in `VENUE["tracks"]`, and the build directory, the PDF and the
zip pick up the track name automatically as soon as there is more than one.

`style/` holds `neurips_2026.sty` as this workshop ships it, from
[`neurips_2026_artifacts_as_data.zip`](https://artifactsasdata.org/assets/neurips_2026_artifacts_as_data.zip)
on artifactsasdata.org, with the template and checklist beside it and the whole
zip for provenance. It is the standard NeurIPS kit one revision behind the copy
in [`../newinml_article/style/`](../newinml_article/style/): it lacks the
`education` track option, which nothing here uses. Each folder carries the file
its own venue hands out, which is why the two are not shared.

## What the call has not settled

The call says the "official template, policies, anonymity requirements, and
track-selection instructions will be linked here when available." Until they
are, the build assumes double-blind — the call names anonymity requirements as
something that will apply, and ships the standard NeurIPS workshop kit. If it
turns out to be single-blind, the fix is one word in `make_artifacts.py`:
`sglblindworkshop`.

## Where the conversion lives

In [`venue_common/neurips2026.py`](../venue_common/neurips2026.py), shared with
[`newinml_article/`](../newinml_article/). Both workshops want the same paper on
the same NeurIPS 2026 kit and differ only in the name in the footer and the
page limit, so this file is the venue and that one is the conversion. What the
conversion does, and the two link defects it repairs, are documented in
[`../newinml_article/README.md`](../newinml_article/README.md#what-the-conversion-does);
they are the same here.

## The author block

`artifacts_authors.tex`, read only under `--mode final`. Kept separate from
`newinml_article/newinml_authors.tex` rather than shared: the two are the same
list today, and an author added for one venue and not the other is exactly the
divergence a shared file would hide. It carries TODOs and the build lists them
until they are filled in.

## Checks

The same set the NewInML edition runs, documented there. The build fails rather
than ship a broken document, and after compiling reports undefined references,
undefined citations, duplicate destinations, overfull boxes, and the length of
the main text against the limit.

Where it stands today:

```
33 pages, main text 9, 0 undefined, 0 duplicate, 0 overfull   <-- fits
```

Nine pages of main text against 8–12, so this edition needs no cut. A cut made
in `report.tex` for another venue will shorten it, and the build will say so;
below eight pages it would report the paper as short.

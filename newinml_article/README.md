# NewInML @ NeurIPS 2026 edition

`make_newinml.py` puts `icomp_v2/report.tex` on the NeurIPS 2026 workshop
template. It reads the article where it lies and writes nothing back to it.

```sh
cd newinml_article
python make_newinml.py                 # anonymous submission (default)
python make_newinml.py --mode final    # camera-ready, author names shown
```

Each run writes `build/workshop/`, a compiled `icomp_newinml.pdf`, and an
`icomp_newinml.zip` that compiles on its own in an empty directory.

**Edit `report.tex`, rerun that one line, and this edition follows.** That is
the whole point of it: reviews, camera-ready and a page cut all happen in
`report.tex`.

## The venue

| | |
| --- | --- |
| Call | <https://newinml.github.io/NewInML2026NeurIPS/> |
| Template | `\documentclass{article}` + `\usepackage[dblblindworkshop]{neurips_2026}` |
| Length | 2 to 8 pages excluding references |
| Review | double-blind on OpenReview, full anonymization |
| Deadline | 29 August 2026, 11:59pm AoE |
| Archival | non-archival; concurrent submission elsewhere is allowed |
| Eligibility | authors who have not yet published at NeurIPS, ICML or ICLR |

That last row is a condition on the authors, not the paper, and no build can
check it. The script prints it every run.

`style/` holds the official `neurips_2026.sty` from
[`Formatting_Instructions_For_NeurIPS_2026.zip`](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
on media.neurips.cc, with the template and checklist it ships beside it. The
whole zip is kept for provenance.

## Where the conversion lives

In [`venue_common/neurips2026.py`](../venue_common/neurips2026.py), shared with
[`artifacts_article/`](../artifacts_article/). Both workshops want the same
paper on the same NeurIPS 2026 kit and differ only in the name in the footer
and the page limit, so this file is the venue and that one is the conversion.

## What the conversion does

| | |
| --- | --- |
| class line | `\documentclass{article}` + `\usepackage[dblblindworkshop]{neurips_2026}` + `\workshoptitle` |
| `\usepackage{icomp2026_conference,times}` | dropped. The first is the old venue's style file; the second neurips_2026 loads itself, and loading it twice risks an option clash. One line, so package lists are rewritten member by member rather than line by line |
| `\input{math_commands.tex}` | inlined, so the zip compiles with only the style file beside it |
| `\bibliographystyle{icomp2026_conference}` | `plainnat`, that `.bst` having stayed behind |
| the title's three `\\` | removed. They are in `report.tex` because the ICOMP style file sets a title in small caps at about forty characters to the line and hyphenates it otherwise. NeurIPS centres a title of its own width, where the same breaks land in arbitrary places |
| the author block | withheld under `anon` — the style file prints "Anonymous Author(s)" itself — and under `final` taken from `newinml_authors.tex` |
| `float` | lifted above `hyperref` (see below) |
| float and theorem destinations | made unique (see below) |

## Two link defects it repairs

Both are invisible at ICOMP and appear when the paper moves.

**`float` after `hyperref`.** Both patch `\@caption`, and in that order every
float's destination is written twice and the second is dropped. `report.tex`
loads `hyperref` on line 13 and `float` on line 21, so **as it stands it has 36
of these**, one per table and figure, and every `\cref` to a float lands on the
wrong one. Its own `report.log` says so 36 times; the converted build says it 0
times. Loading `float` first fixes all of them and changes nothing visible.
That is worth carrying back into `report.tex` itself.

**Duplicate destinations.** hyperref names a destination after the raw counter
value, so two objects sharing a value share a destination. The appendix
restarts float numbering, so appendix Table 1 collides with body Table 1. The
conversion qualifies `\theH<counter>` with the section, which changes nothing
visible.

## Modes

Under `anon` the style file prints "Anonymous Author(s)" and numbers the lines
for reviewers, and the footer reads "Submitted to 40th Conference on Neural
Information Processing Systems (NeurIPS 2026). Do not distribute." That is the
style file's own behaviour for a submission: `\workshoptitle` reaches the
footer only under `final`, where it reads "…(NeurIPS 2026). Workshop:
NewInML."

`--workshop-title "Some Other Workshop"` retargets it, which is the only change
another double-blind NeurIPS workshop needs.

## The author block

`newinml_authors.tex`, read only under `--mode final`. `report.tex` carries
`\author{Anonymous}`, right for its own double-blind submission and wrong for a
camera-ready anywhere, so the names are written here rather than inherited.
NeurIPS separates authors with `\And` or `\AND` and takes the affiliation as
further lines of the same block, which is not how ICOMP spells it. It carries
TODOs today and the build lists them until they are filled in.

## Checks

The build fails rather than ship a broken document: one `\documentclass`, one
balanced `document`, one `\maketitle`, one `\bibliographystyle`, one
`\workshoptitle`, no surviving ICOMP style file or `\usepackage{times}`, no
`\author` in an anonymous build and exactly one in a camera-ready, and no label
referenced but defined nowhere.

That camera-ready author count is not a formality. The authors file is itself
an `\author{...}`, so a substitution that rescans from the top finds the block
it has just inserted, counts it as a second one and deletes it — a
camera-ready with no names on it, which compiles clean and is wrong only in the
rendered PDF. The count catches it.

After compiling it reports undefined references, undefined citations, duplicate
destinations, overfull boxes, and the length of the main text against the
limit.

Where it stands today:

```
34 pages, main text 8, 0 undefined, 0 duplicate, 0 overfull   <-- fits
```

Eight pages of main text against a limit of eight, so the body is at the
ceiling and has no slack. Anything added to it needs a matching cut, and the
cut belongs in `report.tex`, where both editions of the paper pick it up.

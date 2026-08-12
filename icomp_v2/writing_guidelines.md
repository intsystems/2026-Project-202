# Writing guidelines for `report.tex`

Distilled from seven rounds of review. Read before editing prose. The faults below are ones
I actually made and was corrected on, in most cases more than once.

## 1. Sentence construction

- **One claim per sentence, subject first, verb early.** Compressed periodic sentences read as
  obscure, not dense.
- **Never delete a head noun.** Not "is a point cloud's, not a trajectory's", "the level is the
  embedding's", "as the direct measurement's". Write the noun out.
- **No possessive on an inanimate name holding a number.** Not "modular addition's 1.000", "that
  task's own 0.062". Recast: "the 1.000 that modular addition sets".
- **No pseudo-clefts.** Not "What the experiment shows is that…", "what changes is…". Subject first.
- **`, and` is not a way to bolt a second finding onto a finished sentence.** Ask what relation the
  two clauses have and use it: subordination, `so`, `while`, a semicolon, or a full stop. Target
  under ~8 clause-joining uses per thousand words.
- **`where` is not `whereas`.** Keep `where` only for genuine locatives ("at zero learning rate,
  where the parameters do not move").
- **No sentence-initial `And`/`But`** to append a last observation. Make it its own sentence, with
  `also`/`finally` if the sequence matters.
- **No fragments for emphasis**, no absolute constructions ("with nothing dimensional occurring"),
  no dangling participles ("Read at the stride…, the phenomenon is erased").
- **Order premises before conclusions.** A → B → C, not C followed by unexplained A and B.
- **Intensifiers need ground.** "rejects one observer outright" is empty unless the sentence says
  what made it outright.

## 2. Word choice

- **Name a thing when it first appears, then refer to it descriptively.** Never a bare jargon plural
  ("the controls") for a set the reader has to reconstruct.
- **No coined metaphor.** Replaced: instrument → estimator; hazard → the specific failure; pay for
  the trajectory → store it; the diagnostics refuse → place outside the admissible regime; the
  ladder → the constructed systems; the figure carries one hue → drawn in one colour; each
  hypothesis's knob → the variable it names; a want of resolution → the limitation is resolution.
- **Banned as informal:** sits, arm, cheap, flag, free, outright, landing near, cut up, gains
  correctness, pushes through.
- **Watch for a word doing two jobs** in one section (`clean`, `control`, `cost`) and for one verb
  carrying everything (`carries`, `read`, `falls`, `count`).
- **`read`** is legitimate only in the interpretive sense the paper turns on (whether a number *may
  be read as* a dimension). Elsewhere use *used*, *compared*, *concluded*, *tracking*.
- **Em dashes: budget about 1.5 per thousand words.** Most become commas or a colon.
- **Headings are descriptive, not journalistic.** No "Two cautions, one of them serious". If a
  paragraph does not need a name, do not give it one.

## 3. Where information lives

- **Measured values live in tables and figures.** The body tells the story and leaves the arithmetic
  to the reader. Every number removed from prose keeps a `\cref` to the float that carries it.
- **Walk the reader through floats.** "Table 5 collects the outcome", not a bare `(\cref{tab:ladder})`
  appended to a finished claim. Some parenthetical refs are fine; all of them is not.
- **Captions**: a short title naming the object or the question, then only what is needed to parse
  the panel. Target ≤ 60 words for a figure. Longer only for tables whose columns need defining.
  The argument the figure supports, the system description, and the reason a panel looks odd all
  belong in the text — check there before writing them into a caption.

## 4. What to cut

- A clause that restates the clause before it ("…and is unremarkable in the controls" right after
  "…while the controls show no comparable movement").
- Textbook glosses (what box-counting dimension means).
- Facts already stated in another section — check the introduction before writing a mechanism into
  §3, and check the text before writing it into a caption.
- Trailing "which is why…" when the why was just given.

Cutting improves the paper more than rewriting does. Prefer it.

## 5. Mechanics of this paper

- **Nine pages excluding references.** The body must end on page 9. Verify, never assume.
- **The last page break is float-driven.** Cutting text *before* Figure 2 can move the float and make
  the spill worse. Only cuts *after* it reliably help.
- **`\parskip` is 6pt = half a line.** Cuts spread thinly across paragraphs only shorten last lines
  and drop nothing. To lose a line, concentrate ~16 words in one paragraph.
- **Captions are the cheapest space.** Shortening a caption on page *n* pulls text up from page
  *n+1*.

## 6. How to edit

- Exact-string replacement scripts, one pass per theme, with `assert text.count(old) == 1`.
  Regex on LaTeX is not worth the risk.
- **Python string literals must be raw (`r"""…"""`).** A non-raw `"\rho … \approx"` once wrote a
  literal newline and a `0x07` byte into the file, which rendered as `ho_ident pprox 1` and survived
  several commits.
- Rebuild and check after every pass: `Overfull`, `LaTeX Warning`, `undefined` all zero, and the
  page-9 test below.
- **Re-run the scans after a pass, not just before.** Every large pass so far has introduced at
  least one new instance of a fault it was fixing.

## 7. Measurements

```bash
# fault census; -v prints the offending context
python style_scan.py            # or: python style_scan.py report.tex -v

# body must end on page 9: expect 1 (the running header alone)
pdftotext -layout -f 10 -l 10 report.pdf - | awk '/REFERENCES/{exit} NF{n++} END{print n+0}'

# build health: all three must be zero
grep -c "Overfull\|LaTeX Warning\|undefined" report.log
```

Baseline at the end of this round, from `style_scan.py`. Treat these as ceilings, not targets:

| | body | appendix |
|---|---|---|
| words / sentences | 5681 / 220 | 12275 / 454 |
| `, and` clause-joins | 17 | 27 |
| em dashes | 5 | 14 |
| `read` | 11 | 16 |
| inanimate `'s` | 14 | 16 |

Captions: 41, 3171 words, mean 77. The eight worst are still 110–162; they are the next thing to
cut if space is needed.

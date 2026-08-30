# Writing guidelines for `report.tex`

Distilled from eleven rounds of review. Read before editing prose.

## Read it. Do not only scan it.

`style_scan.py` finds instances of faults I have already been corrected on. It cannot find the
fault nobody has named yet, and it cannot tell a sentence that parses from a sentence that
communicates. **Every round in which I leaned on the scanner, the reply came back with a page of
things it had no pattern for**: an unattributed "the available dimension is $k$"; a "textbook
definition" of nothing in particular; three statistics reported with no stated purpose; a caption
whose words did not match the figure's own legend; a claim about label-matched pairs that was
simply false, because the two members of a pair are different polynomials.

So read the rendered PDF sentence by sentence and ask of each what it is doing. Run the scanner
afterwards, to catch the regressions the reading introduced. It is the second step, never the
first.

## The one instruction that matters

**Cut.** Not "merge the sentences", not "find a better connective" — *delete the content*. The
user's diagnosis: *you try to say too many things; most of them could be left out without any loss
of meaning*. Almost every complaint in the last three rounds is downstream of that. A clause I add
to be careful, to hedge, to gloss a term, or to signpost what comes next is usually a clause the
reader has to carry for nothing.

Before keeping any clause, ask: **would the paragraph be worse without it?** If the answer is "no,
but it's true", cut it.

### Where the cuts actually are

Four appendices were read in parallel and 4 000 words came out without a claim being lost. Almost
none of it was fat inside sentences. It was the same fact stated in three places. **Give every fact
one home**, and know which:

- the **caption** owns what a reader meets at the float — how a column is defined, what the marker
  is, what the shaded band means;
- the **appendix outline** owns the overview of what each appendix does;
- the **prose** owns the argument, and nothing else.

From that rule alone: *every appendix opener went*, because each was the outline entry rewritten;
*every "Three things follow"* went, because the three italic heads follow; every caption sentence
that restated the paragraph beside it went, and every paragraph sentence that restated the caption.

Then three structural sweeps, each mechanical and each worth a quarter-page or more:

- **A float nobody `\cref`s is a float nobody needs.** Five went this round — `tab:classes`,
  `tab:aniso`, `fig:observers`, `fig:prwindow`, `fig:ceiling` — every one of them plotting or
  tabulating a float that *is* referenced. Check every `\label` for a matching `\cref` before
  anything else; it is the cheapest page in the paper.
- **A symbol defined and never reported is dead.** `\PR^{\mathrm{upd}}` was defined in the body,
  defined again in an appendix, and appeared in no table or figure. Grep each symbol you introduce
  and count its *uses*, not its definitions.
- **A number in prose must match the table beside it.** "Plain mini-batch descent stays at $6.2$"
  stood two lines above a row reading 6.56, 6.58, 6.60. Read every numeral against its own table.
  While you are there: prose that promises a column ("`tab:matched` reports the detrended cells")
  must be checked against the table, which had no such column.

**Do this arithmetic explicitly, every round.** A separate pass that recomputed each prose numeral
from its own table found *fourteen* disagreements, most of them years old and all of them invisible
to reading:

| claimed | the table says |
|---|---|
| "LB is more accurate at both ranges" | MG 0.382 against LB 0.40 at $r \le 8$; true only at $r \le 20$ |
| "$256$ and $1024$ bands order the ranks more faithfully" | 1024 bands gives $\rho = 0.979$ against MG's $0.997$ |
| "exceeds the plateau by an order of magnitude in every run" | one run recovers by a factor of 1.7 |
| "exceeds it in three runs of the four" | two of the four |
| "the same proportional increase in $E_{\max}$" (against 64× in the record) | the sweep spans 5.6× |
| "three times the $1.53$-component floor" | $3.54 / 1.53 = 2.3$ |
| "at $8 \to 3 \to 8$ the high level is under-reported by a component" | 8.32 — over-reported, and by 0.32 |
| "recovers to within $0.01$" | 0.00, 0.05, 0.04 |
| "the recurrent cases are unchanged" | the slow recurrent case runs 2.06 to 3.27 |
| "$\rho_{\mathrm{ident}}$ near unity" (perceptron) | 1.39, 1.42, 1.51 |
| "$E_{\max}/2$ is right at $E_{\max} = 20$" | 8.1 against a predicted 10; it is right near 14 |
| "two orders of magnitude above the published window" | $10^4$ against 600 is 17× |

The lesson is not "check the numbers" — it is that **a comparative claim carries an arithmetic
claim inside it**, and "more accurate", "three times", "unchanged", "near unity" and "the same
proportional increase" are all arithmetic. Divide the two numbers before writing the adverb.

Things cut for exactly that reason:

| kept because it was true | why it went |
|---|---|
| "whose size changes over the run" | never used again |
| "with no tunable threshold" | nobody asked whether there was one |
| "the path a converging run follows" (of a transient) | gloss of a term already glossed |
| "fixed before any result was computed" (in the intro) | §4 is about that; the intro does not need it |
| "The difficulty is that the procedure returns a number whether or not the record contains a dimension for it to measure" | pompous restatement of "returns a number on any record" |
| "One further quantity is measured throughout this paper and is not a dimension at all" | just say "We also measure …" |
| "Three dimensions are defined on $\gW$. They take different values." | the three definitions follow |
| "with no per-system calibration", "that we then locate" | padding on a contribution |
| "Six requirements were fixed … Each has a name, used to refer to it below" | obvious from the list that follows |
| "Companion statistics", "Matched bandwidth" as protocol requirements | neither is a requirement; one is a fact about the construction, the other a description of what we report |
| "We distinguish the active dimension from …" as contribution 1 | distinguishing terms is not a contribution |
| "all three of which the literature regularly confuses with it" | the confusion is mine to avoid, not theirs to be accused of |
| "so a negative result there would be decisive" | overclaims and is not needed |
| "and we report it rather than estimate it" | nobody asked |
| "We also measure a quantity that is not a dimension" | evident from the definition that follows |
| the degeneracy indicator, in the body | mentioned once, used only in the appendix; it belongs there |

**A whole item can be the thing to cut, not just a clause.** Two of six "requirements" and one of
four "contributions" went because they were not the kind of thing they were listed as. Ask what
each list item *is*, not only whether it is true.

**Cut a term that is never defined rather than defining it.** *Admissible* ran through the body,
four captions, two appendices and two figure panels without a definition; every use became a plain
statement ("the regime the estimate needs", "where the accurate cases fell", "passes at both $r$").
The same for *calibration* and *resolvable*. A term earns its place only if the paper turns on it.
The converse also holds: a term the paper *does* turn on must be explained the first time, and
"the path fills an $r$-dimensional torus" was not — it now says the path lies on the torus and
passes arbitrarily close to every point of it.

### Numbers in the appendix

Appendices attract numbers that no argument uses. Measure the offenders — count the numerals per
word in each paragraph — and then cut. The test is whether the sentence still makes its point with
the figure replaced by its magnitude:

- "$138\,048$ close pairs surviving of $138\,761$; it costs the transient all $131\,849$ of its
  own" → *costs the fast drive almost none of its close pairs and the transient all of them*.
- "returns $1.19928$ exactly; the transient returns $1.1969 \pm 0.0011$ across all forty-eight
  cells" → *returns $1.199$; the transient returns that value in every cell*.
- "$41.75$ seconds without the sketch against $40.44$ with it, a nominal overhead of $-3.1\,\%$" →
  *the measured difference is negative*.
- "the error runs from $0.17$ to $2.03$", "$1.1\,\%$ at the median and $8.4\,\%$ at its worst",
  "a $120$-parameter model to $6\times10^{-5}$ relative" — all replaced by their magnitudes.

Numbers that stay are the ones a reader would otherwise have to take on trust and cannot look up:
the frozen configuration's constants, the grid a headline cell was chosen from, the one figure a
claim turns on. Everything else belongs in the table it came from.

## Then: say the thing, not the relation

- **A free-standing `what`-clause hides a noun.** "fix what the estimator is scored against" → *the
  ground truth*. "records what was actually run" → *records the procedure as run*. "below what we
  can measure" → *below our resolution*. Zero of these should survive, including in column headings
  and panel titles.
- **`is what` is never needed.** "is what supports the claim" → *supports the claim*.
- **`is the quantity that is`, `the pair of conditions X asks for`, `the one that …`** — all
  periphrasis. Name the thing.
- **Appositive `, which is` is a colon or nothing.** "Windows are sixty logged rows, which is 600
  optimiser steps" → *Windows are sixty logged rows: 600 optimiser steps*.
- **Do not gloss with `, meaning …`.** "over a window of training, meaning a block of consecutive
  optimiser steps" → just write *over a block of consecutive optimiser steps*. If the term is not
  needed at that point, the gloss is not needed either.
- **Italicise only the thing being emphasised.** Not a whole sentence around a question; italicise
  the question.

## Then: no counting, no agency, no informality

**Counting.** Drop it wherever it does no work. Not "Four such counts are in circulation, and they
do not agree. Three of those counts are already named." — just define them. Not "The twelve
observers" as a table title; "The observers". Not "Two configurations …, one for each …, Both
were …" — three counting words for two objects. Never make a bare determiner the subject: "Both lie
at a grid boundary", "The last two are the records …", "The first three fix …".

**Agency.** An estimator does not *see*, *ask for*, or *work at*; a paper does not *choose*; a
condition is not something a section *asks for*; a property does not *bear on* a construction.
Replacements used: the estimator sees them → as raw scalar logs; standardised as the estimator
standardises them → standardised to zero mean and unit variance as in the algorithm; the pair of
conditions §3.3 asks for → the two conditions of §3.3; the paper could not choose → we could not
choose; the torus it sees → the torus it resolves; two properties bear on the construction → the
backbone has two properties relevant to the construction.

**Informality.** Struck out: survives (the projection) → can be recovered from; free to move →
confined to; lifts the cap → removes the cap; scan → sweep; clear that floor → exceed that floor;
confirms the diagnosis → confirms this; price → cost; read off the logits → taken from the logits;
for nothing → at no cost; the winning cell → the result; runs sit in a band → lie in a band; with
only the window shortened → shortening the window and changing nothing else; the budget (of a
training run) → the allotted steps, or the run. Earlier rounds: cut against, tell against, rescues,
manufactures, flattering, go flat, gets right, gives the same picture, destroys, worth having, the
ladder, knob, arm, pipeline, inventory (verb).

**Do not name a pair by its count.** "the two settings", "the two diagnostics", "the two groups",
"the two conditions", "the two removals", "the two failure regimes", "the two sets of runs" — every
one of these asks the reader to remember which pair. Name them: *the transformer runs and the
perceptron runs*, *the identifiability ratio and the trend-crossing count*, *the generalising runs
and the memorising ones*, *a transient and a stochastic drive*. The same for "In the first … In the
second …" and "The first is … The second concerns …" when the two have names, and for "Two of our
systems fail this check" → *the frozen nonlinear decoder and the perceptron in a $k$-subspace fail
it*.

This check is advisory too. "the two hash families", "the three regimes of §3.3" and "the two
copies" are legitimate: the pair was named in the sentence before, or the `\cref` names it. Read
each hit.

**Cut "A rather than B" wherever A alone does the work.** "in hours rather than days" → *in a few
hours*. "over a family of observers rather than for one" → *over a family of observers*. Keep the
contrast only where the negative half is the point — "an assumption about the run rather than a
property we can verify from the log" earns it.

The scanner's contrast check is **advisory, and must be judged one by one**. After the sweep, all
29 surviving instances in the appendix earn their B: it is the reading the reader would otherwise
take. "scored against the measured effective rank rather than against $r$" ($r$ is what one would
assume), "a limit of resolution rather than a fact about the run", "removed by that stride rather
than blurred by it", "a bound rather than a result". Do not cut this construction mechanically to
drive a number down.

**Do not stack negations.** "is not a level but the value a divergent quantity happens to take at
the cap" → *is a divergent quantity evaluated at the cap rather than a level*. "is the lattice bias
of a deterministically sampled curve, not an error" → drop the second half. Worst of all is the
triple: "Recurrence is needed here and nowhere else: not by the estimator, which would accept
points drawn from independent runs, but by a protocol whose entire sample is one trajectory" →
*The estimator itself would accept points drawn from independent runs; recurrence is needed because
our entire sample is one trajectory.* State what is true and let the contrast follow.

**Swapping one piece of jargon for another is not a fix.** Replacing "window" with "block of
consecutive optimiser steps" made the abstract no clearer. The fix was to drop the term where the
abstract does not need it and say *over a stretch of training*.

**Attribute a name.** "The available dimension is $k$" invites "says who?". Write *we call $k$ the
available dimension*, or cite whoever does. The same applies to a quantity: the participation ratio
now carries its citation at the point of definition.

**Do not appeal to a "textbook definition" without saying of what.** "$\varepsilon$ stays finite,
where the textbook definition would let it go to zero" → *whereas a box-counting dimension is
ordinarily defined as a limit $\varepsilon \to 0$*.

**Name the experiment, not just the section.** "run under excitations covering all three regimes of
§3.3" → *driven quasiperiodically, with mini-batch noise, and by plain gradient descent, covering
the recurrent, stochastic and transient regimes*. Likewise, list the six systems where they are
first claimed rather than only in the appendix, and say what a table checks and on what.

**If a statistic is reported, say why it is reported.** Three companion statistics sat in §3.2 with
no stated purpose for several rounds. They exist because each can grow with $r$ with no geometry
involved, so if one matches the truth as well as the estimator does, the estimator's accuracy is no
evidence of a dimension. That sentence is the reason the statistics are in the paper, and it was
missing.

**One term per thing.** Fewer terms, used precisely. In this paper: **$k$ counts directions,
$r$ counts components.** So "a ceiling near eight directions" → *near eight components*; "at two and
four directions" → *at $r = 2$ and $r = 4$*; "the fewer directions are active" → *the smaller $r$
is*; the frozen configurations are the *eight-component* and *twenty-component* ones, while the
systems they run on have $k = 10$ or $k = 20$ *available directions*. Do not alternate between
"modes", "phases" and "directions" for the same object — the constructed systems have $r$
**phases**.

## Then: one claim, in order

- **`, and` is the tell.** It marks a sentence that was already finished. The fix is to cut one of
  the two things; if both survive, put a stop between them. Serial-comma lists are the legitimate
  use — and they are the *only* one.

  **Count them yourself; the scanner sees a fifth of them.** Its clause-join check fires only when
  a subject and a finite verb stand on both sides, so it reported 18 in a document that contained
  **81**. The reader does not apply that test. Everything it missed was still the fault: the
  afterthought (`returns $2.36$ at every exclusion, and correctly so`), the compound predicate
  (`It measures the cost of that departure, and varies the exclusion`), the two-item caption title
  (`the accuracy of each observer, and whether smoothness alone could produce it`). Enumerate every
  `, and` in the file, decide each one, and expect roughly a quarter of them to be lists you keep.
  This round: 81 → 22, all 22 lists.

  Each has its own fix, and choosing by eye is the point: an afterthought is **cut**
  (`and correctly so` → *correctly returns*); a compound predicate **loses the comma**; two clauses
  take a **full stop**, or `so`/`yet`/`nor`/`;` where the relation is real
  (`the record contains no returns, and in place of $1$ …` → *…, so in place of $1$ …*). A
  document at zero `, and` clause-joins reads as one that decided, not one that ran a filter.

- **Do not fix a monoculture by installing another one.** Driving `, and` to zero pushed `, so` to
  **93** — one every 160 words, four in the abstract's single paragraph, three in one paragraph of
  §7, and twice inside a single sentence (*"measures variation inside the window, so the collapse
  is a collapse in that variation, so $d_{\mathrm{act}}$ may be unchanged"*). The reader feels the
  repetition of the *repair* as sharply as the repetition of the fault.

  The remedy is not a third connective but a different sentence. Reach for: a **colon** when the
  second half delivers what the first promised (*"a run that has merely come to rest falls as far:
  the sharpness and the reversal mark the transition"*); a **semicolon** for two balanced facts; a
  **participle** for a consequence that is really a property (*"so that $d_{\mathrm{act}} = r$"* →
  *"giving $d_{\mathrm{act}} = r$"*, *"so the delay lag keeps its meaning"* → *"leaving the delay
  lag its meaning"*); **`therefore` inside the clause** rather than a connective in front of it;
  and a plain **full stop**, which is right more often than it feels.

  Check the shape, not the count: no two adjacent sentences on the same connective, none twice in
  one sentence, none four times in a paragraph. 93 → 70 fixed every cluster; the survivors are
  spread out and read as choices.

- **Watch the inversions a rewrite leaves behind.** *"…; so are the four generalising runs"* is
  correct and unreadable. Give an inversion its own sentence: *"So is the surrogate comparison, in
  which each run is measured only against surrogates of itself."*
- **A clause that only contrasts with what you just wrote is usually cuttable.** "the check was
  never carried out on them, not because they failed it" — the next sentences explain it.
- **Do not cram.** Three ideas in one period is two too many.
- **Cause before effect, premise before conclusion.**
- **Do not announce what the next sentences will do.** "Two features of that definition are
  deliberate", "The qualification is the point", "Two consequences matter here" — delete and start
  with the content. The worst form counts as well as announces, and it is everywhere in an
  appendix: "Three things follow", "Five properties of these choices need to be stated", "The
  result is qualified in five ways", "Two departures from the rest of the paper are deliberate",
  "Three of the rows deserve a note". All five were deleted outright; the items that followed them
  lost nothing.
- **Do not talk to yourself in the paper.** "they should not be read as three results", "we draw no
  general rule from it", "Only the difference between the columns is meaningful", "it is listed for
  completeness and not used", "This is a departure from the stated protocol rather than a
  deliberate choice", "That choice moves the numbers, so it is stated wherever it applies". Each is
  the author instructing the author. If a reading needs forbidding, the sentence that states the
  result should be written so the wrong reading does not arise.
- **Do not raise an objection in order to answer it.** "That is less serious than it appears …
  Even so, a wider grid could move the level" — keep the concession and delete the argument. Same
  for the defence nobody asked for: "The oscillation is real and not a rounding artefact", "No
  trajectory is regenerated, so the systems cannot drift", "They are distinct quantities".
- **No pseudo-clefts**, no sentence-initial `And`/`But`, no fragments, no absolute constructions, no
  dangling participles, never delete a head noun, no possessive on an inanimate name holding a
  number, `where` is not `whereas`.

## Referential debt

The reader has the last two paragraphs and not all of those.

- **Name a numbered item, do not number it.** *the zero-learning-rate check*, not *requirement 4* —
  inside tables above all. If numbering exists only to be cited, drop the numbering.
- **Name a thing rather than its rank in a list** ("the one candidate the literature offers" → *the
  edge of stability*).
- **Gloss a term where the reader first meets it**, not where it is defined; a term defined only in
  an appendix may not appear bare in the body.
- **Agentless passive invites "by whom?"** — "the three quantities it is confused with" → *which
  the literature regularly confuses with it*, or cut.
- **A section title is a debt too.** "Four quantities, and the one we estimate" → *The active
  dimension*.
- **Cutting creates debt of its own.** Deleting the sentence that glossed the `tab:obs` row
  "parameter norms (two)" left a row naming nothing; the row had to be renamed. Deleting the
  surrogate half of a paragraph left the heading "Linear detrending removes the effect, whereas the
  surrogates do not" claiming something the paragraph no longer argued. Cutting the clause that
  defined "its soft half" left the term defined nowhere; cutting the values of the two pre-specified
  criteria left "Both pre-specified criteria hold" holding nothing; cutting a sentence from §7.2
  left "That growth is the one §7.2 reports" pointing at a paragraph that no longer reports it.
  **After every cut, re-read the paragraph around it, the heading above it, and anything that
  pointed into it** — captions, the appendix outline, and every `\cref` aimed at that passage.
  A cross-reference is a claim about what the target *says*, and `\cref` resolving proves only that
  the label exists.

## Floats

- **Measured values live in tables and figures**; the body tells the story.
- **A figure earns its place by showing a claim the tables can only assert.** Four went in
  because the paper proved everything in tables and a reader never once saw a log with the
  estimate running beside it: the log at four ranks against the level read off it, a known
  change of the number of phases tracked as it happens, the delay reconstruction itself,
  and the neighbours a transient loses to the exclusion. None needed a new paragraph. Each
  sits in the appendix that already argues its claim and is reached by a `\cref` added to a
  sentence already there.
- **Put the float after the complete paragraph, never inside it.** `\begin{figure}[H]` in
  the middle of a paragraph splits it: the sentence after the float is set as a new
  paragraph, half a page from the sentence it continues. Both new appendix figures had to
  be moved.
- **A new caption reintroduces every fault the prose was cleaned of.** Three of the four
  captions written this round arrived carrying `what` clauses, an `is what`, and a bare
  pair — in a document that had been at zero on all three for two rounds. *Run the scan
  after adding a figure, not only after cutting one.*
- **Deleting a float breaks things outside the document.** `actdim/tables.py` registers one
  auditor per table label, and `tests/test_tables.py` asserts the article's table count and
  the exact set of known disagreements. Removing three tables left five stale registry
  entries and six failing tests, discovered a round later. Renaming a column does it too:
  `budget` → `steps` orphaned an errata check. **Run `python -m pytest` after changing the
  article's floats or column headings**, not just `pdflatex`.
- **A caption must let the reader see what question the figure answers**, in words used elsewhere
  in the paper. The two worst offenders this round opened with "the two candidate estimands" and
  with a bare "label-matched pairs" — neither reconstructible from the caption alone. Say what is
  being compared and why.
- Target ≤ 60 words for a figure; the argument the figure supports belongs in the text.
- **Panel titles are prose too**, and live in `../code/actdim/figures/panels.py`. Fixing one means
  regenerating the figure.

## Mechanics

- **`pdflatex` passing does not mean the venue build passes.** A page break landing on a
  hyperlink makes pdfTeX say `\pdfendlink ended up in different nesting level than
  \pdfstartlink`. Plain `pdflatex` treats that as a warning and finishes; the venue
  scripts run `latexmk ... -halt-on-error`, which stops on it, and the whole build dies
  with `latexmk failed` and a log truncated mid-sentence. It cost an hour once, so:
  **after editing anything on the first two pages, run `python make_newinml.py`, not only
  `pdflatex`.**

  Nothing is wrong with the text when this happens. It is knife-edge — one word anywhere
  before the break moves it — so do not hunt for the offending link. Reword the nearest
  sentence in a way you wanted anyway and rebuild. Bisect with the venue script if it is
  not obvious which edit did it: revert one edit at a time and see which build succeeds.

- **Nine pages excluding references.** Check where the heading falls:
  `pdftotext -layout -f 9 -l 9 report.pdf - | grep -c EFERENCES` should be 1. Grep for
  `EFERENCES`, not `REFERENCES`: the style sets the heading in small caps and `pdftotext` extracts
  it as `R EFERENCES`, so the obvious pattern silently returns 0 and the check looks like a
  failure. The older test (`-f 10 -l 10 … END{print n+0}` equal to 1) misreports in the other
  direction, when the body is short enough that the references start on page 9.
- **Space is word count, not sentence count.** Splitting a sentence inside a paragraph is free; a
  new paragraph costs half a line. One body line is about 20 words.
- Cutting is the only reliable way to make room, and it is what the paper needs anyway.

## How to edit

- Exact-string replacement scripts, one theme per pass, `assert count == 1`.
- **Always write the script to a file and run it.** Piping a heredoc through bash mangles
  backslashes even with a quoted delimiter: `\rho` and `\tau` silently became `ho` and `au` again
  this round, and the run reported success while every macro-bearing substitution had quietly
  failed to match. If a replacement "cannot be found" for no apparent reason, this is why.
- Raw Python string literals (`r"""…"""`).
- Rebuild and re-scan after every pass; passes reliably reintroduce what they fixed. `, and`
  climbed back three separate times this round, because cutting words raises the *rate* even when
  the count falls — watch the per-thousand figure, not the count.
- **Read the rendered PDF end to end at least three times, on different passes.** One reading finds
  perhaps a fifth of what is there.
- **Parallel readers find the cuts; a must-keep list makes them safe.** Four auditors, one per
  slice of the appendix, each asked for candidates ranked by (words saved) × (confidence it is safe
  to lose) *and* for a list of passages some other part of the paper depends on. The second list is
  what makes the first usable: it caught the `tab:gt` scoring note, the $q = 0.5$ mechanism and the
  reference-band asymmetry, each of which reads as filler in isolation. Apply nothing without
  checking it against the must-keep lists of the other slices.
- Figures: `(cd ../code && python -m actdim run paper.figures --no-deps)`. Without `--no-deps` it
  re-runs the experiments, which takes hours and fails here.

## Measurements

```bash
python style_scan.py            # or: python style_scan.py report.tex -v
pdftotext -layout -f 9 -l 9 report.pdf - | grep -c EFERENCES      # expect 1
grep -c "Overfull\|LaTeX Warning\|undefined" report.log           # expect 0
grep -o '\\label{[^}]*}' report.tex | sort > /tmp/l                # every float
grep -o '\\[Cc]ref{[^}]*}' report.tex | sort -u                    # what is cited: diff them
```

Baseline at the end of this round. Treat these as ceilings.

| | body | appendix |
|---|---|---|
| words / sentences | 5284 / 233 | 9332 / 391 |
| `, and` clause-joins | 8 | 11 |
| bare pair ("the two X") | 0 | 0 |
| `'what'` clauses | 0 | 0 |
| `is what` | 0 | 0 |
| `, which` | 14 | 22 |
| count as subject | 0 | 0 |
| bare numeral for a set | 0 | 0 |
| oblique naming | 0 | 0 |
| em dashes | 0 | 0 |
| pseudo-cleft | 0 | 0 |

**The appendix fell from 13 503 words to 9 332 and the paper from 40 pages to 33**, without losing
a claim: five duplicated floats, every appendix opener, every announcement-with-a-count, and every
fact that had a second home. Captions: 33, mean 64 words, down from 72. The body is unchanged in
length, because it is fixed at nine pages — **all page savings have to come from the appendix**,
and body edits buy style, not space.

**Watch the rate, not the count.** Cutting words raises every per-thousand figure even when the
absolute count falls, so `, and` appeared to regress three separate times in one round. Re-scan
after every pass and read the rate column.

**A compression pass reintroduces the faults it is not looking at.** Rewriting the appendix for
length took `, and` from 18 to 33 and `what` from 0 back to 4 in a single pass, because a shorter
paraphrase reaches for the shortest connective. Always re-run the full scan after compressing, not
only the check for what you were compressing.

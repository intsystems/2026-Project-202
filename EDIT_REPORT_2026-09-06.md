# Editing report — 2026-09-06

## Scope

This pass applies the supervisor's comments from September 6, 2026 and the substantive ICOMP reviews to the current `icomp_v2` article. The edits preserve the reported experiments and numbers; they change claims, exposition, and one figure design.

## Article text

- Rewrote the abstract so its evidence hierarchy is explicit: one learned-system benchmark satisfies the prespecified validation protocol; the remaining constructions are stress tests; the grokking trajectory result is reported as a regularisation-associated observation in four runs.
- Removed the implication that the scalar estimate provides a dimensional result on the grokking logs. The matched-window result is now described as a transition-correlated signal whose level has no dimensional interpretation.
- Rewrote the Introduction's Contributions paragraph as three concrete contributions:
  1. the definition and separation of active dimension from available dimension, functional dimension, and trajectory effective rank;
  2. the frozen, held-out validation protocol and its failure modes;
  3. the cautious application to grokking and the direct trajectory measurement.
- Rephrased the externally forced versus trained distinction in the validity section as three short sentences. This makes the scope limitation visible without the repeated causal `so` construction.
- Kept the distinction between active dimension and covariance participation ratio explicit. The revised abstract now names the latter as a covariance statistic.

## Figure and colour changes

- Simplified `fig_timing`: removed the fitted diagonal line for a hypothetical fixed absolute transition step. That line was visually dominant and made the panel harder to interpret. The plot now shows each minimum relative to its run's own generalisation step, with the search interval retained.
- Updated the corresponding caption so it describes the displayed quantities and no longer refers to the removed line.
- Kept the existing Paul Tol palette and regenerated the complete figure set through the registered figure pipeline. The palette remains: deep blue for recurrent, brick rose for stochastic, dark gold for transient, and grey for reference elements. It is used consistently across the paper.

## Validation

- Rebuilt the ICOMP PDF with `pdflatex`, `bibtex`, and repeated `pdflatex` passes.
- Rebuilt the Artifacts edition with `python artifacts_article/make_artifacts.py`.
- Artifacts build: 34 pages total, 9 pages of main text, 0 undefined references, 0 undefined citations, 0 duplicate links, and 0 overfull boxes.
- Ran `icomp_v2/style_scan.py`: no hits for the principal prohibited constructions (`what` clauses, `is what`, bare pairs, bare numerals, pseudo-clefts, em dashes, banned informal terms, or initial `And/But`).
- Ran `git diff --check` successfully.

## Remaining editorial points

The scanner still reports advisory contrast constructions and several long captions. These require sentence-level judgement because many are scientifically meaningful. The appendix also contains a small number of legitimate `, and` constructions in theorem statements and lists. No numerical experiment was changed in this pass.

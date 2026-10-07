# Unified manuscript (7 October 2026)

The deliverable is aistats2027.pdf: main text, AI Use Statement, references, checklist, and one-column appendices in one document.
The optional aistats2027_blue.pdf is the same complete document with editorial markup, not a separate supplement.

Run python build.py to regenerate figures/tables, verify campaign numbers, compile the complete black/blue documents, and check citations, layout, the eight-page main-text limit, text equality, and actual blue coloring.
Run python package.py after a successful build to package the current sources.

Edit sections/ and the shared preamble.tex. main_part.tex and appendix_part.tex are source fragments only.
Previous split-document wrappers and PDFs are retained under _retired_split/ for local history and excluded from the source package. build_split.py now delegates to the unified build.

# ASCOMP / ICOMP 2026 poster

English, A1 portrait (594 x 841 mm), one page. A1 follows the supplied visual template; no venue-specific size requirement was provided.

## Files

- `main.pdf`: printable vector PDF.
- `main.tex`: editable LaTeX source.
- `figures/`: four vector figures copied from the article assets.
- `preview.png`: screen preview.

## Build

Run from this directory in a TeX installation with tikzposter, Latin Modern, hyperref and qrcode:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

All scientific figures are included locally; the build does not download files.

## Sources and editorial choices

Authoritative scientific source: `../../artifacts_article/icomp_artifacts.pdf`, version available on 19 September 2026. The presentation `../../talk/main.pdf` was secondary. Visual starting point: `../poster/main.tex`; the old poster was not used as a scientific source.

The included figures are `fig_method.pdf` (delay reconstruction), `fig_regimes.pdf` (controlled recovery and regimes), `fig_map.pdf` (grokking diagnostic map), and `fig_dip.pdf` (aligned trajectory/scalar results). Their contents match the article figure assets.

The text distinguishes conditional active-dimension recovery from covariance effective rank in grokking; it does not claim early prediction or an active-dimension collapse. Limitations include the forced validation setting, low resolution ceiling, window dependence, regularisation/outcome confounding and the full-batch non-replication. Figure shading is identified as hash-family spread rather than uncertainty across runs.

Author/contact: Nikolay Karlov, MIPT, karlov.na@phystech.edu. The QR code points to the project repository.

## Checks

The PDF compiles successfully as one A1 page. No overfull boxes were reported. Text bounds were checked, the layout was visually inspected, and the repository QR code was decoded from a 300 dpi rendering.

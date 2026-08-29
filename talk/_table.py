# -*- coding: utf-8 -*-
"""Regenerate the README's frame table straight from main.tex, so it cannot drift."""
import pathlib
import re

src = pathlib.Path("talk/main.tex").read_text(encoding="utf-8")
rows, section, n = [], "", 0
for line in src.splitlines():
    m = re.match(r"\\section\{(.+?)\}", line.strip())
    if m:
        section = m.group(1)
        continue
    m = re.match(r"\\begin\{frame\}(?:\{(.*)\})?", line.strip())
    if m:
        n += 1
        rows.append([n, section, (m.group(1) or "титул"), ""])
        continue
    m = re.search(r"\\slidefig\{figures/(\w+)\}", line)
    if m and rows:
        rows[-1][3] = "`%s`" % m.group(1)

for r in rows:
    title = (r[2].replace("\\dact", "$d_{act}$").replace("---", "—")
             .replace("$\\to$", "→").replace("\\approx", "≈")
             .replace("$", "").replace("d_{act}", "$d_{act}$"))
    print("| %d | %s | %s | %s |" % (r[0], r[1], title, r[3] or "—"))
print()
print("frames:", n)

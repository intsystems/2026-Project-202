# AISTATS table-format audit

The official AISTATS 2027 sample paper requires tables to be centered, neat, clean, and legible, with the number and caption above the table. It does not impose a separate numeric prohibition on small font commands or reduced column spacing. The practical constraint is readability at the submitted PDF size.

For the main paper we therefore use the template table settings and small font where needed. The detector table no longer uses the smallest font or a 2pt column spacing; its caption remains above the table. Appendix tables may use compact formatting when necessary, but their text is checked in the rendered PDF.

The official sample also permits the appendix to be included after the main paper or supplied separately. The source is split into main_part.tex and appendix_part.tex; the latter switches to one-column layout for the appendix material.

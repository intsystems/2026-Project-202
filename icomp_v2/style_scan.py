# -*- coding: utf-8 -*-
"""Census of the prose faults listed in writing_guidelines.md.

    python style_scan.py [report.tex]

Prints counts and, for the structural faults, the offending context. Run before
and after every editing pass: passes reliably reintroduce the faults they fix.
"""
import io
import os
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "report.tex")

SUBJ = (r"(?:it|they|we|this|that|these|those|there|the|its|their|each|every|both|neither|either|"
        r"no|one|two|three|four|five|six|seven|eight|nine|ten|all|most|some|nothing|only)")
VERB = r"(?:is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|will|would|\w+(?:s|es|ed))"

# (label, pattern, show_context)
CHECKS = [
    ("elliptical possessive", r"[a-z]'s(?:,| rather than| and not| not) ", True),
    ("inanimate 's",          r"\b[a-z][a-z-]+'s\b", False),
    ("pseudo-cleft",          r"\bWhat [a-z][^.]{0,60}? is\b", True),
    ("'where' for 'whereas'", r"[a-z], where [a-z]", True),
    ("initial And/But",       r"\. (?:And|But) ", True),
    ("absolute 'X being'",    r", the [a-z ]{3,25} being ", True),
    ("informal nouns",        r"\b(?:sits|cheap|flags?|arm|outright)\b", True),
    ("', and ' clause-join",  r", and (" + SUBJ + r"\b[^,.;:]{0,60}?\b" + VERB + r"\b)", True),
    ("em dash",               r" --- ", False),
    ("read*",                 r"\bread(?:s|ing)?\b", False),
    ("cost*",                 r"\bcosts?\b|\bcosting\b", False),
    ("control*",              r"\bcontrols?\b", False),
]


def flatten(s):
    s = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", " ", s, flags=re.S)
    return re.sub(r"\s+", " ", s)


def captions(text):
    out = []
    for m in re.finditer(re.escape("\\caption{"), text):
        i, d, j = m.end(), 1, m.end()
        while d:
            d += (text[j] == "{") - (text[j] == "}")
            j += 1
        label = re.search(re.escape("\\label{") + r"([^}]+)\}", text[j:j + 140])
        out.append((label.group(1) if label else "?", len(flatten(text[i:j - 1]).split())))
    return out


def main():
    text = io.open(PATH, encoding="utf8").read()
    body = text[text.index(r"\begin{abstract}"):text.index(r"\bibliographystyle")]
    app = text[text.index(r"\appendix"):]

    verbose = "-v" in sys.argv
    for name, seg in (("BODY", body), ("APPENDIX", app)):
        flat = flatten(seg)
        words = len(flat.split())
        sents = len(re.split(r"(?<=[.!?]) (?=[A-Z\\])", flat))
        print("== %s  %d words, %d sentences" % (name, words, sents))
        for label, pat, ctx in CHECKS:
            hits = list(re.finditer(pat, flat, re.I))
            print("   %-22s %4d   (%.1f / 1000 words)" % (label, len(hits), 1000.0 * len(hits) / words))
            if verbose and ctx:
                for m in hits[:12]:
                    print("        ...%s" % flat[max(0, m.start() - 62):m.end() + 58])

    caps = sorted(captions(text), key=lambda c: -c[1])
    total = sum(n for _, n in caps)
    print("== CAPTIONS  %d, %d words, mean %.0f" % (len(caps), total, total / len(caps)))
    for label, n in caps[:8]:
        print("   %4d  %s" % (n, label))
    print("   (target: <= 60 words for a figure; longer only to define table columns)")


main()

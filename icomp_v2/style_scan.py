# -*- coding: utf-8 -*-
"""Census of the prose faults listed in writing_guidelines.md.

    python style_scan.py [report.tex]

Prints counts and, for the structural faults, the offending context. Run before
and after every editing pass: passes reliably reintroduce the faults they fix.

The checks fall into two groups. The first group is about sentence construction
and word choice. The second, added after the round that rewrote the body for
readability, is about *referential debt*: places where the text points at
something the reader has not been given or is expected to have memorised.
"""
import io
import os
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "report.tex")

SUBJ = (r"(?:it|they|we|this|that|these|those|there|the|its|their|each|every|both|neither|either|"
        r"no|one|two|three|four|five|six|seven|eight|nine|ten|all|most|some|nothing|only)")
VERB = r"(?:is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|will|would|\w+(?:s|es|ed))"

NUM = r"(?:two|three|four|five|six|seven|eight|nine|ten)"

# A bare number standing in for a set the reader must reconstruct: "the four",
# "the two settings fail", "Two of them". Fires when the numeral is followed by
# punctuation or by a verb rather than by the noun it counts.
BARE_NUM = (r"\bthe " + NUM + r"\b(?=[,.;:]|\s+(?:is|are|was|were|do|does|did|fail|fails|failed|"
            r"show|shows|give|gives|report|reports|and|but|which|that|however|above|below)\b)")

# Announcing how many items are coming instead of naming the first one. A count is
# legitimate only when the items are enumerated immediately after it, so the pattern
# fires only where no enumerator follows within the next stretch of text.
COUNT_NOUN = (r"(?:statistics|things|observations|qualifications|reasons|properties|departures|"
              r"checks|results|consequences|conditions|features|points|notes|arguments|"
              r"explanations|comparisons|criteria|caveats|failures|ways|kinds|respects)")
ENUM = (r"(?:First|Second|Third|The first|The second|One of|\\emph\{|\\item|\\textbf\{|"
        r"\\paragraph\{|Its |In the first|: )")
COUNT_HEAD = (r"\b(?:Two|Three|Four|Five|Six|Seven|Eight|Nine)\s+(?:further\s+|other\s+|more\s+)?"
              + COUNT_NOUN + r"\b(?![^.]*\bof\b)(?![\s\S]{0,240}?" + ENUM + r")")

# A numbered item cited without its name.
UNNAMED_REQ = r"requirement~?\d(?!\s*,\s*the\b)"

# Naming a thing by its rank in a list instead of naming the thing.
OBLIQUE = (r"\bthe (?:one|only|single) (?:[a-z]+ ){0,2}"
           r"(?:candidate|alternative|option|choice|example|instance|case) (?:the|that|which)\b")

# Metaphor where a plain verb exists, and programmer's nouns for research objects.
BANNED = (r"\b(?:sits|cheap|flags?|arm|outright|pipelines?|inventor(?:y|ies|ied)|knobs?|"
          r"cut against|cuts against|tell against|tells against|rescues?|rescued|manufactures?|"
          r"flatter(?:s|ing|ed)?|fires\b|the ladder|there to (?:return|find|measure|count))\b")

# Periphrasis. A free-standing "what" clause almost always hides a plainer noun:
# "what the estimator is scored against" -> "the ground truth"; "is what supports the
# claim" -> "supports the claim". Likewise "is the quantity that is unbiased".
WHAT = r"\bwhat\b"
IS_WHAT = r"\b(?:is|are|was|were)\s+what\b"
PERIPHRASIS = (r"\b(?:is|are)\s+the\s+(?:quantity|one|thing|reason|value|case|version|point)\s+"
               r"(?:that|which)\b")

# "the two settings", "the two diagnostics", "the two groups": a count standing in for
# a pair the reader is expected to have memorised. Name them instead.
BARE_PAIR = (r"\bthe (?:two|three|four) (?:settings?|diagnostics|groups?|conditions?|statistics|"
             r"quantities|extremes|removals|failures|candidates|estimands|halves|arms|regimes|"
             r"rules|norms|spaces|sweeps|controls|findings|qualifications|departures|checks|"
             r"exceptions|measurements|sets of|predictions|centres|hypotheses|copies|"
             r"projections|hash families)\b")

# "A rather than B" where A alone would do. Advisory: many of these are the point of
# the sentence, so read every hit rather than trusting the count.
CONTRAST = r"\brather than\b|\bnot a\b|\bnone of\b"

# A bare count or determiner carrying a sentence with no noun to anchor it:
# "Both lie at a grid boundary", "The first three fix ...", "Both are reported".
# A determiner followed by its noun ("Both penalties are zero") is fine.
BARE_DET = r"(?:Both|Neither|Either|All (?:three|four|five|six)|The (?:first|last|other) " + NUM + r")"
FINITE = (r"(?:is|are|was|were|do|does|did|lie|lies|show|shows|report|reports|fail|fails|fix|"
          r"fixes|ask|asks|hold|holds|matter|matters|remain|remains|apply|applies|give|gives|"
          r"carr(?:y|ies)|separat\w+|explain\w*|behav\w+|appear\w*|say|says)\b")
COUNT_SUBJ = r"(?:^|(?<=[.;:] ))" + BARE_DET + r"\s+" + FINITE

# (label, pattern, show_context)
CHECKS = [
    ("elliptical possessive", r"[a-z]'s(?:,| rather than| and not| not) ", True),
    ("inanimate 's",          r"\b[a-z][a-z-]+'s\b", False),
    ("pseudo-cleft",          r"\bWhat [a-z][^.]{0,60}? is\b", True),
    ("'where' for 'whereas'", r"[a-z], where [a-z]", True),
    ("initial And/But",       r"\. (?:And|But) ", True),
    ("absolute 'X being'",    r", the [a-z ]{3,25} being ", True),
    ("banned word",           BANNED, True),
    ("', and ' clause-join",  r", and (" + SUBJ + r"\b[^,.;:]{0,60}?\b" + VERB + r"\b)", True),
    ("em dash",               r" --- ", False),
    ("read*",                 r"\bread(?:s|ing)?\b", False),
    ("cost*",                 r"\bcosts?\b|\bcosting\b", False),
    ("control*",              r"\bcontrols?\b", False),
    # referential debt
    ("bare numeral for a set", BARE_NUM, True),
    ("count, items not listed", COUNT_HEAD, True),
    ("unnamed requirement",    UNNAMED_REQ, True),
    ("oblique naming",         OBLIQUE, True),
    # periphrasis
    ("'what' clause",          WHAT, True),
    ("'is what'",              IS_WHAT, True),
    ("'is the X that'",        PERIPHRASIS, True),
    ("count as subject",       COUNT_SUBJ, True),
    ("bare pair",              BARE_PAIR, True),
    ("contrast (advisory)",    CONTRAST, False),
    ("', which'",              r", which ", False),
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
            print("   %-24s %4d   (%.1f / 1000 words)" % (label, len(hits), 1000.0 * len(hits) / words))
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

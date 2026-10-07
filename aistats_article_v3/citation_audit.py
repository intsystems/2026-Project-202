from pathlib import Path
import json
import re

H = Path(__file__).resolve().parent


def cited(paths):
    text = "\n".join((H / p).read_text(encoding="utf-8") for p in paths)
    keys = []
    forms = []
    for m in re.finditer(r"\\(cite[a-zA-Z*]*|Cite[a-zA-Z*]*)\s*\{([^}]*)\}", text):
        forms.append(m.group(1))
        keys.extend(k.strip() for k in m.group(2).split(",") if k.strip())
    return sorted(set(keys)), sorted(set(forms))


def bbl_keys(path):
    text = (H / path).read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", text)))


main_paths = [
    "sections/abstract.tex", "sections/introduction.tex", "sections/method.tex",
    "sections/controlled.tex", "sections/applications.tex",
    "sections/forecast_utility.tex", "sections/campaign_main.tex", "sections/cost_discussion.tex",
]
supp_paths = ["sections/appendix.tex", "sections/practical_appendix.tex", "sections/campaign_appendix.tex"]
main_cites, main_forms = cited(main_paths)
supp_cites, supp_forms = cited(supp_paths)
bib_text = (H / "references.bib").read_text(encoding="utf-8") + (H / 'campaign_references.bib').read_text(encoding='utf-8')
bib_keys = sorted(set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)))
main_bbl = bbl_keys("aistats2027.bbl")
supp_bbl = main_bbl

result = {
    "citation_style": sorted(set(main_forms + supp_forms)),
    "main": {
        "cited_keys": main_cites,
        "bibliography_keys": main_bbl,
        "missing_bib_keys": sorted(set(main_cites) - set(bib_keys)),
        "missing_rendered_entries": sorted(set(main_cites) - set(main_bbl)),
    },
    "supplement": {
        "cited_keys": supp_cites,
        "bibliography_keys": supp_bbl,
        "missing_bib_keys": sorted(set(supp_cites) - set(bib_keys)),
        "missing_rendered_entries": sorted(set(supp_cites) - set(supp_bbl)),
    },
    "bibtex_warnings": [],
}
for stem in ["aistats2027", "aistats2027_blue"]:
    log = (H / (stem + ".blg")).read_text(encoding="utf-8", errors="replace")
    result["bibtex_warnings"].extend(
        f"{stem}: {line}" for line in log.splitlines() if "Warning" in line
    )

if result["citation_style"] != ["citep", "citet"]:
    raise SystemExit(f"Unexpected citation commands: {result['citation_style']}")
if result["main"]["missing_bib_keys"] or result["main"]["missing_rendered_entries"]:
    raise SystemExit(f"Main citation audit failed: {result['main']}")
if result["supplement"]["missing_bib_keys"] or result["supplement"]["missing_rendered_entries"]:
    raise SystemExit(f"Supplement citation audit failed: {result['supplement']}")
if result["bibtex_warnings"]:
    raise SystemExit(f"BibTeX warnings: {result['bibtex_warnings']}")

(H / "citation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

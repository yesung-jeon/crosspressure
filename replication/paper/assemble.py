#!/usr/bin/env python
"""Assemble the manuscript from the filled pages (run fill_intro.py first): introduction.md, sections/02-07, then one reference list built
from the per-page reference blocks (deduplicated, sorted), then Appendix A. HTML comments are dropped. Runs a heuristic citation
diagnostic (author-year patterns; review its output by hand) and an approximate whitespace word count (tables and appendices excluded). Writes paper/manuscript.md.
Usage: python paper/assemble.py"""
import re
from pathlib import Path

P = Path(__file__).resolve().parent
pages = [P / "abstract.md", P / "introduction.md"] + sorted((P / "sections").glob("0[2-7]_*.md"))
pages = [p for p in pages if not p.name.endswith(".template.md")]
appendix = P / "sections" / "A_design_details.md"; appendix_b = P / "sections" / "B_supplementary.md"
body, refs = [], {}
for p in pages + [appendix]:
    t = re.sub(r"<!--.*?-->\s*", "", p.read_text(encoding="utf-8"), flags=re.S)
    m = re.search(r"^#+ References.*$", t, flags=re.M)
    if m:
        for entry in [e.strip() for e in t[m.end():].strip().split("\n\n") if e.strip()]:
            refs.setdefault(entry.split("(")[0].strip() + entry[entry.find("("):entry.find(")") + 1], entry)
        t = t[:m.start()]
    if p.name == "introduction.md":
        t = t.replace("# Introduction", "# 1. Introduction", 1)
    body.append((p, t.strip()))
main = "\n\n".join(t for p, t in body if p != appendix)
ref_list = sorted(refs.values(), key=lambda e: e.lower())
out = main + "\n\n# References\n\n" + "\n\n".join(ref_list) + "\n\n" + dict(body)[appendix] + "\n\n" + appendix_b.read_text(encoding="utf-8").strip() + "\n"
(P / "manuscript.md").write_text(out, encoding="utf-8")

# citation check: for every entry, its first surname (or group author) and year must appear together in the text; and every
# "Name (YYYY" or "Name, YYYY" / "Name et al., YYYY" in the text must match an entry
text = main + dict(body)[appendix]
def lead(e):
    for g in ("Qwen Team", "Llama Team", "OLMo Team", "von der Heyde"):
        if e.startswith(g): return g
    return e.split(",")[0]
keys = [(lead(e), re.search(r"\((\d{4})", e).group(1)) for e in ref_list]
unused = [k for k in keys if not re.search(re.escape(k[0]) + r"[^()]{0,60}?\(?" + k[1], text)]
found = set(re.findall(r"((?:von der )?[A-Z][A-Za-z\-]+(?: Team)?)(?: et al\.| and [A-Z][A-Za-z\-]+| & [A-Z][A-Za-z\-]+|, [A-Z][A-Za-z\-]+,? and [A-Z][A-Za-z\-]+)?,? \((?:pp?\. [^)]*; )?((?:19|20)\d\d)|\(((?:von der )?[A-Z][A-Za-z\-]+(?: Team)?)(?: et al\.| & [A-Z][A-Za-z\-]+)?, ((?:19|20)\d\d)", text))
cited = {(a or c, b or d) for a, b, c, d in found}
missing = sorted(c for c in cited if c not in set(keys) and c[0] not in ("Section", "Figure", "Table", "Congress", "Appendix"))
words = len(re.findall(r"\S+", re.sub(r"(?m)^\|.*\|$", "", main))); rwords = sum(len(e.split()) for e in ref_list)
print(f"manuscript.md: {len(ref_list)} references; body words {words}; reference words {rwords}; total {words + rwords} (appendix excluded)")
print("cited but no entry:", missing); print("entry never cited:", unused)

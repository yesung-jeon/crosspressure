#!/usr/bin/env python
"""Pre-publication check: no secrets, personal paths, e-mail addresses or .env files in replication/. Exits 1 on any hit.
Usage: python scrub_check.py"""
import re, sys
from pathlib import Path
R = Path(__file__).resolve().parent
PAT = {"windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I), "posix user path": re.compile(r"/(?:c/)?Users/[A-Za-z]", re.I),
       "home path": re.compile(r"/home/[a-z]"), "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
       "huggingface token": re.compile(r"hf_[A-Za-z0-9]{20,}"), "openai/anthropic key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
       "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "aws key": re.compile(r"AKIA[0-9A-Z]{16}")}
ALLOW = {"e-mail": {"noreply@anthropic.com"}}
TEXT = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".sha256", ".example", ".gitignore", ""}
hits = []
for p in sorted(R.rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts: continue
    if p.name == ".env" or p.name.endswith(".env"): hits.append((p, ".env file", "")); continue
    if p.suffix not in TEXT or p.stat().st_size > 50_000_000: continue
    t = p.read_text(encoding="utf-8", errors="replace")
    for name, rx in PAT.items():
        for m_ in rx.finditer(t):
            if m_.group(0) in ALLOW.get(name, set()) or p.name == "scrub_check.py": continue
            hits.append((p.relative_to(R), name, t[max(0, m_.start() - 40):m_.end() + 40].replace("\n", " ")))
for h in hits: print("HIT", *h, sep=" | ")
print(f"scrub check: {len(hits)} hits"); sys.exit(1 if hits else 0)

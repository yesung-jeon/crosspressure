#!/usr/bin/env python
"""Step 2 - download Voteview member file (DW-NOMINATE scores for every member of every Congress).
Source: https://voteview.com/data (Lewis, Poole, Rosenthal, Boche, Rudkin & Sonnet). Writes data/HSall_members.csv (~6 MB)."""
import sys, urllib.request
from pathlib import Path
URL = "https://voteview.com/static/data/out/members/HSall_members.csv"
dst = Path(sys.argv[1] if len(sys.argv) > 1 else "data") / "HSall_members.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists():
    urllib.request.urlretrieve(URL, dst)
print(dst, dst.stat().st_size // 1024, "KB")

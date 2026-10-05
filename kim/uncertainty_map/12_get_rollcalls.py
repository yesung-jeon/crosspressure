#!/usr/bin/env python
"""Step 12 - download Voteview roll-call descriptions and individual votes for the Congresses used in step 13.
Source: https://voteview.com/static/data/out/{rollcalls,votes}/{H,S}{congress}_{rollcalls,votes}.csv (public; download
approved by the user 2026-10-04). Files are kept unchanged in ../data/voteview/ with a SHA-256 manifest.
Usage: python 12_get_rollcalls.py [--congresses 100 106 112 118]
"""
import argparse, hashlib, json, urllib.request
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--congresses", nargs="+", type=int, default=[100, 106, 112, 118]); args = ap.parse_args()
D = cm.HERE.parent / "data" / "voteview"; D.mkdir(parents=True, exist_ok=True); man = {}
for c in args.congresses:
    for ch in ("H", "S"):
        for kind in ("rollcalls", "votes"):
            f = D / f"{ch}{c}_{kind}.csv"
            if not f.exists():
                url = f"https://voteview.com/static/data/out/{kind}/{ch}{c}_{kind}.csv"
                urllib.request.urlretrieve(url, f); cm.log("downloaded", url)
            man[f.name] = dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
(D / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8"); cm.log(f"{len(man)} files in {D}")

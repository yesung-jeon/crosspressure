#!/usr/bin/env python
"""Fetch the public roll-call data into replication/data/ and check it against the files used for the paper.

Voteview updates its files as Congress votes, so a fresh download can differ from the copy the paper used (downloaded 2026-10-04/05).
The analyses use Congresses 80-118, so a hash mismatch caused only by new 119th-Congress rows does not change the inputs; this script
reports the mismatch and the row counts for Congresses <= 118 instead of failing.
The Policy Agendas Project roll-call topics (rcv_voteview_v3_4.csv) are not redistributed here: place the file in data/agendas/ yourself
(see README); this script only checks its sha256.

Usage:
  python data/fetch_data.py --members-only   # path B (analysis from intermediates) needs only HSall_members.csv
  python data/fetch_data.py                  # path A (full rerun): members, roll calls, votes (about 700 MB) + PAP check
"""
import hashlib, sys, urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent
FILES = {  # file -> (url, sha256 of the copy used for the paper, local path)
    "HSall_members.csv": ("https://voteview.com/static/data/out/members/HSall_members.csv",
                          "d78f55120b59534e99f30541469325059f953b210f0ddb6ebb54601c9303c0f6", DATA / "HSall_members.csv"),
    "HSall_rollcalls.csv": ("https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
                            "98fcc6eac8d966557c4f10c3a64aaee321b88b68522bf339ee18da858f5b1988", DATA / "voteview_all" / "HSall_rollcalls.csv"),
    "HSall_votes.csv": ("https://voteview.com/static/data/out/votes/HSall_votes.csv",
                        "84eca8a15b9bfe3b50e37c346b9d9fe1198338c5b4bbfceef1114b27059e226e", DATA / "voteview_all" / "HSall_votes.csv"),
}
PAP = (DATA / "agendas" / "rcv_voteview_v3_4.csv", "f4d95c110d7b55e07720ea050fd10dfbe82fe4d61d1c89c8dae5246c45504ac8")
PAP_URL = "https://minio.la.utexas.edu/compagendas/datasetfiles/rcv_voteview_v3_4.csv"   # listed on comparativeagendas.net/project/us/datasets

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def rows_to_118(p):
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f); head = next(r); i = head.index("congress")
        return sum(1 for row in r if int(row[i]) <= 118)

names = ["HSall_members.csv"] if "--members-only" in sys.argv else list(FILES)
for n in names:
    url, ref, dst = FILES[n]; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print("downloading", url); urllib.request.urlretrieve(url, dst)
    h = sha(dst)
    if h == ref: print(f"{n}: sha256 matches the paper's copy")
    else: print(f"{n}: sha256 differs from the paper's copy (Voteview updates its files); rows in Congresses <= 118: {rows_to_118(dst)}")
if "--members-only" not in sys.argv:
    p, ref = PAP
    if not p.exists(): print(f"PAP file missing: download {PAP_URL} into {p.parent} (README, 'Data'); sha256 expected {ref}")
    else: print("PAP:", "sha256 matches" if sha(p) == ref else "sha256 DIFFERS from the paper's copy")

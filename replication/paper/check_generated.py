#!/usr/bin/env python
"""Checks used by build_paper.sh.
  snapshot  record sha256 of the generated paper files (before the build) in paper/.generated_before.json
  seals     check post/members_by_era and post/benchmark outputs against the sha256 in their SEAL.json (stops on a mismatch)
  compare   compare the rebuilt files with the snapshot and print identical / different (exit 1 if any differ)
Usage: python paper/check_generated.py snapshot|seals|compare"""
import hashlib, json, sys
from pathlib import Path
P = Path(__file__).resolve().parent; R = P.parent; SNAP = P / ".generated_before.json"
GEN = ["abstract.md", "introduction.md", "intro_values.json", "manuscript.md", "figures/fig1_map.png", "figures/fig2_profiles.png",
       "figures/fig3_packages.png"] + [f"sections/{p.name}" for p in sorted((P / "sections").glob("*.md")) if not p.name.endswith(".template.md")]
POST = {"post/members_by_era": ["members_by_era.json", "members_table.csv"], "post/benchmark": ["benchmark_recall.json"]}
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "snapshot":
    json.dump({g: h(P / g) for g in GEN}, open(SNAP, "w"), indent=1); print(f"snapshot: {len(GEN)} generated files hashed")
elif mode == "seals":
    for d, files in POST.items():
        seal = json.load(open(R / d / "SEAL.json"))["sha256"]
        for f in files:
            assert h(R / d / f) == seal[f], f"{d}/{f} differs from its SEAL.json"
    print("seals: post/ outputs match SEAL.json")
elif mode == "compare":
    before = json.load(open(SNAP)); diff = [g for g in GEN if before.get(g) != h(P / g)]
    print(f"compare: {len(GEN) - len(diff)} of {len(GEN)} generated files byte-identical to the shipped versions")
    for g in diff: print("  different:", g)
    sys.exit(1 if diff else 0)
else:
    sys.exit(__doc__)

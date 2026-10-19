#!/usr/bin/env python
"""Path A guard: the scripts skip a stage when its output file exists (so an interrupted run can resume), and the package ships the
model outputs that path B reads at those same paths. Before a full rerun, this moves the shipped model outputs of the selected models
to _shipped_intermediates/ (same relative paths), so every GPU stage actually runs. Other models' outputs stay in place, because the
joint selection steps read every model's coordinates. Roll-call selections and the fixed A12 codes (out/s4b, out/s4c) stay in place.
Usage: python clear_intermediates.py qwen [llama olmo]   (run_all.sh calls it once; restore with --restore)"""
import json, shutil, sys
from pathlib import Path
R = Path(__file__).resolve().parent; AWAY = R / "_shipped_intermediates"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
led = json.load(open(R / "COPY_LEDGER.json", encoding="utf-8"))["intermediate"]
mine = [f for f in led if any(f.startswith(f"STUDY_D_quadrant/out/{m}/") or f.startswith(f"STUDY_L_battery/out/{m}/") for m in args)]
moved = 0
for f in mine:
    src, dst = (AWAY / f, R / f) if "--restore" in sys.argv else (R / f, AWAY / f)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst)); moved += 1
print(("restored" if "--restore" in sys.argv else "moved aside"), moved, "shipped files for", args)

#!/usr/bin/env python
"""Path A guard: the A12 text codes (out/s4b and out/s4c coding_codes.csv) are fixed data keyed to the rows of the original selections.
After s4b_select.py and s4c_select.py rerun from a fresh Voteview download, their outputs must be byte-identical to the shipped ones;
otherwise stop (the codes would not align)."""
import hashlib, json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
ref = json.load(open(R / "COPY_LEDGER.json", encoding="utf-8"))["intermediate"]
bad = [f for f in ("STUDY_D_quadrant/out/s4b/selection.csv", "STUDY_D_quadrant/out/s4c/selection.csv", "STUDY_D_quadrant/out/s4c/coding_sheet.csv")
       if hashlib.sha256(open(R / f, "rb").read()).hexdigest() != ref[f]["sha256"]]
print("selections identical to the coded selections" if not bad else f"DIFFERS from the coded selection (A12 codes would not align): {bad}")
sys.exit(1 if bad else 0)

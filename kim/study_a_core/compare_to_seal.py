#!/usr/bin/env python
"""Compare a fresh Study A core run with the sealed values (release_core/RELEASE.json), tier 2.

The value definitions are read from release_core/reproduce_core.py (one source, by AST; that script is not executed).
Tolerance rule (fixed 2026-10-05, before any fresh run; revised the same day, still before any fresh run, for near-zero values):
a value 'holds' if |new - sealed| <= max(0.05, 0.30 * |sealed|) and it has the sealed sign, except for the values whose claim is
'about zero' (NULL_CLAIMS below: employee-number control, response to real roll-call change, Southern-rural baseline change), where
the sign is not required. Sign stays required for small but directional values such as the per-year calendar drifts. Judge AUCs must stay >= 0.80. Output: <out>/compare_to_seal.csv and a summary line.
Usage: python compare_to_seal.py --out <run folder>
"""
import argparse, ast, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
HERE = Path(__file__).resolve().parent; RC = HERE.parent / "release_core"; OUT = Path(args.out)
ns = {"json": json}
for node in ast.parse((RC / "reproduce_core.py").read_text(encoding="utf-8")).body:
    if (isinstance(node, ast.FunctionDef) and node.name in ("J", "get", "peak_year")) or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "V"):
        exec(compile(ast.Module([node], []), "reproduce_core", "exec"), ns)
V = ns["V"]; sealed = json.loads((RC / "RELEASE.json").read_text(encoding="utf-8"))["values"]
NULL_CLAIMS = {"s2_m2_employee_control_ratio", "s3_real_drift_internal_np1", "s5_rep_change_south_rural"}
rows = []
for k, (step, claim, fn) in V.items():
    try: new = float(fn(OUT))
    except Exception as e: rows.append(dict(key=k, step=step, claim=claim, sealed=sealed[k]["value"], new=None, holds=False, note=f"missing: {e}")); continue
    s = sealed[k]["value"]
    near_zero = k in NULL_CLAIMS
    holds = (new >= 0.80) if k.startswith("judge_auc") else (near_zero or new * s > 0) and abs(new - s) <= max(0.05, 0.30 * abs(s))
    rows.append(dict(key=k, step=step, claim=claim, sealed=s, new=new, holds=bool(holds), note=""))
T = pd.DataFrame(rows); T.to_csv(OUT / "compare_to_seal.csv", index=False)
print(T[["key", "sealed", "new", "holds"]].to_string(index=False))
print(f"\n{int(T.holds.sum())}/{len(T)} sealed values hold within the tier-2 tolerance (see {OUT / 'compare_to_seal.csv'})")

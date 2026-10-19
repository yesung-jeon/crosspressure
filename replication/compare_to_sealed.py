#!/usr/bin/env python
"""Compare reproduced outputs with the sealed outputs of the paper (sealed/ holds byte copies of the original files).

Rows are the numbers behind acts 1-4 of the argument (see README, "What is reproduced"). Rules were fixed before any reproduction run:
  --path B (analysis from intermediates, CPU): every number equal to the sealed value after rounding to 3 decimals; verdict flags equal.
  --path A (full rerun from raw data, GPU, FP8): FP8 batched inference is not bit-reproducible, so
     rows with a 95% interval: |reproduced - sealed| <= 0.25 x half-width of the sealed interval, and the same verdict
                               (pre-registered 'supported' flag where it exists; otherwise the same sign and the same 0-exclusion);
     descriptive shares, correlations and gaps: |difference| <= 0.05;
     placement counts (out of 15): |difference| <= 2;
     patterns: in every model aligned recall (EL-SP, ER-ST) > cross recall (EL-ST, ER-SP), and economic party gap > social party gap.
Usage: python compare_to_sealed.py --path B [--models qwen llama olmo]   -> prints a table, writes REPRODUCTION_path<B|A>.json"""
import argparse, json
from pathlib import Path

R = Path(__file__).resolve().parent; S = R / "sealed"
ap = argparse.ArgumentParser(); ap.add_argument("--path", choices=["A", "B"], required=True)
ap.add_argument("--models", nargs="+", default=["qwen", "llama", "olmo"]); a = ap.parse_args()
Q4 = ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]

def load(root, rel): return json.load(open(root / rel, encoding="utf-8"))
def dsub(m, base): return f"STUDY_D_quadrant/out/{base}_olmo" if m == "olmo" else f"STUDY_D_quadrant/out/{base}"

def rows_for(m):
    """(row id, act, kind, getter(root) -> value or dict(est, ci95, supported))"""
    out = []
    an = lambda r: load(r, dsub(m, "s4") + "/anchors.json")[m]
    out += [(f"{m} A3 r(x, DW-NOMINATE dim1)", 1, "desc", lambda r: an(r)["r_x_dim1_by_congress_mean"]),
            (f"{m} A4 r(x, economic scale)", 1, "desc", lambda r: an(r)["r_x_economic_scale_within_cc"][0]),
            (f"{m} A4 r(y, abortion scale)", 1, "desc", lambda r: an(r)["r_y_abortion_scale_within_cc"][0]),
            (f"{m} A6 party gap x", 1, "desc", lambda r: an(r)["party_gap_x_d"]),
            (f"{m} A6 party gap y", 1, "desc", lambda r: an(r)["party_gap_y_d"])]
    for q in Q4:
        out.append((f"{m} A9 recall {q}", 2, "desc", lambda r, q=q: load(r, dsub(m, "s4") + "/vote_vs_map.json")[m]["recall_by_vote_quadrant"][q]))
    for q in Q4:
        out.append((f"{m} A12 placed {q} (/15)", 2, "count", lambda r, q=q: load(r, dsub(m, "s4c") + "/S4C.json")[m][q]["placed"]))   # A12 = S4C (15 per quadrant)
    for h, lab in (("H3_C2", "A'3 H3 consistency"), ("H4_C3hit", "A'4 H4 quadrant hit"), ("H1_C1a", "A'1 H1 choice entropy")):
        out.append((f"{m} {lab}", 3, "ci", lambda r, h=h: load(r, dsub(m, "s5") + "/S5_RESULTS.json")[m]["hyp"][h]))
    LB = lambda r: load(r, f"STUDY_L_battery/out/{m}/RESULTS.json")["blockB_natural_cross_pressure"]["tests"]
    out += [(f"{m} B1 reversed - party-like entropy, Democrat", 4, "ci", lambda r: LB(r)["B1_Democrat"]),
            (f"{m} B2 reversed - party-like entropy, Republican", 4, "ci", lambda r: LB(r)["B2_Republican"]),
            (f"{m} B5 economic share, Democrat (descriptive)", 4, "desc", lambda r: LB(r)["economic_dominance_share_by_party"]["Democrat"]),
            (f"{m} B5 economic share, Republican (descriptive)", 4, "desc", lambda r: LB(r)["economic_dominance_share_by_party"]["Republican"])]
    NP = lambda r: (lambda i: {"est": i["est"], "ci95": i["ci95_pigeonhole"]})(load(r, "post/no_persona/no_persona_package_effect.json" if r == R
                                                                                    else "review/correction_2026-10-16c/no_persona_package_effect.json")[m]["interval_2026_10_18"])
    out.append((f"{m} no-persona reversed - party-like (two-way item bootstrap)", 4, "ci", NP))
    return out

def verdict(v):
    if "supported" in v: return bool(v["supported"])
    lo, hi = v["ci95"]; return ("+" if v["est"] > 0 else "-") + ("excl0" if lo > 0 or hi < 0 else "incl0")

res, ok_all = [], True
for m in a.models:
    for rid, act, kind, get in rows_for(m):
        try: new, old = get(R), get(S)
        except (FileNotFoundError, KeyError) as e: res.append(dict(row=rid, act=act, status="MISSING", note=repr(e))); ok_all = False; continue
        if kind == "ci":
            d = abs(new["est"] - old["est"]); hw = (old["ci95"][1] - old["ci95"][0]) / 2
            if a.path == "B":
                ok = all(round(x, 3) == round(y, 3) for x, y in zip([new["est"], *new["ci95"]], [old["est"], *old["ci95"]])) and verdict(new) == verdict(old)
            else: ok = d <= 0.25 * hw and verdict(new) == verdict(old)
            res.append(dict(row=rid, act=act, sealed=old["est"], reproduced=new["est"], abs_diff=d, sealed_ci95=old["ci95"], reproduced_ci95=new["ci95"],
                            verdict_sealed=verdict(old), verdict_reproduced=verdict(new), status="PASS" if ok else "FAIL"))
        else:
            d = abs(float(new) - float(old)); tol = {"desc": 0.05, "count": 2}[kind]
            ok = round(float(new), 3) == round(float(old), 3) if a.path == "B" else d <= tol
            res.append(dict(row=rid, act=act, sealed=old, reproduced=new, abs_diff=d, status="PASS" if ok else "FAIL"))
        ok_all &= res[-1]["status"] == "PASS"
pat = []
for m in a.models:
    try:
        rec = load(R, dsub(m, "s4") + "/vote_vs_map.json")[m]["recall_by_vote_quadrant"]; an_ = load(R, dsub(m, "s4") + "/anchors.json")[m]
        p = min(rec["EL-SP"], rec["ER-ST"]) > max(rec["EL-ST"], rec["ER-SP"]) and an_["party_gap_x_d"] > an_["party_gap_y_d"]
    except (FileNotFoundError, KeyError): p = False
    pat.append(dict(model=m, aligned_recall_gt_cross_and_gap_x_gt_y=p)); ok_all &= p
for r in res:
    s = r.get("sealed"); n = r.get("reproduced")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"{r['status']:7s} act {r['act']}  {r['row']:60s} sealed {fmt(s):>8s}  reproduced {fmt(n):>8s}" if "sealed" in r else f"{r['status']} {r['row']} {r['note']}")
for p in pat: print(("PASS   " if p["aligned_recall_gt_cross_and_gap_x_gt_y"] else "FAIL   ") + f"pattern {p['model']}: aligned recall > cross recall; party gap x > y")
n_pass = sum(r["status"] == "PASS" for r in res)
print(f"\npath {a.path}: {n_pass}/{len(res)} rows PASS; patterns {sum(p['aligned_recall_gt_cross_and_gap_x_gt_y'] for p in pat)}/{len(pat)}; overall {'PASS' if ok_all else 'FAIL'}")
json.dump({"path": a.path, "models": a.models, "overall": "PASS" if ok_all else "FAIL", "rows": res, "patterns": pat},
          open(R / f"REPRODUCTION_path{a.path}.json", "w", encoding="utf-8"), indent=1, default=float)

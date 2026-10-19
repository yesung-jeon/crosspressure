#!/usr/bin/env python
"""Fill the 'Results of our reproduction' block of README.md from REPRODUCTION_path*.json (numbers are never typed).
Usage: python fill_readme.py"""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parent
blocks = []
for path in ("B", "A"):
    f = R / f"REPRODUCTION_path{path}.json"
    if not f.exists(): continue
    J = json.load(open(f, encoding="utf-8")); rows = J["rows"]; n = sum(r["status"] == "PASS" for r in rows)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    ci = lambda r: f" [{fmt(r['reproduced_ci95'][0])}, {fmt(r['reproduced_ci95'][1])}]" if "reproduced_ci95" in r else ""
    blocks.append(f"**Path {path}** ({', '.join(J['models'])}): {n} of {len(rows)} rows pass; "
                  f"patterns {sum(p['aligned_recall_gt_cross_and_gap_x_gt_y'] for p in J['patterns'])} of {len(J['patterns'])}; overall {J['overall']}.\n")
    blocks.append("| Act | Row | Sealed | Reproduced | Status |\n|---|---|---|---|---|")
    blocks += [f"| {r['act']} | {r['row']} | {fmt(r.get('sealed', ''))} | {fmt(r.get('reproduced', ''))}{ci(r)} | {r['status']} |" for r in rows]
    blocks.append("")
# Path A layer note: the layer the rerun's rule chose (its rule.json, copied as REPRODUCTION_pathA_qwen_rule.json) against the shipped
# rule.json, and the rerun with the shipped layer forced (REPRODUCTION_pathA_qwen_layer20.json). Written only if those files exist.
rr, l20 = R / "REPRODUCTION_pathA_qwen_rule.json", R / "REPRODUCTION_pathA_qwen_layer20.json"
if (R / "REPRODUCTION_pathA.json").exists() and rr.exists() and l20.exists():
    lr = json.load(open(rr, encoding="utf-8"))["common_layer"]
    ls = json.load(open(R / "STUDY_D_quadrant" / "out" / "qwen" / "rule.json", encoding="utf-8"))["common_layer"]
    J = json.load(open(l20, encoding="utf-8")); rows = J["rows"]; n = sum(r["status"] == "PASS" for r in rows)
    fails = [r["row"] for r in json.load(open(R / "REPRODUCTION_pathA.json", encoding="utf-8"))["rows"] if r["status"] != "PASS"]
    still = [r for r in rows if r["status"] != "PASS"]; fixed = [f for f in fails if f not in {r["row"] for r in still}]
    same_verdict = [r["row"] for r in still if "verdict_sealed" in r and r["verdict_sealed"] == r.get("verdict_reproduced")]
    blocks.append(f"In this Path A rerun the layer rule selected layer {lr} for Qwen, while the sealed results use layer {ls}. "
                  f"Rerunning with layer {ls} forced, {n} of {len(rows)} rows pass: {'; '.join(fixed) or 'no row'} then match"
                  + (f", while {'; '.join(r['row'] for r in still)} stays outside the tolerance" if still else "")
                  + (f" with the same preregistered verdict as the sealed result ({'; '.join(same_verdict)})" if same_verdict else "")
                  + ". Reruns on the same machine reproduce the rerun's own values exactly.\n")
md = (R / "README.md").read_text(encoding="utf-8")
body = "\n".join(blocks) if blocks else "No reproduction has been run yet."
md = re.sub(r"(<!-- BEGIN GENERATED: fill_readme.py -->\n).*?(<!-- END GENERATED -->)", lambda m: m.group(1) + body + "\n" + m.group(2), md, flags=re.S)
(R / "README.md").write_text(md, encoding="utf-8"); print("README results block filled:", len(blocks), "lines")

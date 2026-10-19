#!/usr/bin/env python
"""Descriptive benchmark checks N1 and N2 (council 2026-10-19; plan written before computing: kim/review/council_2026-10-19/N1_N2_PLAN.md).
Same rows as the sealed vote_vs_map comparison (Democratic and Republican member-Congress rows from the 93rd Congress on with both voting
scales; the last such Congress is the 117th). Voting quadrants as in STUDY_D_quadrant/code/s4_quadrant.py.
N1: a DW-NOMINATE map (x = dim1, y = dim2 oriented within Congress and chamber to correlate positively with the abortion scale), split at
    medians within Congress and chamber; recall by voting quadrant and pooled.
N2: per model, recall among rows whose party the model names correctly, and the recognition rate of aligned and cross-pressured rows.
Asserts that the all-row model recall equals the sealed vote_vs_map.json. Output: post/benchmark/benchmark_recall.json
Usage: python post/benchmark/benchmark_recall.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[2]; D = R / "STUDY_D_quadrant"; S = R / "sealed" / "STUDY_D_quadrant" / "out"; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(D / "code"))
import dcommon as dc  # noqa: E402
M = ["qwen", "llama", "olmo"]; Q = ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]; AL, CR = Q[:2], Q[2:]
vm = {"qwen": json.load(open(S / "s4" / "vote_vs_map.json"))["qwen"], "llama": json.load(open(S / "s4" / "vote_vs_map.json"))["llama"],
      "olmo": json.load(open(S / "s4_olmo" / "vote_vs_map.json"))["olmo"]}
quad = lambda hx, hy: pd.Series(np.where(hx, "ER-", "EL-"), index=hx.index) + np.where(hy, "ST", "SP")
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"])]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index()

def rows(key):
    m = pd.read_csv(D / "out" / key / "s3" / "temporal_member_scores.csv"); m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m["cc"] = m.congress.astype(str) + m.chamber
    v = m.merge(w, on=["congress", "chamber", "icpsr"], how="left", validate="one_to_one")
    v = v[(v.congress >= 93) & v.party.isin(["Democrat", "Republican"])].dropna(subset=["economic", "abortion"]).copy()
    for c in ["x", "y", "economic", "abortion"]:
        v[f"hi_{c}"] = v[c] > v.groupby("cc")[c].transform("median")
    v["vote_q"] = quad(v.hi_economic, v.hi_abortion); v["map_q"] = quad(v.hi_x, v.hi_y)
    return v

def recall(v, col):
    r = {q: float((v[v.vote_q == q][col] == q).mean()) for q in Q}
    r["aligned_pooled"] = float((v[v.vote_q.isin(AL)][col] == v[v.vote_q.isin(AL)].vote_q).mean())
    r["cross_pooled"] = float((v[v.vote_q.isin(CR)][col] == v[v.vote_q.isin(CR)].vote_q).mean())
    r["n"] = int(len(v)); return r

out = {"note": "descriptive benchmark checks added 2026-10-19 after sealing (council N1, N2); no tests", "N1_dwnominate": None, "N2_recognized": {}}
for key in M:
    v = rows(key)
    a = recall(v, "map_q")
    assert v.shape[0] == vm[key]["n"] and all(abs(a[q] - vm[key]["recall_by_vote_quadrant"][q]) < 1e-12 for q in Q), key
    k = v[v.known.astype(bool)]
    out.setdefault("r_x_y_within_cc", {})[key] = float(np.corrcoef(v.x - v.groupby("cc").x.transform("mean"), v.y - v.groupby("cc").y.transform("mean"))[0, 1])
    out["N2_recognized"][key] = {"all_rows": a, "recognized_rows": recall(k, "map_q"),
                                 "recognition_rate": {"aligned": float(v[v.vote_q.isin(AL)].known.astype(bool).mean()),
                                                      "cross_pressured": float(v[v.vote_q.isin(CR)].known.astype(bool).mean())}}
    if key == "qwen":   # DW-NOMINATE scores are the same in every model's file; the rows are identical across models
        sign = v.groupby("cc").apply(lambda g: np.sign(np.corrcoef(g.dim2, g.abortion)[0, 1]) or 1.0, include_groups=False)
        v["d2"] = v.dim2 * v.cc.map(sign)
        v["hi_d1"] = v.dim1 > v.groupby("cc").dim1.transform("median"); v["hi_d2"] = v.d2 > v.groupby("cc").d2.transform("median")
        v["nom_q"] = quad(v.hi_d1, v.hi_d2)
        out["N1_dwnominate"] = {"recall": recall(v, "nom_q"), "dim2_sign_flipped_cells": int((sign < 0).sum()), "cells": int(len(sign)),
                                "r_dim1_economic_within_cc": float(np.corrcoef(v.dim1 - v.groupby("cc").dim1.transform("mean"), v.economic - v.groupby("cc").economic.transform("mean"))[0, 1]),
                                "r_dim1_dim2_within_cc": float(np.corrcoef(v.dim1 - v.groupby("cc").dim1.transform("mean"), v.d2 - v.groupby("cc").d2.transform("mean"))[0, 1]),
                                "r_dim2_abortion_within_cc": float(np.corrcoef(v.d2 - v.groupby("cc").d2.transform("mean"), v.abortion - v.groupby("cc").abortion.transform("mean"))[0, 1])}
json.dump(out, open(OUT / "benchmark_recall.json", "w", encoding="utf-8"), indent=1)
n1 = out["N1_dwnominate"]["recall"]
print("N1 DW-NOMINATE pooled aligned %.3f cross %.3f; by quadrant %s" % (n1["aligned_pooled"], n1["cross_pooled"], {q: round(n1[q], 3) for q in Q}))
for key in M:
    r = out["N2_recognized"][key]
    print(key, "all: al %.3f cr %.3f | recognized: al %.3f cr %.3f | recognition al %.3f cr %.3f" % (
        r["all_rows"]["aligned_pooled"], r["all_rows"]["cross_pooled"], r["recognized_rows"]["aligned_pooled"], r["recognized_rows"]["cross_pooled"],
        r["recognition_rate"]["aligned"], r["recognition_rate"]["cross_pressured"]))

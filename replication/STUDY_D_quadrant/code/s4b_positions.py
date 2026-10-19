#!/usr/bin/env python
"""S4b step 2 (PLAN addendum A7): map positions of the vote-selected members in each model (median-origin quadrant over the same
Congresses; within Congress x chamber percentile of x and y), joined with the step-1 codes. -> out/s4b/S4B.json, s4b_rows.csv"""
import argparse
import numpy as np, pandas as pd
import dcommon as dc

ap = argparse.ArgumentParser()   # A12: selection and codes are always read from the sealed out/s4b; --models olmo --out s4b_olmo adds OLMo positions
ap.add_argument("--models", nargs="+", default=["qwen", "llama"], choices=list(dc.MODELS)); ap.add_argument("--out", default="s4b")
a = ap.parse_args(); SEL = a.models
O = dc.OUT / "s4b"; OO = dc.OUT / a.out; OO.mkdir(parents=True, exist_ok=True)
sel = pd.read_csv(O / "selection.csv"); codes = pd.read_csv(O / "coding_codes.csv"); sheet = pd.read_csv(O / "coding_sheet.csv")[["row", "icpsr"]]
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()
JR = dc.s4_selection_rows()
d = sel.merge(sheet, on="icpsr", validate="one_to_one").merge(codes, on="row", validate="one_to_one")
exp_e = d.vote_q.str[:2].map({"EL": "L", "ER": "R"}); exp_s = d.vote_q.str[3:].map({"SP": "P", "ST": "T"})
def status(e, s, ee, es):
    codes = [(e, ee), (s, es)]
    if any(c in ("L", "R", "P", "T") and c != x for c, x in codes): return "contradicted"
    n = sum(c == x for c, x in codes)
    return {2: "confirmed both axes", 1: "confirmed one axis"}.get(n, "no evidence")
d["text_status"] = [status(e, s, ee, es) for e, s, ee, es in zip(d.econ_code, d.social_code, exp_e, exp_s)]
R = {}
for key in SEL:
    m = pd.read_csv(dc.OUT / key / "s3" / "temporal_member_scores.csv"); m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m = m[m.party.isin(["Democrat", "Republican"]) & m.known.astype(bool)].copy(); m["cc"] = m.congress.astype(str) + m.chamber   # A14: recalled D/R rows, as S4
    for c in ["x", "y"]:
        m[f"{c}m"] = m[c] - m.groupby("cc")[c].transform("median"); m[f"{c}p"] = m.groupby("cc")[c].rank(pct=True)
    rows = m.merge(JR, on=["congress", "chamber", "icpsr"])          # A13: exactly the selection's rows (both scales, recalled by Qwen and Llama)
    rows = rows[rows.known.astype(bool)]                              # for OLMo: those of them OLMo also recalls
    g = rows.groupby("icpsr").agg(xm=("xm", "mean"), ym=("ym", "mean"), xp=("xp", "mean"), yp=("yp", "mean"), n=("xm", "size"))
    q = pd.Series(np.where(g.xm > 0, "ER-", "EL-"), index=g.index) + np.where(g.ym > 0, "ST", "SP")
    d[f"{key}_q"] = d.icpsr.map(q); d[f"{key}_xp"] = d.icpsr.map(g.xp); d[f"{key}_yp"] = d.icpsr.map(g.yp)
    nJ = JR.groupby("icpsr").size(); allq = q[q.index.map(nJ).fillna(0) >= 2].value_counts(normalize=True)   # A14: selection eligibility (>= 2 joint Congresses), then >= 1 usable row of this model
    R[key] = {}
    for vq, h in d.groupby("vote_q"):
        hc = h[h.text_status.str.startswith("confirmed both")]     # A15: confirmed = both axes; one axis reported separately
        h1 = h[h.text_status.str.startswith("confirmed one")]
        R[key][vq] = {"n": int(len(h)), "n_recalled": int(h[f"{key}_q"].notna().sum()), "placed_in_vote_quadrant": int((h[f"{key}_q"] == vq).sum()), "n_text_confirmed": int(len(hc)),
                      "confirmed_placed": int((hc[f"{key}_q"] == vq).sum()), "one_axis_placed": int((h1[f"{key}_q"] == vq).sum()), "n_one_axis": int(len(h1)), "map_quadrants": h[f"{key}_q"].value_counts().to_dict(),
                      "mean_x_pct": float(h[f"{key}_xp"].mean()), "mean_y_pct": float(h[f"{key}_yp"].mean()), "chance_share": float(allq.get(vq, 0))}
dc.atomic_csv(d, OO / "s4b_rows.csv"); dc.atomic_json(R, OO / "S4B.json")
pd.set_option("display.width", 220)
print(d[["vote_q", "name", "party", "text_status"] + [f"{k}_{c}" for k in SEL for c in ("q", "xp", "yp")]].round(2).to_string(index=False))
for k in R:
    print(k, {q: (v["placed_in_vote_quadrant"], v["n"], "recalled", v["n_recalled"], "confirmed", v["confirmed_placed"], v["n_text_confirmed"], "chance", round(v["chance_share"], 2)) for q, v in R[k].items()})

#!/usr/bin/env python
"""S4c step 2 (PLAN addendum A8): map positions of the 60 vote-selected members (S4b codes reused for people coded there), per
quadrant and model; deepest and random apart; percentiles against the party average. -> out/s4c/S4C.json, s4c_rows.csv"""
import argparse
import numpy as np, pandas as pd
import dcommon as dc

ap = argparse.ArgumentParser()   # A12: selection and codes always from the sealed out/s4b, out/s4c; --models olmo --out s4c_olmo adds OLMo positions
ap.add_argument("--models", nargs="+", default=["qwen", "llama"], choices=list(dc.MODELS)); ap.add_argument("--out", default="s4c")
a = ap.parse_args(); SEL = a.models
O = dc.OUT / "s4c"; OO = dc.OUT / a.out; OO.mkdir(parents=True, exist_ok=True)
sel = pd.read_csv(O / "selection.csv")
cb = pd.read_csv(dc.OUT / "s4b" / "coding_sheet.csv")[["row", "icpsr"]].merge(pd.read_csv(dc.OUT / "s4b" / "coding_codes.csv"), on="row")
cc_ = pd.read_csv(O / "coding_sheet.csv")[["row", "icpsr"]].merge(pd.read_csv(O / "coding_codes.csv"), on="row")
codes = pd.concat([cb, cc_]).drop(columns="row"); assert codes.icpsr.is_unique
d = sel.merge(codes, on="icpsr", how="left", validate="one_to_one"); assert d.econ_code.notna().all()
exp_e = d.vote_q.str[:2].map({"EL": "L", "ER": "R"}); exp_s = d.vote_q.str[3:].map({"SP": "P", "ST": "T"})
def status(e, s, ee, es):
    pairs = [(e, ee), (s, es)]
    if any(c in ("L", "R", "P", "T") and c != x for c, x in pairs): return "contradicted"
    return {2: "confirmed both", 1: "confirmed one"}.get(sum(c == x for c, x in pairs), "no evidence")
d["text"] = [status(e, s, ee, es) for e, s, ee, es in zip(d.econ_code, d.social_code, exp_e, exp_s)]
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()
JR = dc.s4_selection_rows()
R = {"party_pct": {}}
for key in SEL:
    m = pd.read_csv(dc.OUT / key / "s3" / "temporal_member_scores.csv"); m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m = m[m.party.isin(["Democrat", "Republican"]) & m.known.astype(bool)].copy(); m["cc"] = m.congress.astype(str) + m.chamber   # A14: recalled D/R rows, as S4
    for c in ["x", "y"]:
        m[f"{c}m"] = m[c] - m.groupby("cc")[c].transform("median"); m[f"{c}p"] = m.groupby("cc")[c].rank(pct=True)
    rows = m.merge(JR, on=["congress", "chamber", "icpsr"]); rows = rows[rows.known.astype(bool)]   # A13: the selection's rows; OLMo: those it also recalls
    g = rows.groupby("icpsr").agg(xm=("xm", "mean"), ym=("ym", "mean"), xp=("xp", "mean"), yp=("yp", "mean"))
    q = pd.Series(np.where(g.xm > 0, "ER-", "EL-"), index=g.index) + np.where(g.ym > 0, "ST", "SP")
    d[f"{key}_q"] = d.icpsr.map(q); d[f"{key}_xp"] = d.icpsr.map(g.xp); d[f"{key}_yp"] = d.icpsr.map(g.yp)
    R["party_pct"][key] = rows.groupby("party")[["xp", "yp"]].mean().round(3).to_dict(orient="index")
    R[key] = {}
    for vq, h in d.groupby("vote_q"):
        hc = h[h.text.str.startswith("confirmed both")]     # A15: confirmed = both axes; one axis reported separately
        h1 = h[h.text.str.startswith("confirmed one")]
        R[key][vq] = {"placed": int((h[f"{key}_q"] == vq).sum()), "n": int(len(h)), "n_recalled": int(h[f"{key}_q"].notna().sum()),
                      "placed_deepest": int((h[h.how == "deepest"][f"{key}_q"] == vq).sum()), "placed_random": int((h[h.how == "random"][f"{key}_q"] == vq).sum()),
                      "confirmed_placed": int((hc[f"{key}_q"] == vq).sum()), "one_axis_placed": int((h1[f"{key}_q"] == vq).sum()), "n_one_axis": int(len(h1)), "n_confirmed": int(len(hc)),
                      "mean_x_pct": round(float(h[f"{key}_xp"].mean()), 3), "mean_y_pct": round(float(h[f"{key}_yp"].mean()), 3),
                      "map_quadrants": h[f"{key}_q"].value_counts().to_dict()}
R["text_status"] = d.groupby(["vote_q", "text"]).size().unstack(fill_value=0).to_dict(orient="index")
dc.atomic_csv(d, OO / "s4c_rows.csv"); dc.atomic_json(R, OO / "S4C.json")
pd.set_option("display.width", 230)
print(d.sort_values(["vote_q", "how"])[["vote_q", "how", "name", "party", "text"] + [f"{k}_{c}" for k in SEL for c in ("q", "xp", "yp")]].round(2).to_string(index=False))
for k in SEL:
    print(k, {q: (v["placed"], v["n"], "recalled", v["n_recalled"], "deep", v["placed_deepest"], "rnd", v["placed_random"], "conf", v["confirmed_placed"], v["n_confirmed"], "x%", v["mean_x_pct"], "y%", v["mean_y_pct"]) for q, v in R[k].items()})
print("party pct", R["party_pct"]); print("text", R["text_status"])

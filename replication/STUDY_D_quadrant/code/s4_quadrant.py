#!/usr/bin/env python
"""S4 (PLAN.md, CPU): anchors, quadrants, the qualitative sample (blinded coding sheet + separate key), vote-vs-map quadrants.
Map axes as Kim: x = -econ_orth (right +), y = -social_orth (traditional +).
Outputs out/s4/: anchors.json, quadrants.json, members_<model>.csv, coding_sheet.csv, KEY_open_after_coding.csv, vote_vs_map.json"""
import argparse, hashlib, json
import numpy as np, pandas as pd
import dcommon as dc

ap = argparse.ArgumentParser()   # A12: defaults reproduce the sealed Qwen/Llama run; OLMo: --models olmo --out s4_olmo
ap.add_argument("--models", nargs="+", default=["qwen", "llama"], choices=list(dc.MODELS)); ap.add_argument("--out", default="s4")
a = ap.parse_args(); SEL = a.models
O = dc.OUT / a.out; O.mkdir(parents=True, exist_ok=True)
Q4 = ["EL-SP", "EL-ST", "ER-SP", "ER-ST"]
quad = lambda x, y: pd.Series(np.where(x > 0, "ER-", "EL-"), index=getattr(x, "index", None)) + np.where(y > 0, "ST", "SP")
vv = pd.read_csv(dc.MEMBERS, low_memory=False)[["congress", "icpsr", "bioname", "bioguide_id"]].drop_duplicates(["congress", "icpsr"])
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"])]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index()
kim = pd.read_csv(dc.CMAP / "results" / "temporal_member_scores.csv")
anchors, quads, vote_map, sample = {}, {}, {}, []

def demeaned_r(d, a, b, g):
    d = d[[a, b, g]].dropna(); x = d[a] - d.groupby(g)[a].transform("mean"); y = d[b] - d.groupby(g)[b].transform("mean")
    return float(np.corrcoef(x, y)[0, 1]), int(len(d))

for key in SEL:
    rule = json.load(open(dc.OUT / key / "rule.json"))
    m = pd.read_csv(dc.OUT / key / "s3" / "temporal_member_scores.csv")
    assert len(m) == 21288, len(m)
    m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m["cc"] = m.congress.astype(str) + m.chamber
    k = m[m.known.astype(bool)]
    per = k.groupby("congress").apply(lambda g: np.corrcoef(g.x, g.dim1)[0, 1], include_groups=False)
    j = m.merge(w, on=["congress", "chamber", "icpsr"], how="left", validate="one_to_one")
    A = {"layer": rule["common_layer"], "n_rows": int(len(m)), "recall": float(m.known.mean()),
         "recall_1947": float(m[m.congress == 80].known.mean()), "recall_2023": float(m[m.congress == 118].known.mean()),
         "r_x_dim1_by_congress_mean": float(per.mean()), "r_x_dim1_80": float(per.loc[80]), "r_x_dim1_118": float(per.loc[118]),
         "r_y_dim1_pooled_within_cc": demeaned_r(k, "y", "dim1", "cc")[0],
         "r_x_economic_scale_within_cc": demeaned_r(j[j.known.astype(bool)], "x", "economic", "cc"),
         "r_y_abortion_scale_within_cc": demeaned_r(j[j.known.astype(bool)], "y", "abortion", "cc"),
         "r_y_economic_scale_within_cc": demeaned_r(j[j.known.astype(bool)], "y", "economic", "cc"),
         "r_x_abortion_scale_within_cc": demeaned_r(j[j.known.astype(bool)], "x", "abortion", "cc"),
         "party_gap_x_d": float((k[k.party == "Republican"].x.mean() - k[k.party == "Democrat"].x.mean()) / k.x.std()),
         "party_gap_y_d": float((k[k.party == "Republican"].y.mean() - k[k.party == "Democrat"].y.mean()) / k.y.std())}
    # A15 (2026-10-16): party_gap_*_d divide by the overall recalled-sample SD (not Cohen's d); pooled within-party SD d added
    for c in ["x", "y"]:
        r_, d_ = k[k.party == "Republican"][c], k[k.party == "Democrat"][c]
        sp = np.sqrt(((len(r_) - 1) * r_.var() + (len(d_) - 1) * d_.var()) / (len(r_) + len(d_) - 2))
        A[f"party_gap_{c}_cohen_d_pooled_A15"] = float((r_.mean() - d_.mean()) / sp)
    if key == "qwen":
        jk = m.merge(kim, on=["congress", "icpsr"], suffixes=("", "_kim"), validate="one_to_one")
        A["r_with_kim_map_econ"] = float(np.corrcoef(jk.econ_orth, jk.econ_orth_kim)[0, 1]); A["r_with_kim_map_social"] = float(np.corrcoef(jk.social_orth, jk.social_orth_kim)[0, 1])
        A["recall_agreement_with_kim"] = float((jk.known.astype(bool) == jk.known_kim.astype(bool)).mean())
    anchors[key] = A
    # ---- member level (recalled D/R, >= 2 recalled Congresses) ----
    k = k[k.party.isin(["Democrat", "Republican"])].copy()
    k["xm"] = k.x - k.groupby("cc").x.transform("median"); k["ym"] = k.y - k.groupby("cc").y.transform("median")
    g = k.groupby("icpsr").agg(name=("name", "first"), party=("party", lambda s: s.mode().iloc[0]), chamber=("chamber", lambda s: "/".join(sorted(set(s)))),
                               state=("state", lambda s: "/".join(sorted(set(s)))), first=("year", "min"), last=("year", "max"), n=("year", "size"),
                               x=("x", "mean"), y=("y", "mean"), xm=("xm", "mean"), ym=("ym", "mean"), dim1=("dim1", "mean"), dim2=("dim2", "mean")).reset_index()
    g = g[g.n >= 2].copy()
    g["q_main"] = quad(g.x, g.y); g["q_median"] = quad(g.xm, g.ym)
    g["zx"], g["zy"] = g.xm / g.xm.std(), g.ym / g.ym.std(); g["depth"] = np.minimum(g.zx.abs(), g.zy.abs())   # addendum A1: median origin
    dc.atomic_csv(g, O / f"members_{key}.csv")
    quads[key] = {"n_members": int(len(g)), "main": g.q_main.value_counts().reindex(Q4, fill_value=0).to_dict(),
                  "median_origin": g.q_median.value_counts().reindex(Q4, fill_value=0).to_dict(), "agreement_main_vs_median": float((g.q_main == g.q_median).mean()),
                  "main_by_party": pd.crosstab(g.q_main, g.party).to_dict()}
    for q in Q4:
        h = g[g.q_median == q]      # addendum A1: sample on median-origin quadrants
        if len(h) <= 6:
            pick = [(i, "all") for i in h.icpsr]
        else:
            deep = h.sort_values("depth", ascending=False).icpsr.head(3).tolist()
            rest = np.sort(h[~h.icpsr.isin(deep)].icpsr.values)
            rnd = np.random.default_rng(dc.seed_for("D-S4", key, q) % 2**32).choice(rest, 3, replace=False).tolist()
            pick = [(i, "deepest") for i in deep] + [(i, "random") for i in rnd]
        for i, how in pick:
            r = h[h.icpsr == i].iloc[0]; sample.append(dict(model=key, quadrant=q, how=how, icpsr=int(i), xm=r.xm, ym=r.ym, x=r.x, y=r.y, q_main=r.q_main, party=r.party))
    # ---- vote vs map quadrants (93rd on; median splits within Congress x chamber) ----
    v = j[(j.congress >= 93) & j.party.isin(["Democrat", "Republican"])].dropna(subset=["economic", "abortion"]).copy()
    for c in ["x", "y", "economic", "abortion"]:
        v[f"hi_{c}"] = v[c] > v.groupby("cc")[c].transform("median")
    v["vote_q"] = quad(v.hi_economic.astype(int) - 0.5, v.hi_abortion.astype(int) - 0.5)
    v["map_q"] = quad(v.hi_x.astype(int) - 0.5, v.hi_y.astype(int) - 0.5)
    ct = pd.crosstab(v.vote_q, v.map_q).reindex(index=Q4, columns=Q4, fill_value=0)
    vote_map[key] = {"n": int(len(v)), "agreement": float((v.vote_q == v.map_q).mean()), "confusion_vote_rows_map_cols": ct.to_dict(orient="index"),
                     "recall_by_vote_quadrant": {q: float((v[v.vote_q == q].map_q == q).mean()) for q in Q4},
                     "known_only_agreement": float((v[v.known.astype(bool)].vote_q == v[v.known.astype(bool)].map_q).mean())}

S = pd.DataFrame(sample)
key_rows = S.copy()
people = S.drop_duplicates("icpsr")[["icpsr"]].copy()
info = []
for i in people.icpsr:
    rows = []
    for key in SEL:
        g = pd.read_csv(O / f"members_{key}.csv"); r = g[g.icpsr == i]
        if len(r): rows.append(r.iloc[0])
    r = rows[0]; bg = vv[vv.icpsr == i].bioguide_id.dropna()
    info.append(dict(icpsr=int(i), name=r["name"], chamber=r.chamber, state=r.state, years=f"{int(r['first'])}-{int(r['last'])+1}",
                     bioguide_id=bg.iloc[0] if len(bg) else "", bioname=vv[vv.icpsr == i].bioname.iloc[0]))
sheet = pd.DataFrame(info)
sheet["order"] = [hashlib.sha256(f"D-sheet|{i}".encode()).hexdigest() for i in sheet.icpsr]
sheet = sheet.sort_values("order").drop(columns="order").reset_index(drop=True); sheet.insert(0, "row", range(1, len(sheet) + 1))
for c in ["econ_code", "econ_quote", "social_code", "social_quote", "race_note", "source"]: sheet[c] = ""
dc.atomic_csv(sheet, O / "coding_sheet.csv"); dc.atomic_csv(key_rows, O / "KEY_open_after_coding.csv")
dc.atomic_json(anchors, O / "anchors.json"); dc.atomic_json(quads, O / "quadrants.json"); dc.atomic_json(vote_map, O / "vote_vs_map.json")
dc.log(f"S4: sample rows {len(S)}, people {len(sheet)}; anchors and quadrants written (key not printed)")

#!/usr/bin/env python
"""Step 17 - representation measures for the LLM as a representative of U.S. legislators (CPU; Kim's bf16 coordinates only).

Frame (author decision 2026-10-04): political representation. "Real" position = DW-NOMINATE dim1 (static) or Nokken-Poole dim1
(per Congress); "represented" position = Kim's projection (x economic, y social). Both are z-scored within era over all D/R
member-Congresses so slopes and distances share a scale. Main pairing: x ~ dim1 and y ~ dim1 (the social axis tracks dim1
more than dim2, E2); sensitivity: y ~ dim2.

1 Achen (1978) measures by party x era (and party overall), member-Congress rows, SE clustered by member:
    responsiveness = slope of z(model) on z(real) within the group (dyadic: distinguishes members inside the party?)
    proximity      = mean squared difference z(model) - z(real) after removing the group means (fit of within-party ordering)
    centrism       = SD of z(model) / SD of z(real) within the group (< 1: model compresses the party toward its centre)
   Unequal representation: Republican minus Democrat responsiveness (interaction), by era.
2 Dyadic vs collective: correlation of model and real positions at increasing aggregation, per Congress then averaged:
    members within party | all members | state delegations (state x chamber means) | party means over time (levels, differences)
3 Dynamic representation by party: (a) within member, member FE: model ~ NP dim1 x party + tenure x party;
   (b) party means: change in model position on change in NP dim1, slope and r by party (first differences, 38 changes).
Output: out/representation_summary.json, out/REPRESENTATION_RESULTS.md, out/representation_by_group.csv, out/fig_representation.png
Usage: python 17_representation.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}; f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
m = cm.members(); m = m[m.party.isin(["Democrat", "Republican"])].copy()
v = pd.read_csv(cm.VOTEVIEW, low_memory=False)
v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])[["congress", "icpsr", "nokken_poole_dim1"]]
m = m.merge(v, on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1"})
m["rep"] = (m.party == "Republican").astype(int)
for c in ("x", "y", "dim1", "dim2", "np1"):
    m[f"z_{c}"] = m.groupby("era")[c].transform(lambda s: (s - s.mean()) / s.std())
PAIRS = {"x~dim1": ("z_x", "z_dim1"), "y~dim1": ("z_y", "z_dim1"), "y~dim2": ("z_y", "z_dim2")}
R["n"] = dict(member_congresses=int(len(m)), members=int(m.icpsr.nunique()))

# ======================= 1. Achen measures =======================
rows, A = [], {}
for pk, (zm, zr) in PAIRS.items():
    for (era, party), g in list(m.groupby(["era", "party"])) + [(("all", p), g) for p, g in m.groupby("party")]:
        g = g.dropna(subset=[zm, zr])
        r = cm.fit_cr1(g, f"{zm} ~ {zr}", "icpsr")
        dm, dr = g[zm] - g[zm].mean(), g[zr] - g[zr].mean()
        rows.append(dict(pair=pk, era=era, party=party, n=int(len(g)), responsiveness=float(r.params[zr]), resp_se=float(r.bse[zr]),
                         within_party_r=float(np.corrcoef(g[zm], g[zr])[0, 1]), proximity_msd=float(((dm - dr) ** 2).mean()),
                         centrism_sd_ratio=float(g[zm].std() / g[zr].std())))
    gap = {}
    for era in ["all"] + sorted(m.era.unique()):
        g = m if era == "all" else m[m.era == era]
        r = cm.fit_cr1(g.dropna(subset=[zm, zr]), f"{zm} ~ {zr} * rep", "icpsr"); gap[era] = cm.coef(r, f"{zr}:rep")
    A[pk] = {"rep_minus_dem_responsiveness": gap}
G1 = pd.DataFrame(rows); G1.to_csv(OUT / "representation_by_group.csv", index=False)
A["table"] = G1.round(4).to_dict("records"); R["1_achen"] = A

# ======================= 2. dyadic vs collective =======================
D2 = {}
for pk, (zm, zr) in PAIRS.items():
    out = {}
    per = []
    for c, g in m.groupby("congress"):
        row = {"congress": c, "all_members": np.corrcoef(g[zm], g[zr])[0, 1]}
        for p, gp in g.groupby("party"):
            row[f"within_{p}"] = np.corrcoef(gp[zm], gp[zr])[0, 1] if len(gp) > 5 else np.nan
        sd = g.groupby(["state", "chamber"])[[zm, zr]].mean()
        row["state_delegations"] = np.corrcoef(sd[zm], sd[zr])[0, 1]
        per.append(row)
    per = pd.DataFrame(per)
    for k in ["within_Democrat", "within_Republican", "all_members", "state_delegations"]:
        out[k] = dict(mean_r=float(per[k].mean()), by_era={e: float(per[per.congress.between(a, b)][k].mean()) for e, (a, b) in
                                                          {"1947-79": (80, 96), "1981-2005": (97, 109), "2007-23": (110, 118)}.items()})
    pm = m.groupby(["congress", "party"])[[zm[2:], zr[2:]]].mean().reset_index()   # raw (unstandardized) party means over time
    lv, df_ = [], []
    for p, g in pm.groupby("party"):
        g = g.sort_values("congress"); lv.append(pearsonr(g[zm[2:]], g[zr[2:]])[0]); df_.append(pearsonr(g[zm[2:]].diff().dropna(), g[zr[2:]].diff().dropna())[0])
    out["party_means_over_time"] = dict(levels_r_by_party=dict(zip(sorted(pm.party.unique()), map(float, lv))), diff_r_by_party=dict(zip(sorted(pm.party.unique()), map(float, df_))))
    gap = pm.pivot(index="congress", columns="party", values=[zm[2:], zr[2:]]); gm = gap[(zm[2:], "Republican")] - gap[(zm[2:], "Democrat")]; gr = gap[(zr[2:], "Republican")] - gap[(zr[2:], "Democrat")]
    out["party_gap_over_time"] = dict(levels_r=float(pearsonr(gm, gr)[0]), diff_r=float(pearsonr(gm.diff().dropna(), gr.diff().dropna())[0]))
    D2[pk] = out
R["2_dyadic_vs_collective"] = D2

# ======================= 3. dynamic representation by party =======================
d = m.dropna(subset=["np1"]).copy(); d = d[d.groupby("icpsr").congress.transform("size") >= 2]
d["tenure"] = d.year - d.groupby("icpsr").year.transform("min"); d["np1_rep"], d["tenure_rep"] = d.np1 * d.rep, d.tenure * d.rep
D3 = {}
for yv in ("y", "x"):
    cols = [yv, "np1", "np1_rep", "tenure", "tenure_rep"]
    w = d[cols] - d.groupby("icpsr")[cols].transform("mean"); w["icpsr"] = d.icpsr.values
    r = cm.fit_cr1(w, f"{yv} ~ 0 + np1 + np1_rep + tenure + tenure_rep", "icpsr")
    D3[f"within_member_{yv}"] = {k: cm.coef(r, k) for k in ("np1", "np1_rep", "tenure", "tenure_rep")}
ps = m.groupby(["congress", "party"])[["x", "y", "np1"]].mean().reset_index()
for p, g in ps.groupby("party"):
    g = g.sort_values("congress")[["x", "y", "np1"]].diff().dropna()
    for yv in ("x", "y"):
        b = np.polyfit(g.np1, g[yv], 1)[0]; r_, pv = pearsonr(g.np1, g[yv])
        D3[f"party_mean_changes_{yv}_{p}"] = dict(slope=float(b), r=float(r_), p=float(pv), n=int(len(g)))
R["3_dynamic"] = D3
(OUT / "representation_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

# ======================= figure =======================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
t = G1[(G1.pair == "x~dim1") & (G1.era != "all")]; t2 = G1[(G1.pair == "y~dim1") & (G1.era != "all")]
eras = ["1947-79", "1981-2005", "2007-23"]; w_ = .2; xs = np.arange(3)
for i, (tt, lab) in enumerate(((t, "economic x"), (t2, "social y"))):
    for j, (p, c) in enumerate((("Democrat", "tab:blue"), ("Republican", "tab:red"))):
        q = tt[tt.party == p].set_index("era").reindex(eras)
        ax[0].bar(xs + (2 * i + j - 1.5) * w_, q.responsiveness, w_, yerr=1.96 * q.resp_se, color=c, alpha=.9 if i == 0 else .45,
                  label=f"{p}, {lab}")
ax[0].set_xticks(xs, eras); ax[0].axhline(0, color="k", lw=.6); ax[0].legend(fontsize=7); ax[0].set_ylabel("responsiveness (z model on z DW dim1, within party)")
ax[0].set_title("1  Dyadic responsiveness within party (Achen)", fontsize=10)
lv = ["within_Democrat", "within_Republican", "all_members", "state_delegations"]
for pk, mk in (("x~dim1", "o-"), ("y~dim1", "s--")):
    vals = [D2[pk][k]["mean_r"] for k in lv] + [D2[pk]["party_gap_over_time"]["levels_r"]]
    ax[1].plot(range(5), vals, mk, label=pk)
ax[1].set_xticks(range(5), ["members\nwithin D", "members\nwithin R", "all\nmembers", "state\ndelegations", "party gap\nover time"], fontsize=8)
ax[1].set_ylabel("correlation model vs DW-NOMINATE"); ax[1].legend(fontsize=8); ax[1].set_title("2  Collective yes, dyadic no", fontsize=10)
labs, vals, cols = [], [], []
for yv in ("x", "y"):
    for p, c in (("Democrat", "tab:blue"), ("Republican", "tab:red")):
        o = D3[f"party_mean_changes_{yv}_{p}"]; labs.append(f"{yv}, {p[:3]}"); vals.append(o["r"]); cols.append(c)
ax[2].bar(labs, vals, color=cols); ax[2].axhline(0, color="k", lw=.6); ax[2].set_ylabel("r: change in model party mean vs change in NP dim1")
ax[2].set_title("3  Dynamic responsiveness by party (first differences)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_representation.png", dpi=110)

L = ["# Representation results (generated by 17_representation.py; Kim's bf16 coordinates; exploratory)", "", f"{R['n']['member_congresses']} D/R member-Congresses, {R['n']['members']} members.", "", "## 1 Achen measures (z within era)"]
for pk in PAIRS:
    L.append(f"### {pk}")
    for r in [r for r in rows if r["pair"] == pk]:
        L.append(f"- {r['era']} {r['party']}: responsiveness {r['responsiveness']:+.3f} (se {r['resp_se']:.3f}), within-party r {r['within_party_r']:+.3f}, proximity {r['proximity_msd']:.3f}, centrism SD ratio {r['centrism_sd_ratio']:.3f}")
    L.append("- Republican minus Democrat responsiveness: " + "; ".join(f"{e} {f(c)}" for e, c in A[pk]["rep_minus_dem_responsiveness"].items()))
L += ["", "## 2 Dyadic vs collective (mean correlation over Congresses)"]
for pk, o in D2.items():
    L.append(f"- {pk}: within D {o['within_Democrat']['mean_r']:.2f}, within R {o['within_Republican']['mean_r']:.2f}, all members {o['all_members']['mean_r']:.2f}, state delegations {o['state_delegations']['mean_r']:.2f}; party gap over time levels {o['party_gap_over_time']['levels_r']:.2f}, differences {o['party_gap_over_time']['diff_r']:.2f}; party means levels {o['party_means_over_time']['levels_r_by_party']}, differences {o['party_means_over_time']['diff_r_by_party']}")
    L.append(f"  by era: within D {o['within_Democrat']['by_era']}, within R {o['within_Republican']['by_era']}")
L += ["", "## 3 Dynamic representation"]
for yv in ("y", "x"):
    o = D3[f"within_member_{yv}"]; L.append(f"- within member, {yv}: NP dim1 (Democrats) {f(o['np1'])}; extra for Republicans {f(o['np1_rep'])}; years (D) {f(o['tenure'])}; extra years (R) {f(o['tenure_rep'])}")
for k, o in D3.items():
    if k.startswith("party_mean"): L.append(f"- {k}: slope {o['slope']:+.3f}, r {o['r']:+.2f} (p {o['p']:.3g}, n {o['n']})")
(OUT / "REPRESENTATION_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

#!/usr/bin/env python
"""Step 26 - analyse step 25 against the criteria fixed in kim/0_protocol/PROTOCOL_R3_mechanism_robustness.md. CPU; exploratory.
Output: out/r3/mech_robust_summary.json, out/r3/MECH_ROBUST_RESULTS.md (with the pass / fail gate), out/r3/fig_mech_robust.png
Usage: python 26_analyze_mech_robust.py [--out out]
"""
import argparse, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R3 = OUT / "r3"; S, G = {}, {}
f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
def auc(score, label):
    s = pd.Series(np.asarray(score)).rank(); pos = np.asarray(label).astype(bool)
    return float((s[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))

# ======================= M1 time steering =======================
M1 = pd.read_csv(R3 / "judged_m1.csv"); m1 = {}
for (kind, by, axis), g in M1.groupby(["kind", "base_year", "axis"]) if "base_year" in M1 else []:
    pass
M1["grp"] = np.where(M1.kind == "ordinary", "ordinary_" + M1.base_year.astype(str), "legislator")
for (grp, axis), g in M1.groupby(["grp", "axis"]):
    r = cm.fit_cr1(g, "score_B ~ C(cond, Treatment('none')) + C(pid) + C(q)", "pid")
    m1[f"{grp}|{axis}"] = {c: cm.coef(r, f"C(cond, Treatment('none'))[T.{c}]") for c in sorted(g.cond.unique()) if c != "none"}
    m1[f"{grp}|{axis}|means"] = g.groupby("cond").score_B.mean().round(4).to_dict()
S["M1"] = m1
o23, o55, lg = m1["ordinary_2023|social"], m1["ordinary_1955|social"], m1["legislator|social"]
G["M1a"] = o23["past_1.0"]["ci95"][0] > 0 and lg["past_1.0"]["ci95"][0] > 0
G["M1b"] = o23["past_1.0"]["est"] > o23["past_0.5"]["est"]
G["M1c"] = o55["present_1.0"]["ci95"][1] < 0
G["M1d"] = max(abs(o23["random1_1.0"]["est"]), abs(o23["random2_1.0"]["est"])) < abs(o23["past_1.0"]["est"]) and abs(lg["random1_1.0"]["est"]) < abs(lg["past_1.0"]["est"])

# ======================= M4 placebo =======================
m4 = {}
Vd = pd.read_csv(R3 / "judged_m4_validation.csv")
m4["validation_auc"] = {d: auc(g.score_B, g.side == "B") for d, g in Vd.groupby("dim")}
m4["gate_auc_ge_0.80"] = {d: v >= 0.80 for d, v in m4["validation_auc"].items()}
def ys_effect(df):
    r = cm.fit_cr1(df, "score_B ~ C(cond, Treatment('own')) + C(pid) + C(q)", "pid")
    return (r.params["C(cond, Treatment('own'))[T.y1955]"] - r.params["C(cond, Treatment('own'))[T.y2015]"]) / df.score_B.std()
def ord_effect(df):
    df = df.copy(); df["k"] = df.pid.str[3:]; w = df.groupby(["k", "cond"]).score_B.mean().unstack()
    return float((w["C7_nonpolitical_1955"] - w["C4_nonpolitical"]).mean() / df.score_B.std())
ideo_ys = pd.read_csv(OUT / "stance_counterfactual.csv"); ideo_ys = ideo_ys[(ideo_ys.design == "year_swap") & (ideo_ys.axis == "social")]
ideo_or = pd.read_csv(OUT / "stance_offset.csv"); ideo_or = ideo_or[ideo_or.cond.isin(["C4_nonpolitical", "C7_nonpolitical_1955"]) & (ideo_or.axis == "social")]
m4["standardized_1955_vs_2015_yearswap"] = {"social_ideology": float(ys_effect(ideo_ys))}
m4["standardized_1955_vs_2023_ordinary"] = {"social_ideology": ord_effect(ideo_or)}
for dim in ("tone", "formality"):
    m4["standardized_1955_vs_2015_yearswap"][dim] = float(ys_effect(pd.read_csv(R3 / f"judged_placebo_{dim}_yearswap.csv")))
    m4["standardized_1955_vs_2023_ordinary"][dim] = ord_effect(pd.read_csv(R3 / f"judged_placebo_{dim}_ordinary.csv"))
for key in ("standardized_1955_vs_2015_yearswap", "standardized_1955_vs_2023_ordinary"):
    e = m4[key]; ide = abs(e["social_ideology"]); pl = max(abs(e["tone"]), abs(e["formality"]))
    m4[f"reading_{key}"] = "broad era styling" if pl >= ide else ("ideology-specific" if pl < ide / 3 else "intermediate")
S["M4"] = m4

# ======================= M2 year expressions =======================
M2 = pd.read_csv(R3 / "m2_projections.csv"); m2 = {}
for (kind, ex), g in M2.groupby(["kind", "expr"]):
    w = g.pivot_table(index="pid", columns="period", values="y"); d = w["1950s"] - w["2010s"]
    m2[f"{kind}|{ex}"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n=int(len(d)))
for kind in ("ordinary", "legislator"):
    num = m2[f"{kind}|numeric"]["mean"]
    for ex in ("decade", "era", "relative", "employee_control"):
        if f"{kind}|{ex}" in m2: m2[f"{kind}|{ex}"]["ratio_to_numeric"] = m2[f"{kind}|{ex}"]["mean"] / num
S["M2"] = m2
G["M2a"] = all(m2[f"{k}|{ex}"]["ratio_to_numeric"] >= 0.5 for k in ("ordinary", "legislator") for ex in ("decade", "era"))
G["M2b"] = abs(m2["ordinary|employee_control"]["ratio_to_numeric"]) < 0.25

# ======================= R1 WVS questions =======================
R1 = pd.read_csv(R3 / "r1_projections.csv"); r1 = {}
ys = R1[R1.part == "yearswap"].pivot_table(index="pid", columns="cond", values="y_wvs"); d_w = ys["y1955"] - ys["y2015"]
cp = pd.read_csv(OUT / "counterfactual_personas.csv"); cp = cp[cp.design == "year_swap"].pivot_table(index="pid", columns="cond", values="y"); d_k = cp["y1955"] - cp["y2015"]
r1["yearswap_1955_minus_2015"] = dict(wvs=float(d_w.mean()), kim=float(d_k.mean()), ratio=float(d_w.mean() / d_k.mean()), person_corr=float(np.corrcoef(d_w, d_k.loc[d_w.index])[0, 1]))
od = R1[R1.part == "ordinary"].pivot_table(index="pid", columns="cond", values="y_wvs"); r1["ordinary_1955_minus_2023_wvs"] = float((od["y1955"] - od["y2023"]).mean())
pn = R1[R1.part == "panel"].copy()
v = pd.read_csv(cm.VOTEVIEW, low_memory=False); v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])[["congress", "icpsr", "nokken_poole_dim1"]]
pn = pn.merge(v, on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1"}).dropna(subset=["np1"])
pn["tenure"] = pn.year - pn.groupby("icpsr").year.transform("min"); pn = pn[pn.groupby("icpsr").congress.transform("size") >= 2]
w = pn[["y_wvs", "np1", "tenure"]] - pn.groupby("icpsr")[["y_wvs", "np1", "tenure"]].transform("mean"); w["icpsr"] = pn.icpsr.values
rr = cm.fit_cr1(w, "y_wvs ~ 0 + np1 + tenure", "icpsr"); r1["panel_within_member"] = {k: cm.coef(rr, k) for k in ("np1", "tenure")}; r1["panel_rows"] = int(len(pn))
S["R1"] = r1
G["R1a"] = np.sign(d_w.mean()) == np.sign(d_k.mean()) and r1["yearswap_1955_minus_2015"]["ratio"] >= 0.5
G["R1b"] = r1["panel_within_member"]["tenure"]["ci95"][1] < 0

# ======================= R3 baseline groups =======================
B = pd.read_csv(R3 / "r3_baselines.csv"); mem = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "members_*.csv")))], ignore_index=True)
mem = mem[mem.party.isin(["Democrat", "Republican"])].merge(v, on=["congress", "icpsr"], how="left", validate="m:1").rename(columns={"nokken_poole_dim1": "np1"})
r3 = {}
for gname, g in B.groupby("group"):
    base = g.groupby("year").y.mean(); mm = mem.assign(yg=mem.y_raw - mem.year.map(base))
    pm = mm.groupby(["congress", "party"])[["yg", "np1"]].mean().unstack("party"); rep = pm.xs("Republican", axis=1, level=1).sort_index()
    r3[gname] = dict(rep_corr_np1=float(np.corrcoef(rep.yg, rep.np1)[0, 1]), rep_change=float(rep.yg.iloc[-1] - rep.yg.iloc[0]),
                     baseline_1947=float(base.iloc[0]), baseline_2023=float(base.iloc[-1]), baseline_drift=float(base.iloc[-1] - base.iloc[0]))
    r3[gname]["pass"] = r3[gname]["rep_corr_np1"] > 0.5 and r3[gname]["rep_change"] > 0
pm0 = mem.groupby(["congress", "party"])[["y_raw", "np1"]].mean().unstack("party").xs("Republican", axis=1, level=1).sort_index()
r3["_raw_uncorrected"] = dict(rep_corr_np1=float(np.corrcoef(pm0.y_raw, pm0.np1)[0, 1]), rep_change=float(pm0.y_raw.iloc[-1] - pm0.y_raw.iloc[0]))
S["R3"] = r3; G["R3"] = sum(o["pass"] for k, o in r3.items() if not k.startswith("_")) >= 6
S["gate"] = {k: bool(v) for k, v in G.items()}
(R3 / "mech_robust_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")

# ======================= figure =======================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
labs, vals, errs, cols = [], [], [], []
for grp, conds in (("ordinary_2023", ["past_0.5", "past_1.0", "future_0.5", "random1_1.0", "random2_1.0"]), ("ordinary_1955", ["present_1.0", "random1_1.0"]), ("legislator", ["past_1.0", "present_0.5", "random1_1.0"])):
    for c in conds:
        e = m1[f"{grp}|social"][c]; labs.append(f"{grp.replace('ordinary_', 'ord ')}\n{c}"); vals.append(e["est"]); errs.append(1.96 * e["se"])
        cols.append("tab:gray" if "random" in c else ("tab:orange" if ("past" in c) else "tab:blue"))
ax[0].bar(range(len(vals)), vals, yerr=errs, color=cols); ax[0].axhline(0, color="k", lw=.6); ax[0].set_xticks(range(len(vals)), labs, fontsize=6, rotation=60)
ax[0].set_ylabel("change in judge score vs no steering (social; + = traditional)"); ax[0].set_title("M1 steering along the time direction", fontsize=9)
for i, kind in enumerate(("ordinary", "legislator")):
    exs = [e for e in ("numeric", "decade", "era", "relative", "employee_control") if f"{kind}|{e}" in m2]
    ax[1].bar(np.arange(len(exs)) + i * .4, [m2[f"{kind}|{e}"]["mean"] for e in exs], .4, label=kind)
ax[1].set_xticks(np.arange(5) + .2, ["numeric", "decade", "era", "relative", "employee\ncontrol"], fontsize=8); ax[1].axhline(0, color="k", lw=.6)
ax[1].set_ylabel("internal y: 1950s minus 2010s"); ax[1].legend(fontsize=8); ax[1].set_title("M2 what carries the time signal?", fontsize=9)
gn = [k for k in r3 if not k.startswith("_")]
ax[2].bar(gn, [r3[k]["rep_corr_np1"] for k in gn], color=["tab:green" if r3[k]["pass"] else "tab:red" for k in gn]); ax[2].axhline(r3["_raw_uncorrected"]["rep_corr_np1"], color="k", ls="--", lw=1)
ax[2].axhline(0.5, color=".6", ls=":", lw=1); ax[2].set_xticks(range(len(gn)), gn, rotation=45, fontsize=7); ax[2].set_ylabel("corrected Republican mean vs NP dim1 (r)")
ax[2].set_title("R3 correction under eight baseline groups (dashed = uncorrected)", fontsize=9)
fig.tight_layout(); fig.savefig(R3 / "fig_mech_robust.png", dpi=110)

L = ["# R3 mechanism and robustness (generated by 26_analyze_mech_robust.py; criteria fixed in PROTOCOL_R3)", "", "## Gate", "| criterion | result |", "|---|---|"]
L += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in S["gate"].items()]
L += ["", "## M1 time steering (social questions; vs no steering)"]
for grp in ("ordinary_2023", "ordinary_1955", "legislator"):
    L.append(f"- {grp}: " + "; ".join(f"{c} {f(e)}" for c, e in m1[f"{grp}|social"].items()))
    L.append(f"  economic: " + "; ".join(f"{c} {f(e)}" for c, e in m1[f"{grp}|economic"].items()))
L += ["", "## M4 placebo", f"- validation AUC {m4['validation_auc']} (gate {m4['gate_auc_ge_0.80']})",
      f"- standardized 1955 vs 2015 (year swap): {m4['standardized_1955_vs_2015_yearswap']} -> {m4['reading_standardized_1955_vs_2015_yearswap']}",
      f"- standardized 1955 vs 2023 (ordinary): {m4['standardized_1955_vs_2023_ordinary']} -> {m4['reading_standardized_1955_vs_2023_ordinary']}",
      "", "## M2 year expressions (internal y, 1950s minus 2010s)"] + [f"- {k}: {o}" for k, o in m2.items()]
L += ["", "## R1 WVS questions", f"- {r1['yearswap_1955_minus_2015']}", f"- ordinary 1955 - 2023: {r1['ordinary_1955_minus_2023_wvs']:+.3f}",
      f"- panel within member ({r1['panel_rows']} rows): NP dim1 {f(r1['panel_within_member']['np1'])}; years {f(r1['panel_within_member']['tenure'])}",
      "", "## R3 baseline groups"] + [f"- {k}: {o}" for k, o in r3.items()]
(R3 / "MECH_ROBUST_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

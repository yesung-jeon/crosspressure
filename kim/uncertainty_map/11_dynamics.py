#!/usr/bin/env python
"""Step 11 - R1, R2: does Kim's map move with real U.S. politics over time? (CPU)

R1 Career dynamics (within member). Panel = Kim's member scores (21,288 member-Congresses) joined 1:1 to Voteview
   Nokken-Poole scores (roll-call ideology estimated separately for each Congress, so a member's real drift is visible).
   Fixed effects are removed by demeaning (member; member + Congress by alternating projections), SE clustered by member.
     (a) model y (or x) ~ NP dim1 + NP dim2                     within member
     (b) ... + years since the member's first Congress          within member: real drift vs elapsed time
     (c) ... with member and Congress fixed effects            does the model follow a member's drift relative to peers?
   Also: the 30 members with the largest real change in NP dim1 (first vs last Congress), model change beside it.
R2 Time series (party x Congress). Model party means and gap (R - D) vs DW-NOMINATE and Nokken-Poole party gaps, in levels
   and in first differences (level correlations of trending series are inflated); entropy by party x Congress from step 1.
Output: out/dynamics_summary.json, out/DYNAMICS_RESULTS.md, out/career_movers.csv, out/party_series.csv, out/fig_dynamics.png
Usage: python 11_dynamics.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}
m = cm.members()
v = pd.read_csv(cm.VOTEVIEW, low_memory=False)
v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])[["congress", "icpsr", "nokken_poole_dim1", "nokken_poole_dim2"]]
d = m.merge(v, on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1", "nokken_poole_dim2": "np2"})
d = d[d.party.isin(["Democrat", "Republican"])].dropna(subset=["np1", "np2"]).copy()
d["tenure_years"] = d.year - d.groupby("icpsr").year.transform("min")
d = d[d.groupby("icpsr").congress.transform("size") >= 2]
R["panel"] = dict(rows=int(len(d)), members=int(d.icpsr.nunique()))

def demean(df, cols, by):
    out = df[cols].copy()
    for _ in range(50 if len(by) > 1 else 1):
        for g in by:
            out = out - out.groupby(df[g]).transform("mean")
    return out
def within(df, yv, xs, by, label):
    w = demean(df, [yv] + xs, by); w["icpsr"] = df.icpsr.values
    r = cm.fit_cr1(w, f"{yv} ~ 0 + " + " + ".join(xs), "icpsr")
    ss_tot = float((w[yv] ** 2).sum()); ss_res = float((r.resid ** 2).sum())
    return {k: cm.coef(r, k) for k in xs} | dict(within_R2=1 - ss_res / ss_tot, spec=label)
R1 = {}
for yv in ("y", "x"):
    R1[yv] = {"a_member_FE": within(d, yv, ["np1", "np2"], ["icpsr"], "member FE"),
              "b_member_FE_plus_tenure": within(d, yv, ["np1", "np2", "tenure_years"], ["icpsr"], "member FE + years since first Congress"),
              "c_member_and_congress_FE": within(d, yv, ["np1", "np2"], ["icpsr", "congress"], "member + Congress FE")}
for kn in (True, False):
    dk = d[d.known.astype(bool) == kn]
    R1[f"y_known_{kn}"] = within(dk, "y", ["np1", "np2", "tenure_years"], ["icpsr"], "member FE + tenure, subgroup")
R["R1_career"] = R1
s = d.sort_values("congress").groupby("icpsr").agg(name=("name", "first"), party=("party", "first"), first=("year", "first"), last=("year", "last"),
        np1_first=("np1", "first"), np1_last=("np1", "last"), y_first=("y", "first"), y_last=("y", "last"), x_first=("x", "first"), x_last=("x", "last"))
s["d_np1"], s["d_y"], s["d_x"] = s.np1_last - s.np1_first, s.y_last - s.y_first, s.x_last - s.x_first
mov = s.reindex(s.d_np1.abs().sort_values(ascending=False).index).head(30); mov.round(3).to_csv(OUT / "career_movers.csv")
R["R1_movers_corr"] = dict(corr_dnp1_dx=float(np.corrcoef(mov.d_np1, mov.d_x)[0, 1]), corr_dnp1_dy=float(np.corrcoef(mov.d_np1, mov.d_y)[0, 1]),
                           all_members_corr_dnp1_dx=float(np.corrcoef(s.d_np1, s.d_x)[0, 1]), all_members_corr_dnp1_dy=float(np.corrcoef(s.d_np1, s.d_y)[0, 1]))

# ---- R2 ----
full = m[m.party.isin(["Democrat", "Republican"])].merge(v, on=["congress", "icpsr"], how="left", validate="1:1")
ps = full.groupby(["congress", "party"])[["x", "y", "dim1", "nokken_poole_dim1"]].mean().unstack("party")
ser = pd.DataFrame({"year": [cm.year_of(c) for c in ps.index]}, index=ps.index)
for k in ("x", "y", "dim1", "nokken_poole_dim1"):
    ser[f"gap_{k}"] = ps[(k, "Republican")] - ps[(k, "Democrat")]
    ser[f"dem_{k}"], ser[f"rep_{k}"] = ps[(k, "Democrat")], ps[(k, "Republican")]
G = pd.read_csv(OUT / "member_generations.csv"); G = G[G.party.isin(["Democrat", "Republican"])]
ge = G.groupby(["congress", "party"]).entropy64.mean().unstack("party"); ser["entropy_dem"], ser["entropy_rep"] = ge.get("Democrat"), ge.get("Republican")
ser["entropy_gap"] = ser.entropy_rep - ser.entropy_dem; ser.to_csv(OUT / "party_series.csv")
def cc(a, b):
    z = pd.concat([a, b], axis=1).dropna(); r, p = pearsonr(z.iloc[:, 0], z.iloc[:, 1]); return dict(r=float(r), p=float(p), n=int(len(z)))
R2 = {}
for mk in ("x", "y"):
    for rk in ("dim1", "nokken_poole_dim1"):
        R2[f"gap_{mk}_vs_{rk}"] = dict(levels=cc(ser[f"gap_{mk}"], ser[f"gap_{rk}"]), first_differences=cc(ser[f"gap_{mk}"].diff(), ser[f"gap_{rk}"].diff()))
    for p in ("dem", "rep"):
        R2[f"{p}_{mk}_vs_np1"] = dict(levels=cc(ser[f"{p}_{mk}"], ser[f"{p}_nokken_poole_dim1"]), first_differences=cc(ser[f"{p}_{mk}"].diff(), ser[f"{p}_nokken_poole_dim1"].diff()))
ev = ser.dropna(subset=["entropy_gap"])
R2["entropy_gap_vs_np1_gap"] = dict(levels=cc(ev.entropy_gap, ev.gap_nokken_poole_dim1), first_differences=cc(ev.entropy_gap.diff(), ev.gap_nokken_poole_dim1.diff()))
R2["entropy_gap_vs_model_y_gap"] = dict(levels=cc(ev.entropy_gap, ev.gap_y), first_differences=cc(ev.entropy_gap.diff(), ev.gap_y.diff()))
R["R2_series"] = R2
(OUT / "dynamics_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
z = lambda s: (s - s.mean()) / s.std()
ax[0].plot(ser.year, z(ser.gap_y), label="model social gap (R-D)"); ax[0].plot(ser.year, z(ser.gap_x), label="model economic gap (R-D)")
ax[0].plot(ser.year, z(ser.gap_nokken_poole_dim1), "k--", label="Nokken-Poole dim1 gap"); ax[0].legend(fontsize=7); ax[0].set_title("R2 levels (standardized)", fontsize=10)
ax[1].plot(ser.year, ser.gap_y.diff(), label="model social gap, change"); ax[1].plot(ser.year, ser.gap_nokken_poole_dim1.diff() * ser.gap_y.diff().std() / ser.gap_nokken_poole_dim1.diff().std(), "k--", label="NP dim1 gap, change (rescaled)")
ax[1].axhline(0, color=".7", lw=.6); ax[1].legend(fontsize=7); ax[1].set_title("R2 first differences", fontsize=10)
ax[2].scatter(s.d_np1, s.d_y, s=4, alpha=.3, c=np.where(s.party == "Republican", "tab:red", "tab:blue"))
ax[2].set_xlabel("real change in Nokken-Poole dim1 (first to last Congress)"); ax[2].set_ylabel("model change in y"); ax[2].set_title("R1 career change: model vs roll calls", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_dynamics.png", dpi=110)

f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
L = ["# Dynamics results (generated by 11_dynamics.py; exploratory)", "", f"Panel: {R['panel']['rows']} member-Congresses, {R['panel']['members']} members with >= 2 Congresses.", ""]
for yv in ("y", "x"):
    for k, o in R1[yv].items():
        L.append(f"- R1 {yv}, {o['spec']}: NP dim1 {f(o['np1'])}; NP dim2 {f(o['np2'])}" + (f"; years {f(o['tenure_years'])}" if "tenure_years" in o else "") + f"; within R2 {o['within_R2']:.3f}")
L.append(f"- R1 movers (30 largest real changes): corr(d NP dim1, d x) {R['R1_movers_corr']['corr_dnp1_dx']:.2f}, corr(d NP dim1, d y) {R['R1_movers_corr']['corr_dnp1_dy']:.2f}; all members {R['R1_movers_corr']['all_members_corr_dnp1_dx']:.2f}, {R['R1_movers_corr']['all_members_corr_dnp1_dy']:.2f}")
L.append("")
for k, o in R2.items():
    L.append(f"- R2 {k}: levels r = {o['levels']['r']:.2f} (n {o['levels']['n']}); first differences r = {o['first_differences']['r']:.2f} (p {o['first_differences']['p']:.3g})")
(OUT / "DYNAMICS_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

#!/usr/bin/env python
"""Step 4 - analyses A-E (CPU). All fits: OLS, CR1 clustered SE, reconstructed independently (common.fit_cr1).
Exploratory: p values are unadjusted; every test run here is reported in out/summary.json.

A  Who can the model speak for: state x era hesitation, raw and net of the members' real ideology.
B  Caricature: model's social position beyond what the voting record predicts, and entropy.
C  Same person, different conditions: year swap; party switchers.
D  Political geography without names: anonymous persona vs the real delegation of the same state, chamber, Congress.
E  Reading history from the present: party distance from the model's own default vs entropy; model vs DW-NOMINATE
   polarization over time (Kim's full member file, 1947-2023).
Output: out/summary.json, out/*.csv, out/fig_*.png
Usage: python 04_analyze.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; S = {}
FE = "C(party) * C(era) + C(chamber)"

# ======================= member generations (A, B, E) =======================
G = pd.read_csv(OUT / "member_generations.csv"); G["known"] = G.known.astype(bool)
G = G[G.party.isin(["Democrat", "Republican"])].copy()
M = G.groupby(["icpsr", "congress"]).agg(entropy=("entropy64", "mean"), hedge=("hedge", "mean"), **{c: (c, "first") for c in
        ["name", "party", "chamber", "state", "year", "known", "era", "group", "x", "y", "dim1", "dim2"]}).reset_index()
S["n"] = dict(member_congresses=int(len(M)), members=int(M.icpsr.nunique()), sequences=int(len(G)),
              by_era=M.era.value_counts().to_dict(), known_share=float(M.known.mean()))

# ---- B: caricature ----
rb = cm.fit_cr1(M, f"y ~ dim1 + dim2 + {FE}", "icpsr"); M["y_resid"] = rb.resid
G = G.merge(M[["icpsr", "congress", "y_resid"]], on=["icpsr", "congress"], how="left", validate="m:1")
B = {"R2_y_on_record": float(rb.rsquared), "sd_y_resid": float(M.y_resid.std())}
fb = f"entropy64 ~ y_resid + dim1 + dim2 + {FE} + C(q)"
B["entropy_on_excess_traditionalism"] = cm.coef(cm.fit_cr1(G, fb, "icpsr"), "y_resid")
B["hedging_on_excess_traditionalism"] = cm.coef(cm.fit_cr1(G, fb.replace("entropy64", "hedge"), "icpsr"), "y_resid")
for ax in ("economic", "social"):
    B[f"entropy_on_excess__{ax}_questions"] = cm.coef(cm.fit_cr1(G[G.axis == ax], fb, "icpsr"), "y_resid")
for kn in (True, False):
    B[f"entropy_on_excess__known_{kn}"] = cm.coef(cm.fit_cr1(G[G.known == kn], fb, "icpsr"), "y_resid")
for era in sorted(G.era.unique()):
    B[f"entropy_on_excess__{era}"] = cm.coef(cm.fit_cr1(G[G.era == era], fb.replace(" * C(era)", ""), "icpsr"), "y_resid")
# where does excess traditionalism come from? (member level)
B["excess_by_group_era"] = {f"{k[0]}|{k[1]}": round(float(v), 3) for k, v in M.groupby(["era", "group"]).y_resid.mean().items()}
B["excess_on_abs_dim2_within_party_era"] = cm.coef(cm.fit_cr1(M.assign(abs_dim2=M.dim2.abs()), f"y_resid ~ abs_dim2 + dim1 + {FE}", "icpsr"), "abs_dim2")
B["excess_known_vs_unknown"] = cm.coef(cm.fit_cr1(M.assign(known_i=M.known.astype(int)), f"y_resid ~ known_i + {FE}", "icpsr"), "known_i")
S["B_caricature"] = B

# ---- A: who can the model speak for ----
fa = f"entropy64 ~ dim1 + dim2 + {FE} + C(q)"
ra = cm.fit_cr1(G, fa, "icpsr"); G["e_net"] = ra.resid
st = G.groupby(["state", "era"]).agg(entropy=("entropy64", "mean"), entropy_net_of_ideology=("e_net", "mean"),
                                     members=("icpsr", "nunique"), dim1=("dim1", "mean")).reset_index()
st.to_csv(OUT / "state_by_era.csv", index=False)
stc = G.groupby(["state", "congress"]).agg(entropy=("entropy64", "mean"), members=("icpsr", "nunique")).reset_index()
stc.to_csv(OUT / "state_by_congress.csv", index=False)
G["south"] = G.state.isin(cm.SOUTH).astype(int)
A = {"south_raw": cm.coef(cm.fit_cr1(G, f"entropy64 ~ south + C(era) + C(chamber) + C(q)", "icpsr"), "south"),
     "south_net_of_ideology": cm.coef(cm.fit_cr1(G, f"entropy64 ~ south + dim1 + dim2 + {FE} + C(q)", "icpsr"), "south"),
     "south_net_of_ideology_and_model_position": cm.coef(cm.fit_cr1(G, f"entropy64 ~ south + dim1 + dim2 + x + y + {FE} + C(q)", "icpsr"), "south")}
top = st[st.members >= 5].sort_values("entropy_net_of_ideology")
A["states_most_hesitant_net"] = top.tail(8)[["state", "era", "entropy_net_of_ideology", "members"]].round(3).to_dict("records")
A["states_least_hesitant_net"] = top.head(8)[["state", "era", "entropy_net_of_ideology", "members"]].round(3).to_dict("records")
S["A_who_can_it_speak_for"] = A

# ---- E: reading history from the present ----
M["dist_default"] = np.hypot(M.x, M.y)
pc = M.groupby(["congress", "party"]).agg(entropy=("entropy", "mean"), dist=("dist_default", "mean"), y=("y", "mean"),
                                          x=("x", "mean"), n=("icpsr", "size")).reset_index()
pc["year"] = cm.year_of(1) + 2 * (pc.congress - 1); pc.to_csv(OUT / "party_by_congress_generated.csv", index=False)
E = {"member_entropy_on_distance_from_default": cm.coef(cm.fit_cr1(G.merge(M[["icpsr", "congress", "dist_default"]], on=["icpsr", "congress"], validate="m:1"),
                                                               f"entropy64 ~ dist_default + C(party) + C(congress) + C(chamber) + C(q)", "icpsr"), "dist_default"),
     "member_entropy_on_distance_and_y": {k: cm.coef(cm.fit_cr1(G.merge(M[["icpsr", "congress", "dist_default"]], on=["icpsr", "congress"], validate="m:1"),
                                                              f"entropy64 ~ dist_default + y + C(party) + C(congress) + C(chamber) + C(q)", "icpsr"), k) for k in ("dist_default", "y")}}
full = cm.members(); full = full[full.party.isin(["Democrat", "Republican"])]
pol = full.groupby(["congress", "party"])[["x", "y", "dim1", "dim2"]].mean().unstack("party")
pol_df = pd.DataFrame({"year": [cm.year_of(c) for c in pol.index],
                       "gap_x_model": pol[("x", "Republican")] - pol[("x", "Democrat")], "gap_y_model": pol[("y", "Republican")] - pol[("y", "Democrat")],
                       "gap_dim1": pol[("dim1", "Republican")] - pol[("dim1", "Democrat")],
                       "dem_dim1": pol[("dim1", "Democrat")], "rep_dim1": pol[("dim1", "Republican")],
                       "dem_x": pol[("x", "Democrat")], "rep_x": pol[("x", "Republican")], "dem_y": pol[("y", "Democrat")], "rep_y": pol[("y", "Republican")]},
                      index=pol.index)
pol_df.to_csv(OUT / "polarization_by_congress.csv")
f, l = pol_df.iloc[0], pol_df.iloc[-1]
E["polarization_corr_gap_x_vs_dim1"] = float(pearsonr(pol_df.gap_x_model, pol_df.gap_dim1)[0])
E["polarization_corr_gap_y_vs_dim1"] = float(pearsonr(pol_df.gap_y_model, pol_df.gap_dim1)[0])
E["change_1947_to_2023"] = {"dim1_dem": float(l.dem_dim1 - f.dem_dim1), "dim1_rep": float(l.rep_dim1 - f.rep_dim1),
                            "model_x_dem": float(l.dem_x - f.dem_x), "model_x_rep": float(l.rep_x - f.rep_x),
                            "model_y_dem": float(l.dem_y - f.dem_y), "model_y_rep": float(l.rep_y - f.rep_y),
                            "note": "dim1 change is mechanically constrained by DW-NOMINATE's scaling; compare signs and relative sizes, cite Voteview before use"}
S["E_history_from_the_present"] = E

# ======================= C: counterfactual =======================
if (OUT / "counterfactual_generations.csv").exists():
    Cg = pd.read_csv(OUT / "counterfactual_generations.csv"); Cp = pd.read_csv(OUT / "counterfactual_personas.csv")
    Cc = {}
    ys = Cg[Cg.design == "year_swap"].copy(); yp = Cp[Cp.design == "year_swap"].copy()
    fy = "entropy64 ~ C(cond, Treatment('own')) + C(pid) + C(q)"
    r = cm.fit_cr1(ys, fy, "pid")
    Cc["year_swap_entropy_vs_own_year"] = {c: cm.coef(r, f"C(cond, Treatment('own'))[T.{c}]") for c in ("y1955", "y1985", "y2015")}
    r = cm.fit_cr1(yp, "y ~ C(cond, Treatment('own')) + C(pid)", "pid")
    Cc["year_swap_y_vs_own_year"] = {c: cm.coef(r, f"C(cond, Treatment('own'))[T.{c}]") for c in ("y1955", "y1985", "y2015")}
    ys["rep"] = (ys.party == "Republican").astype(int); ys["yr2015_vs_1955"] = ys.cond.map({"y1955": 0, "y2015": 1})
    r = cm.fit_cr1(ys.dropna(subset=["yr2015_vs_1955"]), "entropy64 ~ yr2015_vs_1955 * rep + C(pid) + C(q)", "pid")
    Cc["entropy_2015_vs_1955_by_party"] = {k: cm.coef(r, k) for k in ("yr2015_vs_1955", "yr2015_vs_1955:rep")}
    Cc["year_swap_cell_means"] = {f"{k[0]}|{k[1]}": round(float(v), 4) for k, v in ys.groupby(["party", "cond"]).entropy64.mean().items()}
    Cc["year_swap_y_means"] = {f"{k[0]}|{k[1]}": round(float(v), 3) for k, v in yp.groupby(["party", "cond"]).y.mean().items()}
    sw = Cg[Cg.design == "switcher"].groupby(["pid", "cond", "switch"]).entropy64.mean().unstack("cond").reset_index()
    swp = Cp[Cp.design == "switcher"].pivot_table(index=["pid", "switch"], columns="cond", values=["y", "x"]).reset_index()
    swp.columns = ["_".join(c).strip("_") for c in swp.columns]
    sw = sw.merge(swp, on=["pid", "switch"], validate="1:1")
    sw["d_entropy"], sw["d_y"], sw["d_x"] = sw.after - sw.before, sw.y_after - sw.y_before, sw.x_after - sw.x_before
    sw.to_csv(OUT / "switchers.csv", index=False)
    for d, g in sw.groupby("switch"):
        Cc[f"switch_{d}"] = dict(n=int(len(g)), d_entropy_mean=float(g.d_entropy.mean()), d_y_mean=float(g.d_y.mean()), d_x_mean=float(g.d_x.mean()),
                                 wilcoxon_p_entropy=float(wilcoxon(g.d_entropy).pvalue) if len(g) >= 6 else None,
                                 wilcoxon_p_y=float(wilcoxon(g.d_y).pvalue) if len(g) >= 6 else None)
    S["C_counterfactual"] = Cc

# ======================= D: anonymous personas =======================
if (OUT / "anonymous_generations.csv").exists():
    cells = pd.read_csv(OUT / "anonymous_cells.csv"); ag = pd.read_csv(OUT / "anonymous_generations.csv")
    real = pd.read_csv(OUT / "real_member_projections.csv")
    rc = real.groupby(["state", "chamber", "congress"]).agg(x_real=("x_int8", "mean"), y_real=("y_int8", "mean"), dim1_real=("dim1", "mean"),
                                                            dim1_sd=("dim1", "std"), rep_share=("party", lambda p: float((p == "Republican").mean())),
                                                            n_real=("icpsr", "size")).reset_index()
    ae = ag.groupby(["state", "chamber", "congress"]).agg(entropy_anon=("entropy64", "mean"), hedge_anon=("hedge", "mean")).reset_index()
    D = cells.merge(rc, on=["state", "chamber", "congress"], how="left", validate="1:1").merge(ae, on=["state", "chamber", "congress"], how="left", validate="1:1")
    D["split"] = 1 - 2 * (D.rep_share - 0.5).abs(); D["south"] = D.state.isin(cm.SOUTH).astype(int); D["y_gap"] = D.y_anon - D.y_real
    D.to_csv(OUT / "anonymous_vs_real.csv", index=False); Dd = D.dropna(subset=["y_real"])
    Dr = {"cells": int(len(D)), "cells_with_real_members": int(len(Dd)),
          "corr_y_anon_real": float(pearsonr(Dd.y_anon, Dd.y_real)[0]), "corr_x_anon_real": float(pearsonr(Dd.x_anon, Dd.x_real)[0]),
          "corr_y_anon_dim1real": float(pearsonr(Dd.y_anon, Dd.dim1_real)[0]),
          "corr_by_congress_y": {int(c): round(float(pearsonr(g.y_anon, g.y_real)[0]), 3) for c, g in Dd.groupby("congress")}}
    Dr["y_gap_anon_minus_real_by_south_congress"] = {f"south{int(k[0])}|{int(k[1])}": round(float(v), 3) for k, v in Dd.groupby(["south", "congress"]).y_gap.mean().items()}
    Dd = Dd.assign(cell=Dd.state + Dd.chamber + Dd.congress.astype(str))
    ag2 = ag.merge(Dd[["state", "chamber", "congress", "y_anon", "y_real", "dim1_real", "split", "south", "cell"]], on=["state", "chamber", "congress"], validate="m:1")
    r = cm.fit_cr1(ag2, "entropy64 ~ y_anon + split + C(congress) + C(chamber) + C(q)", "cell")
    Dr["anon_entropy_on_own_y_and_split"] = {k: cm.coef(r, k) for k in ("y_anon", "split")}
    r = cm.fit_cr1(ag2, "entropy64 ~ dim1_real + split + south + C(congress) + C(chamber) + C(q)", "cell")
    Dr["anon_entropy_on_real_delegation"] = {k: cm.coef(r, k) for k in ("dim1_real", "split", "south")}
    S["D_anonymous"] = Dr

(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")

# ======================= quick-look figures (checked visually) =======================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
mm = M.sample(min(len(M), 1500), random_state=0)
ax[0].scatter(mm.y_resid, mm.entropy, s=6, c=np.where(mm.party == "Republican", "tab:red", "tab:blue"), alpha=.5)
ax[0].set_xlabel("model y minus what the voting record predicts (+ = placed more traditional)"); ax[0].set_ylabel("persona entropy"); ax[0].set_title("B  Caricature and hesitation", fontsize=10)
ax[1].plot(pol_df.year, pol_df.gap_y_model / pol_df.gap_y_model.abs().max(), label="model social gap (R-D), scaled")
ax[1].plot(pol_df.year, pol_df.gap_x_model / pol_df.gap_x_model.abs().max(), label="model economic gap (R-D), scaled")
ax[1].plot(pol_df.year, pol_df.gap_dim1 / pol_df.gap_dim1.abs().max(), "k--", label="DW-NOMINATE dim1 gap, scaled")
ax[1].legend(fontsize=7); ax[1].set_title("E  Polarization: model map vs roll calls", fontsize=10); ax[1].set_xlabel("year")
if "D_anonymous" in S:
    ax[2].scatter(Dd.y_real, Dd.y_anon, s=8, c=Dd.congress, cmap="viridis"); lim = [min(Dd.y_real.min(), Dd.y_anon.min()), max(Dd.y_real.max(), Dd.y_anon.max())]
    ax[2].plot(lim, lim, "k:", lw=1); ax[2].set_xlabel("real delegation mean y (same run)"); ax[2].set_ylabel("anonymous persona y"); ax[2].set_title("D  Geography without names (color = Congress)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_BED.png", dpi=110)
print(json.dumps(S, indent=1, default=float))

# ======================= RESULTS.md (numbers filled from summary.json, never typed) =======================
def fmt(c): return f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
L = ["# uncertainty_map results (generated by 04_analyze.py from out/summary.json; exploratory, unadjusted p)", "",
     f"Members: {S['n']['member_congresses']} member-Congresses ({S['n']['members']} people), {S['n']['sequences']} generations; party recalled for {S['n']['known_share']:.0%}.", "",
     "## B  Caricature", f"- Model y explained by the voting record: R2 = {B['R2_y_on_record']:.2f}.",
     f"- Entropy on excess traditionalism (model y beyond the record): {fmt(B['entropy_on_excess_traditionalism'])}",
     f"- Economic questions: {fmt(B['entropy_on_excess__economic_questions'])}; social questions: {fmt(B['entropy_on_excess__social_questions'])}",
     f"- Known members: {fmt(B['entropy_on_excess__known_True'])}; unknown: {fmt(B['entropy_on_excess__known_False'])}",
     f"- Excess traditionalism on |dim2| (race / civil-rights dimension): {fmt(B['excess_on_abs_dim2_within_party_era'])}", "",
     "## A  Who can the model speak for", f"- South, raw: {fmt(A['south_raw'])}", f"- South, net of real ideology: {fmt(A['south_net_of_ideology'])}",
     f"- South, net of ideology and model position: {fmt(A['south_net_of_ideology_and_model_position'])}", "",
     "## E  Reading history from the present",
     f"- Entropy on distance from the model default: {fmt(E['member_entropy_on_distance_from_default'])}; with y: distance {fmt(E['member_entropy_on_distance_and_y']['dist_default'])}, y {fmt(E['member_entropy_on_distance_and_y']['y'])}",
     f"- Party gap over time, model vs DW-NOMINATE dim1: r = {E['polarization_corr_gap_x_vs_dim1']:.2f} (economic), {E['polarization_corr_gap_y_vs_dim1']:.2f} (social)",
     "- Change 1947 -> 2023: " + ", ".join(f"{k} {v:+.2f}" for k, v in E["change_1947_to_2023"].items() if k != "note") + " (" + E["change_1947_to_2023"]["note"] + ")"]
if "C_counterfactual" in S:
    Cc = S["C_counterfactual"]
    L += ["", "## C  Same person, different conditions"] + [f"- Entropy, year {k} vs own year: {fmt(v)}" for k, v in Cc["year_swap_entropy_vs_own_year"].items()] + \
         [f"- Model y, year {k} vs own year: {fmt(v)}" for k, v in Cc["year_swap_y_vs_own_year"].items()] + \
         [f"- Entropy 2015 vs 1955: {fmt(Cc['entropy_2015_vs_1955_by_party']['yr2015_vs_1955'])}; extra for Republicans: {fmt(Cc['entropy_2015_vs_1955_by_party']['yr2015_vs_1955:rep'])}"] + \
         [f"- Switchers {k[7:]}: n = {v['n']}, entropy after-before {v['d_entropy_mean']:+.3f} (Wilcoxon p {v['wilcoxon_p_entropy']}), y {v['d_y_mean']:+.3f} (p {v['wilcoxon_p_y']}), x {v['d_x_mean']:+.3f}" for k, v in Cc.items() if k.startswith("switch_")]
if "D_anonymous" in S:
    Dr = S["D_anonymous"]
    L += ["", "## D  Geography without names", f"- Cells {Dr['cells']} ({Dr['cells_with_real_members']} with real members). Anonymous vs real delegation: r(y) = {Dr['corr_y_anon_real']:.2f}, r(x) = {Dr['corr_x_anon_real']:.2f}; by Congress r(y): {Dr['corr_by_congress_y']}",
          f"- Anonymous entropy on its own y: {fmt(Dr['anon_entropy_on_own_y_and_split']['y_anon'])}; on delegation split: {fmt(Dr['anon_entropy_on_own_y_and_split']['split'])}",
          f"- Anonymous entropy on real delegation dim1: {fmt(Dr['anon_entropy_on_real_delegation']['dim1_real'])}; South: {fmt(Dr['anon_entropy_on_real_delegation']['south'])}",
          f"- Anonymous minus real y, by South x Congress: {Dr['y_gap_anon_minus_real_by_south_congress']}"]
(OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

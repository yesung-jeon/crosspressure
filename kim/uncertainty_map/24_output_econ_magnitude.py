#!/usr/bin/env python
"""Step 24 - analyses #2 (output-level distortion), #3 (why the economic correction fails), #5 (size of the distortion). CPU.

Fixed before looking at step 23's outputs (2026-10-04):
- Judge gate: AUC >= 0.80 on the validation set, per axis. If an axis fails, its output-level results are reported as unreliable.
- #3 a-priori grouping of Kim's held-out questions: "present-day issues" not on the national agenda around 1955 = parental leave (q2),
  student debt (q3), gender-neutral language (q5), same-sex adoption (q8); the other six are "long-standing".
#2  Year swap (same legislator, prompt year 1955 / 1985 / 2015 vs own year) and ordinary persons (1955 vs 2023) on the judge's
    right / traditional score; link between internal shift and output shift; legislators' outputs vs DW-NOMINATE within party,
    and a calendar term net of the record.
#3  Per-question time slopes of ordinary persons' internal positions (1900-2023), present-day vs long-standing questions; the
    same contrast in outputs (ordinary persons 1955 vs 2023).
#5  The distortion expressed in DW-NOMINATE units (internal shift / slope of the model position on dim1 within Congress) and
    compared with the parties' real movement 1947-2023 and the current party gap; the same ratio for outputs.
Output: out/output_econ_magnitude.json, out/OUTPUT_ECON_MAGNITUDE.md, out/fig_output_econ_magnitude.png
Usage: python 24_output_econ_magnitude.py [--out out]
"""
import argparse, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}; f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
PRESENT_DAY = {2, 3, 5, 8}
def auc(score, label):
    s = pd.Series(score).rank(); pos = label.astype(bool).values
    return float((s[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))

# ======================= judge validation =======================
V = pd.read_csv(OUT / "stance_validation.csv"); V["is_B"] = (V.side == "neg").astype(int)
R["judge"] = {ax: dict(auc=auc(g.score_B, g.is_B), accuracy=float(((g.score_B > .5).astype(int) == g.is_B).mean()), n=int(len(g)),
                       format_mass_median=float(g.format_mass.median())) for ax, g in V.groupby("axis")}
R["judge_gate"] = {ax: o["auc"] >= 0.80 for ax, o in R["judge"].items()}

# ======================= #2 output-level distortion =======================
C = pd.read_csv(OUT / "stance_counterfactual.csv"); ys = C[C.design == "year_swap"].copy()
o2 = {}
for ax, g in ys.groupby("axis"):
    r = cm.fit_cr1(g, "score_B ~ C(cond, Treatment('own')) + C(pid) + C(q)", "pid")
    o2[f"yearswap_{ax}"] = {c: cm.coef(r, f"C(cond, Treatment('own'))[T.{c}]") for c in ("y1955", "y1985", "y2015")}
    o2[f"yearswap_{ax}_means"] = g.groupby("cond").score_B.mean().round(4).to_dict()
cp = pd.read_csv(OUT / "counterfactual_personas.csv"); cp = cp[cp.design == "year_swap"]
iy = cp.pivot_table(index="pid", columns="cond", values="y"); oy = ys[ys.axis == "social"].groupby(["pid", "cond"]).score_B.mean().unstack()
d_int, d_out = iy["y1955"] - iy["y2015"], oy["y1955"] - oy["y2015"]
o2["person_link_social_1955_minus_2015"] = dict(corr=float(np.corrcoef(d_int.loc[d_out.index], d_out)[0, 1]), n=int(len(d_out)))
O = pd.read_csv(OUT / "stance_offset.csv"); O = O[O.cond.isin(["C4_nonpolitical", "C7_nonpolitical_1955"])].copy(); O["k"] = O.pid.str[3:]
for ax, g in O.groupby("axis"):
    w = g.groupby(["k", "cond"]).score_B.mean().unstack(); dd = w["C7_nonpolitical_1955"] - w["C4_nonpolitical"]
    o2[f"ordinary_1955_minus_2023_{ax}"] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))), n=int(len(dd)))
Mb = pd.read_csv(OUT / "stance_members.csv"); Mb = Mb[Mb.party.isin(["Democrat", "Republican"])].copy()
mem = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "members_*.csv")))], ignore_index=True)[["icpsr", "congress", "dim1", "dim2", "x_kim", "y_kim"]]
Mb = Mb.drop(columns=[c for c in ("dim1", "dim2") if c in Mb.columns]).merge(mem, on=["icpsr", "congress"], how="left", validate="m:1")
for ax, g in Mb.groupby("axis"):
    pm = g.groupby(["icpsr", "congress"]).agg(s=("score_B", "mean"), dim1=("dim1", "first"), party=("party", "first"), year=("year", "first"),
                                             era=("era", "first"), x=("x_kim", "first"), y=("y_kim", "first")).reset_index()
    o2[f"members_{ax}_within_party_r_dim1"] = {p: float(np.corrcoef(h.s, h.dim1)[0, 1]) for p, h in pm.groupby("party")}
    o2[f"members_{ax}_rep_minus_dem"] = float(pm[pm.party == "Republican"].s.mean() - pm[pm.party == "Democrat"].s.mean())
    o2[f"members_{ax}_corr_with_internal"] = float(np.corrcoef(pm.s, pm.x if ax == "economic" else pm.y)[0, 1])
    r = cm.fit_cr1(g.assign(decade=(g.year - 1950) / 10), "score_B ~ decade + dim1 + C(party) + C(chamber) + C(q)", "icpsr")
    o2[f"members_{ax}_calendar_per_decade_net_of_record"] = cm.coef(r, "decade")
R["2_outputs"] = o2

# ======================= #3 economic axis =======================
Q = pd.read_csv(OUT / "ordinary_by_question.csv"); o3 = {}
rows = []
for q, g in Q.groupby("q"):
    col = "x" if g.axis.iloc[0] == "economic" else "y"
    b = np.polyfit((g.year - 1900) / 100, g[col], 1)[0]
    rows.append(dict(q=q, axis=g.axis.iloc[0], present_day=q in PRESENT_DAY, slope_per_century=float(b),
                     mean_1955=float(g[g.year == 1955][col].mean()), mean_2023=float(g[g.year == 2023][col].mean())))
QS = pd.DataFrame(rows); QS.to_csv(OUT / "ordinary_question_slopes.csv", index=False)
o3["slopes_by_question"] = QS.round(4).to_dict("records")
o3["mean_slope_by_axis_and_group"] = {f"{a}|{'present_day' if p else 'long_standing'}": float(g.slope_per_century.mean()) for (a, p), g in QS.groupby(["axis", "present_day"])}
Oq = O.copy(); Oq["present_day"] = Oq.q.isin(PRESENT_DAY)
for (ax, pdq), g in Oq.groupby(["axis", "present_day"]):
    w = g.groupby(["k", "cond"]).score_B.mean().unstack(); dd = w["C7_nonpolitical_1955"] - w["C4_nonpolitical"]
    o3[f"outputs_1955_minus_2023_{ax}_{'present_day' if pdq else 'long_standing'}"] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
R["3_economic"] = o3

# ======================= #5 magnitude =======================
mm = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "members_*.csv")))], ignore_index=True); mm = mm[mm.party.isin(["Democrat", "Republican"])]
by = cm.fit_cr1(mm, "y_raw ~ dim1 + C(congress)", "icpsr").params["dim1"]; bx = cm.fit_cr1(mm, "x_raw ~ dim1 + C(congress)", "icpsr").params["dim1"]
ords = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "ordinary_*.csv"))) if "2023.csv" not in p or "ordinary_2023" not in p], ignore_index=True)
pmean = mm.groupby(["congress", "party"])[["dim1", "y_raw", "x_raw"]].mean().unstack("party")
real = {p: float(pmean[("dim1", p)].iloc[-1] - pmean[("dim1", p)].iloc[0]) for p in ("Democrat", "Republican")}
gap_dw_2023 = float(pmean[("dim1", "Republican")].iloc[-1] - pmean[("dim1", "Democrat")].iloc[-1])
gap_y_2023 = float(pmean[("y_raw", "Republican")].iloc[-1] - pmean[("y_raw", "Democrat")].iloc[-1])
ys_int = float((iy["y1955"] - iy["y2015"]).mean())
d = mm.copy(); d["tenure"] = d.year - d.groupby("icpsr").year.transform("min"); d = d[d.groupby("icpsr").congress.transform("size") >= 2]
w = d[["y_raw", "tenure"]] - d.groupby("icpsr")[["y_raw", "tenure"]].transform("mean"); w["icpsr"] = d.icpsr.values
cal = float(cm.fit_cr1(w, "y_raw ~ 0 + tenure", "icpsr").params["tenure"])
o5 = dict(slope_y_on_dim1=float(by), slope_x_on_dim1=float(bx), real_party_change_dim1_1947_2023=real, party_gap_dim1_2023=gap_dw_2023, party_gap_y_2023=gap_y_2023,
          yearswap_1955_minus_2015_y=ys_int, yearswap_in_dim1_units=ys_int / by,
          calendar_drift_y_per_decade=cal * 10, calendar_drift_dim1_units_per_decade=cal * 10 / by, calendar_drift_over_76_years_dim1_units=cal * 76 / by)
o5["ratio_yearswap_to_party_gap_y_2023"] = ys_int / gap_y_2023
o5["ratio_76yr_calendar_drift_to_real_rep_change"] = (cal * 76 / by) / real["Republican"]
if "members_social_rep_minus_dem" in o2:
    yo = o2["yearswap_social"]; out_shift = yo["y1955"]["est"] - yo["y2015"]["est"]
    o5["outputs_social_1955_minus_2015"] = out_shift; o5["outputs_social_rep_minus_dem"] = o2["members_social_rep_minus_dem"]
    o5["outputs_ratio_yearswap_to_party_gap"] = out_shift / o2["members_social_rep_minus_dem"]
R["5_magnitude"] = o5
(OUT / "output_econ_magnitude.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for i, (axn, col) in enumerate((("social", "tab:purple"), ("economic", "tab:green"))):
    m_ = o2[f"yearswap_{axn}_means"]; order = ["y1955", "y1985", "own", "y2015"]
    ax[0].plot(range(4), [m_.get(c) for c in order], "o-", color=col, label=f"{axn} questions")
ax[0].set_xticks(range(4), ["1955", "1985", "own year", "2015"]); ax[0].set_ylabel("judge score (1 = right / traditional)"); ax[0].legend(fontsize=8)
ax[0].set_title("#2 Same legislator, prompt year changed: generated answers", fontsize=9)
for _, r in QS.iterrows():
    ax[1].bar(r.q, r.slope_per_century, color=("tab:red" if r.present_day else "tab:gray"))
ax[1].axhline(0, color="k", lw=.6); ax[1].set_xticks(range(10), [str(i) for i in range(10)]); ax[1].set_xlabel("question (0-4 economic, 5-9 social; red = present-day issue)")
ax[1].set_ylabel("ordinary persons: change per century (x for 0-4, y for 5-9)"); ax[1].set_title("#3 Which questions drift with the prompt year?", fontsize=9)
vals = [abs(o5["yearswap_in_dim1_units"]), abs(o5["calendar_drift_over_76_years_dim1_units"]), abs(real["Republican"]), abs(real["Democrat"]), abs(gap_dw_2023)]
ax[2].bar(["year swap\n1955 vs 2015", "calendar drift\nover 76 years", "real R change\n1947-2023", "real D change\n1947-2023", "R-D gap\n2023"], vals,
          color=["tab:orange", "tab:orange", "tab:red", "tab:blue", "k"])
ax[2].tick_params(axis="x", labelsize=7); ax[2].set_ylabel("DW-NOMINATE dim1 units"); ax[2].set_title("#5 Distortion vs real political change", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_output_econ_magnitude.png", dpi=110)

L = ["# #2, #3, #5 results (generated by 24_output_econ_magnitude.py; exploratory)", "", f"Judge (Qwen2.5-3B, blind to persona and year): {R['judge']}; gate AUC >= 0.80: {R['judge_gate']}", "", "## #2 outputs"]
for ax_ in ("social", "economic"):
    L.append(f"- year swap, {ax_} questions (judge score vs own year): " + "; ".join(f"{k} {f(v)}" for k, v in o2[f"yearswap_{ax_}"].items()) + f"; means {o2[f'yearswap_{ax_}_means']}")
    L.append(f"- ordinary persons 1955 - 2023, {ax_}: {o2[f'ordinary_1955_minus_2023_{ax_}']}")
    L.append(f"- legislators, {ax_}: within-party r with dim1 {o2[f'members_{ax_}_within_party_r_dim1']}; R - D {o2[f'members_{ax_}_rep_minus_dem']:+.3f}; corr with internal position {o2[f'members_{ax_}_corr_with_internal']:+.3f}; calendar per decade net of record {f(o2[f'members_{ax_}_calendar_per_decade_net_of_record'])}")
L.append(f"- person-level link (social, 1955 - 2015): internal vs output shift corr {o2['person_link_social_1955_minus_2015']}")
L += ["", "## #3 economic axis", f"- mean slope per century by axis and question group: {o3['mean_slope_by_axis_and_group']}"]
L += [f"- q{r['q']} ({r['axis']}, {'present-day' if r['present_day'] else 'long-standing'}): slope {r['slope_per_century']:+.3f}" for r in rows]
L += [f"- {k}: {v}" for k, v in o3.items() if k.startswith("outputs_")]
L += ["", "## #5 magnitude", f"- {o5}"]
(OUT / "OUTPUT_ECON_MAGNITUDE.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

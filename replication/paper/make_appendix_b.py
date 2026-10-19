#!/usr/bin/env python
"""Appendix B (supplementary results): every table is generated here from the sealed files and the sealed descriptive follow-up; nothing
is typed. Order follows Sections 4-7. Writes paper/sections/B_supplementary.md (a generated page: edit this script, not the page).
Usage: python paper/make_appendix_b.py"""
import ast, json
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parent; R = P.parent; S = R / "sealed"; D = S / "STUDY_D_quadrant" / "out"
J = lambda p: json.load(open(p, encoding="utf-8"))
M = ["qwen", "llama", "olmo"]; NAME = {"qwen": "Qwen", "llama": "Llama", "olmo": "OLMo"}; Q = ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]
an = {"qwen": J(D / "s4" / "anchors.json")["qwen"], "llama": J(D / "s4" / "anchors.json")["llama"], "olmo": J(D / "s4_olmo" / "anchors.json")["olmo"]}
vm = {"qwen": J(D / "s4" / "vote_vs_map.json")["qwen"], "llama": J(D / "s4" / "vote_vs_map.json")["llama"], "olmo": J(D / "s4_olmo" / "vote_vs_map.json")["olmo"]}
s5 = {"qwen": J(D / "s5" / "S5_RESULTS.json")["qwen"], "llama": J(D / "s5" / "S5_RESULTS.json")["llama"], "olmo": J(D / "s5_olmo" / "S5_RESULTS.json")["olmo"]}
lb = {m: J(S / "STUDY_L_battery" / "out" / m / "RESULTS.json")["blockB_natural_cross_pressure"]["tests"] for m in M}
nop = J(S / "review" / "correction_2026-10-16c" / "no_persona_package_effect.json")
EM = J(R / "post" / "members_by_era" / "members_by_era.json"); TAB = pd.read_csv(R / "post" / "members_by_era" / "members_table.csv")
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:+.3f}".replace("-", "−"); ci = lambda c: f"[{f3(c[0])}, {f3(c[1])}]"
def table(head, rows): return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)

out = ["# Appendix B. Supplementary Results", "",
       "Tables B1 to B7 report, by model, the quantities that the main text summarizes as ranges. Tables B1 to B5 come from the analyses "
       "fixed before the results were computed. Tables B6 and B7 and the legislator columns of Table B2 come from the descriptive follow-ups described in B.5.", ""]

out += ["## B.1 The Map (Section 4)", "", "**Table B1.** Map and roll-call scales by model.", "",
        table(["", "Qwen", "Llama", "OLMo"], [
            ["Layer", *[an[m]["layer"] for m in M]],
            ["Party recognized (share of rows)", *[f2(an[m]["recall"]) for m in M]],
            ["r(economic coordinate, DW-NOMINATE dim. 1), mean within Congress", *[f2(an[m]["r_x_dim1_by_congress_mean"]) for m in M]],
            ["r(economic coordinate, economic scale)", *[f2(an[m]["r_x_economic_scale_within_cc"][0]) for m in M]],
            ["r(social coordinate, abortion scale)", *[f2(an[m]["r_y_abortion_scale_within_cc"][0]) for m in M]],
            ["r(social coordinate, economic scale)", *[f2(an[m]["r_y_economic_scale_within_cc"][0]) for m in M]],
            ["r(economic coordinate, abortion scale)", *[f2(an[m]["r_x_abortion_scale_within_cc"][0]) for m in M]],
            ["Party gap, economic coordinate (SD units)", *[f2(an[m]["party_gap_x_d"]) for m in M]],
            ["Party gap, social coordinate (SD units)", *[f2(an[m]["party_gap_y_d"]) for m in M]],
            ["Agreement of map and voting quadrants", *[f2(vm[m]["agreement"]) for m in M]],
            *[[f"Recall, {q}", *[f2(vm[m]["recall_by_vote_quadrant"][q]) for m in M]] for q in Q]]),
        "", "The DW-NOMINATE entry is the mean of correlations computed separately in each Congress. The issue-scale correlations pool recognized rows "
        "after demeaning within Congress and chamber. Party recognition uses all " + f"{an['qwen']['n_rows']:,}" + " map rows; recall and agreement use the "
        + f"{vm['qwen']['n']:,}" + " Democratic and Republican rows with both voting scales (93rd to 117th Congress) in every model.", ""]

T = TAB.copy()
mark = lambda r, m: "–" if pd.isna(r[m]) else ("yes" if r[m] == r.vote_q else r[m])
T["years"] = T["first"].astype(str) + "–" + T["last"].astype(str)   # first and last year of the member's rows in the analysis sample (from 1973)
rows = [[r.vote_q, r["name"], r.party[0], r.chamber, r.state, r.years, r.how, *[mark(r, m) for m in M]] for _, r in T.iterrows()]
out += ["## B.2 Legislators Selected by Their Votes (Sections 4 and 7)", "",
        "**Table B2.** The 60 legislators selected by their votes and where each model places them. \"yes\" means placed in the voting quadrant; "
        "otherwise the cell gives the map quadrant; a dash means the model does not recognize the member's party in the selected rows. "
        "\"deepest\" members are the ten farthest from both medians in their quadrant; \"random\" members are drawn from the rest.", "",
        table(["Voting quadrant", "Legislator", "Party", "Chamber", "State", "Years in sample", "Selection", "Qwen", "Llama", "OLMo"], rows), ""]

H = ["H3 consistency", "H4 quadrant reached", "H1 choice uncertainty", "H2 generation uncertainty"]; K = ["H3_C2", "H4_C3hit", "H1_C1a", "H2_C4"]
rows = []
for m in M:
    for h, k in zip(H, K):
        v = s5[m]["hyp"][k]; sc = s5[m]["sensitivity_base_cluster_A14"][k]
        rows.append([NAME[m], h, f2(v["aligned_mean"]), f2(v["cross_mean"]), f3(v["est"]), ci(v["ci95"]), ci(sc["ci95"]), f"{v['min_effect']:+.2f}".replace("-", "−"), "yes" if v["supported"] else "no"])
out += ["## B.3 Stated Profiles (Section 5)", "", "**Table B3.** Cross-pressured profiles are followed less consistently and reach their quadrant less often in all three models; choice "
        "uncertainty rises in two models and generation uncertainty in none beyond the minimum. Preregistered tests: cross-pressured minus aligned profiles with fixed effects for the way the profile "
        "is stated. Primary intervals resample conditions within profile and way; the sensitivity intervals resample base prompts.", "",
        table(["Model", "Outcome", "Aligned mean", "Cross-pressured mean", "Difference", "95% interval", "Base-prompt interval", "Minimum", "Supported"], rows), ""]
WAY = {"I1": "anonymous member, system prompt", "I5": "user-turn instruction", "I2": "extraction instructions", "I3": "four examples", "I4": "named real member"}
rows = [[NAME[m], WAY[w], f3(s5[m]["per_mode_cross_minus_aligned"][w]["C2"]), f3(s5[m]["per_mode_cross_minus_aligned"][w]["hit"])] for m in M for w in ["I1", "I5", "I2", "I3", "I4"]]
out += ["**Table B4.** Cross-pressured minus aligned profiles by the way the profile is stated (descriptive means).", "",
        table(["Model", "Way of stating the profile", "Consistency", "Quadrant reached"], rows), ""]

rows = []
for m in M:
    Hf = ast.literal_eval(lb[m]["entropy_at_NONE_by_party_family"])
    for p, b in (("Democrat", "B1_Democrat"), ("Republican", "B2_Republican")):
        v = lb[m][b]
        rows.append([NAME[m], f"{'Democratic' if p == 'Democrat' else 'Republican'} personas", f"{Hf[(p, 'party_like')]:.3f}", f"{Hf[(p, 'reversed')]:.3f}", f3(v["est"]), ci(v["ci95"]), f"{v['p_holm']:.3f}",
                     "yes" if v["supported"] else "no", f2(lb[m]["own_package_at_NONE_by_party"][p]), f"{round(lb[m]['economic_dominance_share_by_party'][p] * lb[m]['decided_units_by_party'][p]) / lb[m]['decided_units_by_party'][p]:.2f} ({lb[m]['decided_units_by_party'][p]})"])
    i = nop[m]["interval_2026_10_18"]
    rows.append([NAME[m], "no persona", f"{nop[m]['party_like']:.3f}" if isinstance(nop[m].get("party_like"), (int, float)) else "–",
                 f"{nop[m]['reversed']:.3f}" if isinstance(nop[m].get("reversed"), (int, float)) else "–", f3(i["est"]), ci(i["ci95_pigeonhole"]), "–", "–", "–", "–"])
out += ["## B.4 Choices Between Packages (Section 6)", "", "**Table B5.** Crossed choices are more uncertain than party-line choices in Qwen and OLMo and less uncertain in Llama with personas. Entropy of party-line and crossed choices (bits), the crossed minus party-line "
        "contrast with its 95 percent interval (wild cluster bootstrap for personas; two-way passage bootstrap for the no-persona arm), the "
        "Holm-adjusted p value, the mean probability given to the persona's own package on party-line items, and the share of decided crossed "
        "choices that go to the economic position (number of decided choices). The no-persona arm has no registered test.", "",
        table(["Model", "Chooser", "Party-line entropy", "Crossed entropy", "Difference", "95% interval", "Holm p", "Supported", "Own package", "Economic share (n)"], rows), ""]

rows = [[NAME[m], ("1973–79" if d == "1970" else ("2011–21" if d == "2010" else d + "s")), f"{v['rows']:,}", f"{100 * v['cross_share']:.0f}%", f2(v["recall_aligned"]), f2(v["recall_cross"])]
        for m in M for d, v in EM["rows"][m]["by_decade"].items()]
out += ["## B.5 Descriptive Follow-Up (Section 7)", "",
        "The legislator columns of Table B2, Table B6 and Table B7 were computed after the analyses above had been sealed, from the same stored maps and "
        "roll-call scales. They are descriptive, and no test was registered for them.", "",
        "**Table B6.** Recall of aligned rows rises across periods while recall of cross-pressured rows stays low. Democratic and Republican member-Congress rows by period: number of rows, share cross-pressured, and recall of aligned and cross-pressured rows.", "",
        table(["Model", "Period", "Rows", "Cross-pressured", "Recall, aligned", "Recall, cross-pressured"], rows), ""]
BM = J(R / "post" / "benchmark" / "benchmark_recall.json"); n1 = BM["N1_dwnominate"]["recall"]
rows = [["DW-NOMINATE map (dimensions 1 and 2)", *[f2(n1[q]) for q in Q], f2(n1["aligned_pooled"]), f2(n1["cross_pooled"]), f"{n1['n']:,}"]]
for m in M:
    for lab, k in (("all rows", "all_rows"), ("recognized rows", "recognized_rows")):
        r = BM["N2_recognized"][m][k]
        rows.append([f"{NAME[m]} map, {lab}", *[f2(r[q]) for q in Q], f2(r["aligned_pooled"]), f2(r["cross_pooled"]), f"{r['n']:,}"])
out += ["**Table B7.** A map built from DW-NOMINATE places most cross-pressured rows in their voting quadrant, partly because its two nearly "
        "unrelated dimensions put about half of all rows in the cross-pressured quadrants; the models do not, and restricting them to rows whose "
        "party they recognize changes little. Recall by voting quadrant and pooled over aligned and over cross-pressured rows. "
        "Within Congress and chamber, DW-NOMINATE's two dimensions correlate at " + f"{BM['N1_dwnominate']['r_dim1_dim2_within_cc']:.2f}".replace("-", "−")
        + "; the models' two coordinates correlate at " + ", ".join(f"{v:.2f}" for v in BM["r_x_y_within_cc"].values()) + " (Qwen, Llama, OLMo).", "",
        table(["Map", *[f"Recall {q}" for q in Q], "Aligned (pooled)", "Cross-pressured (pooled)", "Rows"], rows), ""]
(P / "sections" / "B_supplementary.md").write_text("\n".join(out), encoding="utf-8")
print("B_supplementary.md written:", sum(1 for l in out if l.startswith("**Table")), "tables")

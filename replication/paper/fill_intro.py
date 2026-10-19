#!/usr/bin/env python
"""Fill paper/introduction.template.md and paper/sections/*.template.md -> the same names without .template. Every number in the text is computed here from the sealed result
files in sealed/ (the same files that run_analysis.sh reproduces and compare_to_sealed.py checks); none is typed. Also writes
paper/intro_values.json so each value can be traced. Usage: python paper/fill_intro.py [--from-reproduced]
  --from-reproduced reads the reproduced outputs (after run_analysis.sh) instead of sealed/; the text should not change."""
import json, re, sys
from pathlib import Path
P = Path(__file__).resolve().parent; R = P.parent
ROOT = R if "--from-reproduced" in sys.argv else R / "sealed"
M = ["qwen", "llama", "olmo"]; NAME = {"qwen": "Qwen", "llama": "Llama", "olmo": "OLMo"}
def J(rel): return json.load(open(ROOT / rel, encoding="utf-8"))
def d(m, base): return f"STUDY_D_quadrant/out/{base}_olmo" if m == "olmo" else f"STUDY_D_quadrant/out/{base}"
def rng(xs, k=2): return f"{min(xs):.{k}f} to {max(xs):.{k}f}"
def pct(xs):
    lo, hi = round(100 * min(xs)), round(100 * max(xs)); return f"{lo} percent" if lo == hi else f"{lo} to {hi} percent"
def sg(x, k=2): return f"{x:+.{k}f}".replace("-", "−")
def ci(c, k=2): return f"[{sg(c[0], k)}, {sg(c[1], k)}]"

an = {m: J(d(m, "s4") + "/anchors.json")[m] for m in M}
vm = {m: J(d(m, "s4") + "/vote_vs_map.json")[m] for m in M}
s4c = {m: J(d(m, "s4c") + "/S4C.json")[m] for m in M}
s5 = {m: J(d(m, "s5") + "/S5_RESULTS.json")[m]["hyp"] for m in M}
lb = {m: J(f"STUDY_L_battery/out/{m}/RESULTS.json")["blockB_natural_cross_pressure"]["tests"] for m in M}
npf = R / ("post/no_persona/no_persona_package_effect.json" if ROOT == R else "sealed/review/correction_2026-10-16c/no_persona_package_effect.json")
nop = {m: json.load(open(npf, encoding="utf-8"))[m]["interval_2026_10_18"] for m in M}

ALIGNED, CROSS = ["EL-SP", "ER-ST"], ["EL-ST", "ER-SP"]
mapped_aligned = {m: sum(vm[m]["confusion_vote_rows_map_cols"][v][q] for v in vm[m]["confusion_vote_rows_map_cols"] for q in ALIGNED) / vm[m]["n"] for m in M}
V = {
    "r_econ": rng([an[m]["r_x_economic_scale_within_cc"][0] for m in M]),
    "r_abort": rng([an[m]["r_y_abortion_scale_within_cc"][0] for m in M]),
    "n_rows": f"{vm['qwen']['n']:,}",
    "rows_equal_across_models": all(vm[m]["n"] == vm["qwen"]["n"] for m in M),
    "cross_share": pct([sum(sum(vm[m]["confusion_vote_rows_map_cols"][v].values()) for v in CROSS) / vm[m]["n"] for m in M]),
    # B1 (author 2026-10-19): pooled recall over the aligned (or cross-pressured) rows of each model, range over models; per-quadrant values in Fig. 1d, Table B1
    "recall_aligned": pct([sum(vm[m]["confusion_vote_rows_map_cols"][q][q] for q in ALIGNED) / sum(sum(vm[m]["confusion_vote_rows_map_cols"][q].values()) for q in ALIGNED) for m in M]),
    "recall_cross": pct([sum(vm[m]["confusion_vote_rows_map_cols"][q][q] for q in CROSS) / sum(sum(vm[m]["confusion_vote_rows_map_cols"][q].values()) for q in CROSS) for m in M]),
    "mapped_aligned": pct(list(mapped_aligned.values())),
    "a12_aligned": f"{min(s4c[m][q]['placed'] for m in M for q in ALIGNED)} to {max(s4c[m][q]['placed'] for m in M for q in ALIGNED)}",
    "a12_cross": f"{min(s4c[m][q]['placed'] for m in M for q in CROSS)} to {max(s4c[m][q]['placed'] for m in M for q in CROSS)}",
    "a12_n": str(s4c["qwen"]["EL-SP"]["n"]),
    "h3": rng([abs(s5[m]["H3_C2"]["est"]) for m in M], 3),           # text says "lower by", so magnitudes
    "h4": rng([abs(s5[m]["H4_C3hit"]["est"]) for m in M], 3),
    "h34_all_negative": all(s5[m][h]["est"] < 0 for m in M for h in ("H3_C2", "H4_C3hit")),
    "h34_all_supported": all(s5[m][h]["supported"] for m in M for h in ("H3_C2", "H4_C3hit")),
    "b_qwen": f"{sg(lb['qwen']['B1_Democrat']['est'])} and {sg(lb['qwen']['B2_Republican']['est'])}",
    "b_olmo": f"{sg(lb['olmo']['B1_Democrat']['est'])} and {sg(lb['olmo']['B2_Republican']['est'])}",
    "b_llama": f"{sg(lb['llama']['B1_Democrat']['est'])} and {sg(lb['llama']['B2_Republican']['est'])}",
    "b_supported_qwen_olmo": all(lb[m][b]["supported"] for m in ("qwen", "olmo") for b in ("B1_Democrat", "B2_Republican")),
    "np_qwen": f"{sg(nop['qwen']['est'])} {ci(nop['qwen']['ci95_pigeonhole'])}",
    "np_olmo": f"{sg(nop['olmo']['est'])} {ci(nop['olmo']['ci95_pigeonhole'])}",
    "np_llama": f"{sg(nop['llama']['est'])} {ci(nop['llama']['ci95_pigeonhole'])}",
}
assert V["rows_equal_across_models"] and V["h34_all_supported"] and V["h34_all_negative"] and V["b_supported_qwen_olmo"], "a statement in the text would be false"
# Section 3 (design): layers, rows, recall, condition and pair counts, all from the same sealed files
s5n = {m: J(d(m, "s5") + "/S5_RESULTS.json")[m]["n_conditions"] for m in M}
V.update({
    "layer_qwen": str(an["qwen"]["layer"]), "layer_llama": str(an["llama"]["layer"]), "layer_olmo": str(an["olmo"]["layer"]),
    "n_map_rows": f"{an['qwen']['n_rows']:,}",
    "recall_range": pct([an[m]["recall"] for m in M]),
    "s5_n": str(s5n["qwen"]),
    "np_n_pairs": str(json.load(open(npf, encoding="utf-8"))["qwen"]["interval_2026_10_18"]["n_pairs"]),
})
assert all(an[m]["n_rows"] == an["qwen"]["n_rows"] for m in M) and all(s5n[m] == s5n["qwen"] for m in M), "Section 3 states one count for all models"
# Section 4 (map): anchors and vote_vs_map, same files as above
conf = {m: vm[m]["confusion_vote_rows_map_cols"] for m in M}
cross_to_aligned = {m: sum(conf[m][v][q] for v in CROSS for q in ALIGNED) / sum(sum(conf[m][v].values()) for v in CROSS) for m in M}
elst_gap = {m: vm[m]["recall_by_vote_quadrant"]["EL-ST"] - vm[m]["recall_by_vote_quadrant"]["ER-SP"] for m in M}
V.update({
    "r_dim1": rng([an[m]["r_x_dim1_by_congress_mean"] for m in M]),
    "r_cross_track": rng([an[m][k][0] for m in M for k in ("r_y_economic_scale_within_cc", "r_x_abortion_scale_within_cc")]),
    "gap_x": rng([an[m]["party_gap_x_d"] for m in M]), "gap_y": rng([an[m]["party_gap_y_d"] for m in M]),
    "agree": pct([vm[m]["agreement"] for m in M]),
    "cross_to_aligned": pct(list(cross_to_aligned.values())),
    "elst_minus_ersp": rng(list(elst_gap.values())),
    "qwen_r_y_econ": f"{an['qwen']['r_y_economic_scale_within_cc'][0]:.2f}", "qwen_r_y_abort": f"{an['qwen']['r_y_abortion_scale_within_cc'][0]:.2f}",
})
assert all(an[m]["party_gap_x_d"] > an[m]["party_gap_y_d"] for m in M), "Section 4: economic gap larger in every model"
assert all(g > 0 for g in elst_gap.values()), "Section 4: EL-ST recall above ER-SP recall in every model"
# Section 5 (stated profiles): S5_RESULTS.json, same files as above
s5all = {m: J(d(m, "s5") + "/S5_RESULTS.json")[m] for m in M}
pm = {m: s5all[m]["per_mode_cross_minus_aligned"] for m in M}
V.update({
    "c2_aligned": rng([s5[m]["H3_C2"]["aligned_mean"] for m in M]), "c2_cross": rng([s5[m]["H3_C2"]["cross_mean"] for m in M]),
    "hit_aligned": pct([s5[m]["H4_C3hit"]["aligned_mean"] for m in M]), "hit_cross": pct([s5[m]["H4_C3hit"]["cross_mean"] for m in M]),
    "h1_llama": f"{sg(s5['llama']['H1_C1a']['est'], 3)} {ci(s5['llama']['H1_C1a']['ci95'], 3)}",
    "h1_olmo": f"{sg(s5['olmo']['H1_C1a']['est'], 3)} {ci(s5['olmo']['H1_C1a']['ci95'], 3)}",
    "h1_qwen": f"{sg(s5['qwen']['H1_C1a']['est'], 3)} {ci(s5['qwen']['H1_C1a']['ci95'], 3)}",
    "h2": rng([s5[m]["H2_C4"]["est"] for m in M], 3), "h2_min": f"{s5['qwen']['H2_C4']['min_effect']:.2f}",
    "n_mode_cells": str(sum(len(pm[m]) for m in M)),
    "hit_neg_cells": str(sum(v["hit"] < 0 for m in M for v in pm[m].values())),
})
assert all(v["C2"] < 0 for m in M for v in pm[m].values()), "Section 5: consistency gap in every model x way"
assert [m for m in M if s5[m]["H1_C1a"]["supported"]] == ["llama", "olmo"], "Section 5: H1 supported in Llama and OLMo only"
assert not any(s5[m]["H2_C4"]["supported"] for m in M) and all(s5[m]["H2_C4"]["ci95"][0] > 0 for m in M), "Section 5: H2 above zero, below minimum"
assert all(s5[m]["H2_C4"]["est"] < s5[m]["H2_C4"]["min_effect"] for m in M)
# Section 6 (package choices): Study L RESULTS.json and the no-persona file, same files as above
import ast
own = {m: lb[m]["own_package_at_NONE_by_party"] for m in M}
Hf = {m: ast.literal_eval(lb[m]["entropy_at_NONE_by_party_family"]) for m in M}
esh = {m: lb[m]["economic_dominance_share_by_party"] for m in M}; dec = {m: lb[m]["decided_units_by_party"] for m in M}
QO = ("qwen", "olmo"); P2 = ("Democrat", "Republican")
def eshx(m, p):   # sealed share is rounded to 3 decimals; recover the count of economic choices and report the exact fraction
    n = dec[m][p]; k = round(esh[m][p] * n); assert abs(k / n - esh[m][p]) < 0.0005 + 1e-9, (m, p); return k / n
V.update({
    "own_qo": rng([own[m][p] for m in QO for p in P2]), "own_llama_d": f"{own['llama']['Democrat']:.2f}", "own_llama_r": f"{own['llama']['Republican']:.2f}",
    "H_party_qo": rng([Hf[m][(p, "party_like")] for m in QO for p in P2], 3), "H_cross_qo": rng([Hf[m][(p, "reversed")] for m in QO for p in P2], 3),
    "H_party_llama": rng([Hf["llama"][(p, "party_like")] for p in P2], 3), "H_cross_llama": rng([Hf["llama"][(p, "reversed")] for p in P2], 3),
    "esh_qwen": f"{eshx('qwen', 'Democrat'):.2f} and {eshx('qwen', 'Republican'):.2f}", "esh_olmo": f"{eshx('olmo', 'Democrat'):.2f} and {eshx('olmo', 'Republican'):.2f}",
    "esh_llama_r": f"{eshx('llama', 'Republican'):.2f}", "esh_llama_d": f"{eshx('llama', 'Democrat'):.2f}",
    "dec_llama_d": str(dec["llama"]["Democrat"]), "dec_llama_r": str(dec["llama"]["Republican"]),
    "esh_cells_over_half": ["zero", "one", "two", "three", "four", "five", "six"][sum(esh[m][p] > 0.5 for m in M for p in P2)],
})
assert all(Hf[m][(p, "party_like")] < Hf[m][(p, "reversed")] for m in QO for p in P2) and all(Hf["llama"][(p, "party_like")] > Hf["llama"][(p, "reversed")] for p in P2)
assert all(own["llama"][p] < min(own[m][q] for m in QO for q in P2) for p in P2), "Section 6: Llama holds the party line less"
assert not any(lb["llama"][b]["supported"] for b in ("B1_Democrat", "B2_Republican")) and all(lb["llama"][b]["ci95"][1] < 0 for b in ("B1_Democrat", "B2_Republican"))
assert nop["llama"]["ci95_pigeonhole"][0] < 0 < nop["llama"]["ci95_pigeonhole"][1] and all(nop[m]["ci95_pigeonhole"][0] > 0 for m in QO)
assert esh["llama"]["Democrat"] < 0.5 and V["esh_cells_over_half"] == "five"
# Section 4 additions (council N1, N2, 2026-10-19): post/benchmark/benchmark_recall.json, sealed in SEAL.json there; not in sealed/
BM = json.load(open(R / "post" / "benchmark" / "benchmark_recall.json", encoding="utf-8")); n1 = BM["N1_dwnominate"]; n2 = BM["N2_recognized"]
V.update({
    "n1_cross": f"{100 * n1['recall']['cross_pooled']:.0f} percent", "n1_aligned": f"{100 * n1['recall']['aligned_pooled']:.0f} percent",
    "n1_r": f"{n1['r_dim1_dim2_within_cc']:.2f}".replace("-", "−"), "xy_r": rng(list(BM["r_x_y_within_cc"].values())),
    "n2_aligned": pct([n2[m]["recognized_rows"]["aligned_pooled"] for m in M]), "n2_cross": pct([n2[m]["recognized_rows"]["cross_pooled"] for m in M]),
})
assert n1["recall"]["cross_pooled"] > max(n2[m]["all_rows"]["cross_pooled"] for m in M), "N1 rule: DW-NOMINATE places more cross-pressured rows than every model"
assert all(abs(n2[m]["all_rows"]["cross_pooled"] - n2[m]["recognized_rows"]["cross_pooled"]) < 0.03 for m in M), "N2: recognition changes little"
assert all(abs(n2[m]["all_rows"]["cross_pooled"] - (sum(vm[m]["confusion_vote_rows_map_cols"][q][q] for q in CROSS) / sum(sum(vm[m]["confusion_vote_rows_map_cols"][q].values()) for q in CROSS))) < 1e-12 for m in M)
# Council 2 (2026-10-19) asserts for statements added in Sections 4 and 7
assert all(an[m]["r_y_economic_scale_within_cc"][0] > an[m]["r_y_abortion_scale_within_cc"][0] for m in M), "Section 4: social coordinate follows economic scale more in every model"
assert abs(n1["r_dim1_dim2_within_cc"]) < 0.1, "Section 4: DW-NOMINATE dimensions nearly unrelated, so a median split puts about half of rows off the diagonal"
# Section 7 (discussion): descriptive follow-up post/members_by_era/members_by_era.json (sealed in SEAL.json there); not in sealed/
EM = json.load(open(R / "post" / "members_by_era" / "members_by_era.json", encoding="utf-8"))
mem, rws = EM["members"], EM["rows"]; cq = rws["qwen"]["cross_rows_party_by_quadrant"]; md_ = mem["misplaced_destination"]
dec = {m: rws[m]["by_decade"] for m in M}
assert all(min(dec[m], key=lambda d: dec[m][d]["recall_cross"]) == "2010" for m in M), "Section 7: cross-pressured recall lowest in 2011-21 in every model"
tab = __import__("pandas").read_csv(R / "post" / "members_by_era" / "members_table.csv").set_index("name")
def quad_of(name, m): return tab.loc[name, m]
V.update({
    "elst_dem": f"{cq['EL-ST']['Democrat']:,}", "elst_all": f"{sum(cq['EL-ST'].values()):,}",
    "ersp_rep": f"{cq['ER-SP']['Republican']:,}", "ersp_all": f"{sum(cq['ER-SP'].values()):,}",
    "cross_none": str(mem["cross_placed_by_k_models"]["0"]),
    "cross_none3": str(int(((tab.vote_q.isin(["EL-ST", "ER-SP"])) & tab[M].notna().all(axis=1) & (tab[M].ne(tab.vote_q, axis=0)).all(axis=1)).sum())),
    "cross_none_missing": ["no", "one", "two", "three"][int(((tab.vote_q.isin(["EL-ST", "ER-SP"])) & (tab.n_placed == 0) & tab[M].isna().any(axis=1)).sum())], "cross_all3": ["none", "one", "two", "three"][mem["cross_placed_by_k_models"]["3"]],
    "aligned_all3": str(mem["aligned_placed_by_k_models"]["3"]),
    "rep_to_erst": "; ".join(f"{NAME[m]} {md_[m]['ER-SP'].get('ER-ST', 0)} of {sum(md_[m]['ER-SP'].values())}" for m in M),
    "dem_to_elsp_lo": f"{md_['llama']['EL-ST'].get('EL-SP', 0)} of {sum(md_['llama']['EL-ST'].values())}",
    "dem_to_elsp_oo": f"{md_['olmo']['EL-ST'].get('EL-SP', 0)} of {sum(md_['olmo']['EL-ST'].values())}",
    "cross_1970s": str(mem["cross_first_decade"]["1970"]),
    "share_70s": f"{100 * dec['qwen']['1970']['cross_share']:.0f} percent", "share_10s": f"{100 * dec['qwen']['2010']['cross_share']:.0f} percent",
    "ra_70s_10s": "; ".join(f"{NAME[m]} {dec[m]['1970']['recall_aligned']:.2f} to {dec[m]['2010']['recall_aligned']:.2f}" for m in M),
    "rc_range": rng([dec[m][d]["recall_cross"] for m in M for d in dec[m]]),  # council 2: text says lowest in 2010 decade, asserted below
})
# every statement about named legislators in Section 7
assert sum(cq["EL-ST"].values()) > 0 and cq["EL-ST"]["Democrat"] > cq["EL-ST"]["Republican"] and cq["ER-SP"]["Republican"] > cq["ER-SP"]["Democrat"]
sen = {s["name"]: s for s in mem["senators_cross"]}
assert [n for n, s in sen.items() if all(s[m] for m in M)] == ["Jennings Randolph"], "only Randolph placed by all three among the senators"
assert all(quad_of("Robert Packwood", m) == "ER-ST" for m in M) and all(quad_of("Walter Huddleston", m) == "ER-ST" for m in M)
assert quad_of("Thomas Eagleton", "qwen") == "EL-ST" and quad_of("Thomas Eagleton", "llama") == quad_of("Thomas Eagleton", "olmo") == "EL-SP"
assert not any(sen["Hiram Fong"][m] for m in M) and set(sen) == {"Thomas Eagleton", "John Melcher", "John Pastore", "Walter Huddleston", "Jennings Randolph",
                                                             "Howell Heflin", "Hiram Fong", "Robert Packwood", "John Chafee"}
_per = ["1970", "1980", "1990", "2000", "2010"]
assert all(dec[m][b]["recall_aligned"] > dec[m][a]["recall_aligned"] for m in M for a, b in zip(_per, _per[1:])), "aligned recall rises in every step"
assert sum(quad_of("Thomas Eagleton", m) != "EL-ST" for m in M) == 2 and all(quad_of(n, m) != tab.loc[n, "vote_q"] for n in ("Robert Packwood", "Walter Huddleston") for m in M)
sens = {m: s5all[m]["sensitivity_base_cluster_A14"] for m in M}
assert all((sens[m][h]["ci95"][0] > 0 or sens[m][h]["ci95"][1] < 0) == (s5[m][h]["ci95"][0] > 0 or s5[m][h]["ci95"][1] < 0)
           for m in M for h in s5[m]), "Section 5: base-prompt bootstrap gives the same zero-exclusion for every test"
assert an["qwen"]["r_y_economic_scale_within_cc"][0] > an["qwen"]["r_y_abortion_scale_within_cc"][0], "Section 4: Qwen social coordinate follows the economic scale more"
assert all(s4c[m][q]["n"] == 15 for m in M for q in ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]), "Section 3 describes 15 members per quadrant in every model"
assert all(json.load(open(npf, encoding="utf-8"))[m]["interval_2026_10_18"]["n_pairs"] == 54 for m in M), "54 pairs in every model"
def _holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
for m in M:   # the preregistered rule (interval excludes zero, Holm < .05, minimum) gives the same verdicts under the base-prompt bootstrap
    hs = list(s5[m]); adj = _holm([sens[m][h]["p_boot"] for h in hs])
    for h, a in zip(hs, adj):
        v, c = s5[m][h], sens[m][h]["ci95"]
        ok = (c[0] > 0 or c[1] < 0) and a < 0.05 and (v["est"] >= v["min_effect"] if v["min_effect"] > 0 else v["est"] <= v["min_effect"])
        assert ok == v["supported"], (m, h, "base-prompt verdict differs")
PAGES = [(P / "abstract.template.md", P / "abstract.md"), (P / "introduction.template.md", P / "introduction.md")] + [(f, f.with_name(f.name.replace(".template", ""))) for f in sorted((P / "sections").glob("*.template.md"))]
for src, dst in PAGES:
    parts = re.split(r"(<!--.*?-->)", src.read_text(encoding="utf-8"), flags=re.S)   # text inside HTML comments is left as written
    body = parts[0::2]
    missing = sorted({k for b in body for k in re.findall(r"\{\{(\w+)\}\}", b)} - set(V)); assert not missing, (src.name, missing)
    n = sum(len(re.findall(r"\{\{\w+\}\}", b)) for b in body)
    parts[0::2] = [re.sub(r"\{\{(\w+)\}\}", lambda m_: str(V[m_.group(1)]), b) for b in body]
    dst.write_text("".join(parts), encoding="utf-8")
    print(f"{dst.relative_to(P)} written;", n, "placeholders filled from", ROOT.name)
_ab = re.sub(r"<!--.*?-->", "", (P / "abstract.md").read_text(encoding="utf-8"), flags=re.S).replace("# Abstract", "")
_n = len(re.findall(r"\S+", _ab)); assert 150 <= _n <= 200, f"abstract has {_n} words; SSCR asks for 150-200"
print("abstract words:", _n)
json.dump({"source": str(ROOT.relative_to(R)) or ".", "values": V, "mapped_aligned_by_model": mapped_aligned},
          open(P / "intro_values.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

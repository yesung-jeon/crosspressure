#!/usr/bin/env python
"""l2 (PLAN v0.1 sections 6-9): one model. Units; Block D (direction asymmetry of lens flips, party-like items, norm H, ten placebos:
P1 D+ - placebo mean and rank among placebos, P2 D+ - D-, P4 D+ - S-; Holm over P1, P2, P4); Block B (natural cross-pressure at NONE:
reversed - party-like entropy by party, B1 Democrats, B2 Republicans, B3 difference; economic-dominance shares); Block C (displacement on
reversed DECIDED items with the pure diagonals: C1 entropy, C2 switch, push toward the losing vs the winning package; Holm); Block S
(single-issue specificity matrix); secondary: norm L, lens entropy, decision layer, by party, model arm, map check, placebo spread.
-> out/<model>/RESULTS.json, tables/. Usage: python l2_analyze.py --model qwen [--draws 9999] [--smoke]"""
import argparse, glob, json
import numpy as np, pandas as pd
import l_common as lc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--draws", type=int, default=9999); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
base = lc.out_dir(a.model, *(["smoke"] if a.smoke else [])); T = lc.out_dir(a.model, *(["smoke"] if a.smoke else []), "tables")
PARTIES = ("Democrat", "Republican"); PLAC = {s: [f"PLAC_{s}_{i}" for i in range(lc.N_PLAC)] for s in ("H", "L")}

def wcb(df, y, X, k, draws, seed, cl="pid", grp="g"):
    d = df[[y, cl, grp] + X].dropna()
    if len(d) == 0 or d[cl].nunique() < 2: return None
    dm = d.groupby(grp)[[y] + X].transform("mean")
    yv = (d[y] - dm[y]).values; Xv = (d[X] - dm[X]).values; c = pd.factorize(d[cl])[0]; G = c.max() + 1
    if np.linalg.matrix_rank(Xv) < len(X): return None
    P = np.linalg.pinv(Xv); b = P @ yv; u = yv - Xv @ b
    def se(res): s = np.bincount(c, weights=P[k] * res, minlength=G); return float(np.sqrt(G / (G - 1) * (s ** 2).sum()))
    s0 = se(u); t0 = b[k] / s0 if s0 > 0 else np.nan; Xr = np.delete(Xv, k, 1)
    fr = Xr @ (np.linalg.pinv(Xr) @ yv) if Xr.shape[1] else np.zeros_like(yv); ur = yv - fr; fu = Xv @ b
    rng = np.random.default_rng(seed); ts, bs = np.empty(draws), np.empty(draws)
    for i in range(draws):
        w = rng.choice([-1.0, 1.0], G)[c]; ys = fr + w * ur; bb = P @ ys; s = se(ys - Xv @ bb); ts[i] = bb[k] / s if s > 0 else 0.0; bs[i] = (P @ (fu + w * u))[k]
    return {"est": float(b[k]), "se_cr1": s0, "p_wcb": float(np.mean(np.abs(ts) >= abs(t0))) if np.isfinite(t0) else None,
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d)), "clusters": int(G)}

def contrast(S, y, hi, lo, draws, seed, col="cond", grp="g"):
    s = S[S[col].isin(hi + lo)].copy()
    if s[col].isin(hi).sum() == 0 or s[col].isin(lo).sum() == 0: return None
    s["x"] = s[col].isin(hi).astype(float); r = wcb(s, y, ["x"], 0, draws, seed, grp=grp)
    if r is not None: r["hi"], r["lo"], r["y"] = hi, lo, y
    return r

def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def rule_ge(v, m): return bool(v is not None and v["est"] >= m and v["ci95"][0] > 0 and v.get("p_holm") is not None and v["p_holm"] < 0.05)
def apply_holm(fam, names):
    ok = [n for n in names if fam.get(n) is not None and fam[n].get("p_wcb") is not None]
    for n, adj in zip(ok, holm(np.array([fam[n]["p_wcb"] for n in ok]))): fam[n]["p_holm"] = float(adj)
def short(v): return {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk in ("est", "ci95", "p_holm", "supported", "rank", "p_rank", "n", "clusters")} if isinstance(v, dict) and "est" in v else v

def units():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(base / "main" / "part_*.csv.gz")))], ignore_index=True)
    D["mass"] = D.pA + D.pB; D["pA_n"] = np.where(D.order == 0, D.pA, D.pB) / D.mass
    D["final_H"] = lc.hbits(D.pA / D.mass); N = int(json.load(open(base / "main" / "pushes.json", encoding="utf-8"))["n_layers"]); D["DLf"] = D.DL / N
    keys = ["pid", "party", "cond", "kind", "strength", "vec", "topic", "family", "kang", "origin"]
    U = D.groupby(keys, as_index=False, dropna=False)[["final_H", "DLf", "lens_H", "flips", "pA_n", "mass"]].mean()
    U["g"] = U.pid + "|" + U.topic + "|" + U.family; U["own_p"] = np.where(U.party == "Republican", 1 - U.pA_n, U.pA_n)
    n0 = U[U.cond == "NONE"].set_index("g"); U["pA0"] = U.g.map(n0.pA_n); U["H0"] = U.g.map(n0.final_H); U["flips0"] = U.g.map(n0.flips)
    def winner(r):
        if r.family != "reversed" or np.isnan(r.pA0): return "na"
        return ("soc" if r.pA0 > 0.5 else "econ") if r.party == "Republican" else ("econ" if r.pA0 > 0.5 else "soc")
    U["winner"] = [winner(r) for r in U.itertuples()]; U["decided"] = (U.pA0 - 0.5).abs() >= 0.10
    U["winner_pkg"] = np.where(U.winner == "na", "na", np.where(U.pA0 > 0.5, "EL-ST", "ER-SP"))      # package A = EL-ST on reversed items
    def now(r):
        if r.family != "reversed": return "na"
        return ("soc" if r.pA_n > 0.5 else "econ") if r.party == "Republican" else ("econ" if r.pA_n > 0.5 else "soc")
    U["winner_now"] = [now(r) for r in U.itertuples()]; U["switch"] = ((U.winner_now != U.winner) & (U.family == "reversed")).astype(float)
    def crole(r):   # Block C: the diagonal pushes toward the losing or the winning package
        if r.family != "reversed" or r.vec not in ("Dp", "Dm"): return "na"
        toward = "EL-ST" if r.vec == "Dp" else "ER-SP"
        return "toward_loser" if toward != r.winner_pkg else "toward_winner"
    U["crole"] = [crole(r) for r in U.itertuples()]
    U["dH"] = U.final_H - U.H0; U["dflips"] = U.flips - U.flips0; U["marg"] = (U.pA_n - 0.5).abs(); U["marg0"] = (U.pA0 - 0.5).abs()
    return D, U

D, U = units(); lc.atomic_csv(U, T / "units.csv")
R = {"model": a.model, "n_prompts": int(len(D)), "mean_mass": float(U.mass.mean()), "draws": a.draws, "pushes": json.load(open(base / "main" / "pushes.json", encoding="utf-8"))}
PL = U[U.party.isin(PARTIES) & (U.family == "party_like")]; RV = U[U.party.isin(PARTIES) & (U.family == "reversed")]
REV = RV[RV.decided]; SG = U[U.party.isin(PARTIES) & (U.family == "single")]

# ---- Block D: direction asymmetry (flips, party-like, norm H) ----
def blockD(S, y, s, seed):
    out = {"P1_Dp_minus_placebo": contrast(S, y, [f"Dp_{s}"], PLAC[s], a.draws, seed + 1), "P2_Dp_minus_Dm": contrast(S, y, [f"Dp_{s}"], [f"Dm_{s}"], a.draws, seed + 2),
           "P3_Sm_minus_placebo": contrast(S, y, [f"Sm_{s}"], PLAC[s], a.draws, seed + 3), "P4_Dp_minus_Sm": contrast(S, y, [f"Dp_{s}"], [f"Sm_{s}"], a.draws, seed + 4),
           "Ep_minus_placebo": contrast(S, y, [f"Ep_{s}"], PLAC[s], a.draws, seed + 5), "Em_minus_placebo": contrast(S, y, [f"Em_{s}"], PLAC[s], a.draws, seed + 6),
           "Sp_minus_placebo": contrast(S, y, [f"Sp_{s}"], PLAC[s], a.draws, seed + 7), "Dm_minus_placebo": contrast(S, y, [f"Dm_{s}"], PLAC[s], a.draws, seed + 8),
           "Xp_minus_placebo": contrast(S, y, [f"Xp_{s}"], PLAC[s], a.draws, seed + 9), "Xm_minus_placebo": contrast(S, y, [f"Xm_{s}"], PLAC[s], a.draws, seed + 10),
           "placebo_minus_NONE": contrast(S, y, PLAC[s], ["NONE"], a.draws, seed + 11)}
    m = S.groupby("cond")[y].mean(); pl = m.reindex(PLAC[s]).dropna()
    out["means_minus_NONE"] = (m - m.get("NONE", np.nan)).round(3).to_dict()
    out["placebo_spread"] = {"min": float(pl.min()) if len(pl) else None, "max": float(pl.max()) if len(pl) else None, "sd": float(pl.std()) if len(pl) > 1 else None}
    for d in lc.DIRS:   # rank of each direction among the placebo seeds (direction-level means), exact p = (1 + #placebos >= dir) / (n_plac + 1)
        if f"{d}_{s}" in m and len(pl):
            v = float(m[f"{d}_{s}"]); out[f"rank|{d}"] = {"mean": round(v, 3), "n_plac_ge": int((pl >= v).sum()), "p_rank": float((1 + (pl >= v).sum()) / (len(pl) + 1))}
    return out
Dh = blockD(PL, "flips", "H", lc.seed_for("L2", a.model, "DH"))
for n in ("P1_Dp_minus_placebo", "P2_Dp_minus_Dm", "P4_Dp_minus_Sm"):
    if Dh[n] is not None: Dh[n]["rank"] = Dh.get("rank|Dp", {}).get("n_plac_ge"); Dh[n]["p_rank"] = Dh.get("rank|Dp", {}).get("p_rank")
apply_holm(Dh, ["P1_Dp_minus_placebo", "P2_Dp_minus_Dm", "P4_Dp_minus_Sm"])
for n in ("P1_Dp_minus_placebo", "P2_Dp_minus_Dm", "P4_Dp_minus_Sm"):
    if Dh[n] is not None: Dh[n]["rule"] = ("ge", 0.5); Dh[n]["supported"] = rule_ge(Dh[n], 0.5) and (n != "P1_Dp_minus_placebo" or Dh[n]["rank"] == 0)
R["blockD_flips_party_like_H"] = {"family": "party-like items, both parties, lens flips, norm H; P1 also requires D+ above every placebo seed; Holm over P1, P2, P4", "tests": Dh}
R["blockD_secondary"] = {"flips|L": blockD(PL, "flips", "L", lc.seed_for("L2", a.model, "DL")), "lens_H|H": blockD(PL, "lens_H", "H", lc.seed_for("L2", a.model, "lens")),
                         "DLf|H": blockD(PL, "DLf", "H", lc.seed_for("L2", a.model, "DLf")), "flips_reversed|H": blockD(RV, "flips", "H", lc.seed_for("L2", a.model, "rev")),
                         "flips_single|H": blockD(SG, "flips", "H", lc.seed_for("L2", a.model, "sg"))}
for p in PARTIES: R["blockD_secondary"][f"flips|H|{p}"] = blockD(PL[PL.party == p], "flips", "H", lc.seed_for("L2", a.model, "DH", p))

# ---- Block B: natural cross-pressure at NONE ----
N0 = U[U.party.isin(PARTIES) & (U.cond == "NONE") & U.family.isin(["party_like", "reversed"])].copy(); N0["g2"] = N0.pid + "|" + N0.topic
Bf = {}
for p, nm in (("Democrat", "B1_Democrat"), ("Republican", "B2_Republican")):
    Bf[nm] = contrast(N0[N0.party == p], "final_H", ["reversed"], ["party_like"], a.draws, lc.seed_for("L2", a.model, nm), col="family", grp="g2")
s = N0.copy(); s["rev"] = (s.family == "reversed").astype(float); s["revR"] = s.rev * (s.party == "Republican")
# A4 (2026-10-16): persona x topic fixed effects (g2) absorb the party main effect and topic, so revR is the Republican - Democrat difference in
# the reversed - party-like effect; clusters = persona. The sealed version used topic means and topic clusters without a party main effect.
r = wcb(s, "final_H", ["rev", "revR"], 1, a.draws, lc.seed_for("L2", a.model, "B3"), grp="g2")
Bf["B3_Republican_minus_Democrat_conflict_effect"] = r
apply_holm(Bf, ["B1_Democrat", "B2_Republican"])
for nm in ("B1_Democrat", "B2_Republican"):
    if Bf[nm] is not None: Bf[nm]["rule"] = ("ge", 0.05); Bf[nm]["supported"] = rule_ge(Bf[nm], 0.05)
if r is not None: r["rule"] = ("two-sided; Democrat-only conflict effect predicted (negative); not in the Holm family (B1, B2); report p_wcb"); r["direct_B2_minus_B1"] = (Bf["B2_Republican"]["est"] - Bf["B1_Democrat"]["est"]) if Bf["B1_Democrat"] and Bf["B2_Republican"] else None
Bf["entropy_at_NONE_by_party_family"] = N0.groupby(["party", "family"]).final_H.mean().round(3).to_dict().__repr__()
Bf["own_package_at_NONE_by_party"] = N0[N0.family == "party_like"].groupby("party").own_p.mean().round(3).to_dict()
dec = REV[REV.cond == "NONE"]
Bf["economic_dominance_share_by_party"] = dec.groupby("party").winner.apply(lambda x: float((x == "econ").mean())).round(3).to_dict(); Bf["decided_units_by_party"] = dec.groupby("party").size().to_dict()
Bf["model_arm_entropy_by_family"] = U[(U.party == "none") & (U.cond == "NONE")].groupby("family").final_H.mean().round(3).to_dict()
R["blockB_natural_cross_pressure"] = {"family": "NONE condition; reversed - party-like final entropy within persona x topic; CR1 by persona; Holm over B1, B2", "tests": Bf}

# ---- Block C: displacement with the pure diagonals (reversed DECIDED, norm H) ----
Cf = {}
for s_ in ("H", "L"):
    S = REV[(REV.strength == s_) & REV.vec.isin(["Dp", "Dm"])]
    Cf[f"C1_entropy_toward_loser_minus_toward_winner|{s_}"] = contrast(S, "final_H", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C1", s_), col="crole")
    Cf[f"C2_switch_toward_loser_minus_toward_winner|{s_}"] = contrast(S, "switch", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C2", s_), col="crole")
    Cf[f"C3_entropy_toward_winner_minus_placebo|{s_}"] = contrast(REV[(REV.strength == s_)], "final_H", ["toward_winner"], ["plac"], a.draws, lc.seed_for("L2", a.model, "C3", s_), col="crole_or_plac") if False else None
apply_holm(Cf, ["C1_entropy_toward_loser_minus_toward_winner|H", "C2_switch_toward_loser_minus_toward_winner|H"])
if Cf["C1_entropy_toward_loser_minus_toward_winner|H"] is not None: Cf["C1_entropy_toward_loser_minus_toward_winner|H"]["rule"] = ("ge", 0.05); Cf["C1_entropy_toward_loser_minus_toward_winner|H"]["supported"] = rule_ge(Cf["C1_entropy_toward_loser_minus_toward_winner|H"], 0.05)
if Cf["C2_switch_toward_loser_minus_toward_winner|H"] is not None: Cf["C2_switch_toward_loser_minus_toward_winner|H"]["rule"] = ("ge", 0.30); Cf["C2_switch_toward_loser_minus_toward_winner|H"]["supported"] = rule_ge(Cf["C2_switch_toward_loser_minus_toward_winner|H"], 0.30)
Cf["dH_by_party_vec|H"] = REV[(REV.strength == "H")].groupby(["party", "vec"]).dH.mean().round(3).to_dict().__repr__()
Cf["switch_by_party_vec|H"] = REV[(REV.strength == "H")].groupby(["party", "vec"]).switch.mean().round(3).to_dict().__repr__()
Cf["dH_toward_winner_vs_placebo|H"] = {"toward_winner": float(REV[(REV.strength == "H") & (REV.crole == "toward_winner")].dH.mean()) if (REV.crole == "toward_winner").any() else None,
                                       "placebo": float(REV[(REV.strength == "H") & (REV.kind == "plac")].dH.mean()) if ((REV.strength == "H") & (REV.kind == "plac")).any() else None}
# A5 (2026-10-16; exploratory sensitivity, second Codex check): decided reversed units whose two presentation orders agree at NONE
# (|P_A(order 0) - P_A(order 1)| <= 0.5, package probabilities). units.csv and every registered test are unchanged.
_n0 = D[(D.cond == "NONE") & (D.family == "reversed") & D.party.isin(PARTIES)].assign(g=lambda x: x.pid + "|" + x.topic + "|" + x.family)
_od = _n0.pivot_table(index="g", columns="order", values="pA_n").pipe(lambda m: (m[0] - m[1]).abs())
REVoc = REV[REV.g.map(_od) <= 0.5]; dec_oc = REVoc[REVoc.cond == "NONE"]
OC = {"n_decided_units_by_party": dec_oc.groupby("party").size().to_dict(), "n_decided_units_all_by_party": REV[REV.cond == "NONE"].groupby("party").size().to_dict(),
      "economic_dominance_share_by_party": dec_oc.groupby("party").winner.apply(lambda x: float((x == "econ").mean())).round(3).to_dict()}
_S = REVoc[(REVoc.strength == "H") & REVoc.vec.isin(["Dp", "Dm"])]
OC["C1_entropy_toward_loser_minus_toward_winner|H"] = short(contrast(_S, "final_H", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C1oc"), col="crole"))
OC["C2_switch_toward_loser_minus_toward_winner|H"] = short(contrast(_S, "switch", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C2oc"), col="crole"))
OC["label_A6"] = "order-discrepancy <= 0.5 sensitivity (does not require the same preferred package; A6 renaming)"
R["sensitivity_order_consistent_A5"] = OC
# A6 (2026-10-16; third Codex check): strict same-side sensitivity: both order probabilities strictly on the same side of 0.5 (exact 0.5 excluded)
_pv = _n0.pivot_table(index="g", columns="order", values="pA_n")
_ss = ((_pv[0] > 0.5) & (_pv[1] > 0.5)) | ((_pv[0] < 0.5) & (_pv[1] < 0.5))
REVss = REV[REV.g.map(_ss).fillna(False).astype(bool)]; dec_ss = REVss[REVss.cond == "NONE"]
SS = {"label": "strict same-side sensitivity (A6, exploratory)", "n_decided_units_by_party": dec_ss.groupby("party").size().to_dict(),
      "economic_dominance_share_by_party": dec_ss.groupby("party").winner.apply(lambda x: float((x == "econ").mean())).round(3).to_dict()}
_S2 = REVss[(REVss.strength == "H") & REVss.vec.isin(["Dp", "Dm"])]
SS["C1_entropy_toward_loser_minus_toward_winner|H"] = short(contrast(_S2, "final_H", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C1ss"), col="crole"))
SS["C2_switch_toward_loser_minus_toward_winner|H"] = short(contrast(_S2, "switch", ["toward_loser"], ["toward_winner"], a.draws, lc.seed_for("L2", a.model, "C2ss"), col="crole"))
R["sensitivity_same_side_A6"] = SS
R["blockC_displacement"] = {"family": "reversed DECIDED items; D+ pushes toward EL-ST, D- toward ER-SP; toward the losing vs the winning package; Holm over C1, C2 at norm H", "tests": Cf}

# ---- Block S: single-issue specificity matrix ----
SG2 = SG.assign(axis=np.where(SG.kang.isna() | (SG.kang == ""), "econ", "soc"))
m0 = SG2[SG2.cond == "NONE"].groupby(["party", "axis"]).pA_n.mean()
spec = SG2[SG2.cond != "NONE"].groupby(["party", "axis", "cond"]).pA_n.mean(); spec = (spec - spec.index.droplevel("cond").map(m0).values).unstack("cond")
lc.atomic_csv(spec.round(3).reset_index(), T / "single_issue_specificity.csv")   # A4: ratios below use the unrounded shifts; rounding for display only
def ratio(s_):
    out = {}
    for p in PARTIES:
        try:
            on_e = abs(spec.loc[(p, "econ"), f"Ep_{s_}"] - spec.loc[(p, "econ"), f"Em_{s_}"]); off_e = abs(spec.loc[(p, "soc"), f"Ep_{s_}"] - spec.loc[(p, "soc"), f"Em_{s_}"])
            on_s = abs(spec.loc[(p, "soc"), f"Sp_{s_}"] - spec.loc[(p, "soc"), f"Sm_{s_}"]); off_s = abs(spec.loc[(p, "econ"), f"Sp_{s_}"] - spec.loc[(p, "econ"), f"Sm_{s_}"])
            out[p] = {"econ_on": round(float(on_e), 3), "econ_off": round(float(off_e), 3), "econ_specificity": round(float(on_e / (on_e + off_e)), 3) if on_e + off_e > 0 else None,
                      "soc_on": round(float(on_s), 3), "soc_off": round(float(off_s), 3), "soc_specificity": round(float(on_s / (on_s + off_s)), 3) if on_s + off_s > 0 else None}
        except KeyError: out[p] = None
    return out
R["blockS_specificity"] = {"matrix(+ = toward left / progressive)": spec.round(3).to_dict().__repr__(), "ratios|H": ratio("H"), "ratios|L": ratio("L")}
# ---- descriptives ----
tabs = {"party_like_flips": PL.groupby(["party", "cond"]).flips.mean().unstack("cond").round(3), "party_like_own_p": PL.groupby(["party", "cond"]).own_p.mean().unstack("cond").round(3),
        "reversed_decided_dH": REV[REV.kind != "none"].groupby(["party", "cond"]).dH.mean().unstack("cond").round(3), "reversed_decided_switch": REV[REV.kind != "none"].groupby(["party", "cond"]).switch.mean().unstack("cond").round(3)}
for k, t in tabs.items(): lc.atomic_csv(t.reset_index(), T / f"{k}.csv")
R["descriptive"] = {"no_persona_arm|party_like_flips_minus_NONE": (lambda m: (m - m.get("NONE", np.nan)).round(3).to_dict())(U[(U.party == "none") & (U.family == "party_like")].groupby("cond").flips.mean()),
                    "party_like_mean_margin_by_cond": PL.groupby("cond").marg.mean().round(3).to_dict(), "decided_units": int(dec.shape[0]), "winner_counts": dec.groupby(["party", "winner"]).size().to_dict().__repr__()}
mf = base / "main" / "mapcheck.csv"
if mf.exists():
    M = pd.read_csv(mf); K_ = ["pid", "party", "cond"]   # A5 (2026-10-16): axis-specific questions, as Studies D, F, G (was: both axes averaged)
    Q = pd.concat([M[M.qaxis == "economic"].groupby(K_).proj_e.mean(), M[M.qaxis == "social"].groupby(K_).proj_s.mean()], axis=1).reset_index()
    n0 = Q[Q.cond == "NONE"].set_index("pid"); d_, r_ = n0[n0.party == "Democrat"], n0[n0.party == "Republican"]
    gap_e, gap_s = d_.proj_e.mean() - r_.proj_e.mean(), d_.proj_s.mean() - r_.proj_s.mean()
    Q["de_gap"] = (Q.proj_e - Q.pid.map(n0.proj_e)) / gap_e; Q["ds_gap"] = (Q.proj_s - Q.pid.map(n0.proj_s)) / gap_s
    R["map_check"] = {"party_gap_e": float(gap_e), "party_gap_s": float(gap_s), "move_in_party_gap_units(+ = left / progressive)": Q[Q.cond != "NONE"].groupby("cond")[["de_gap", "ds_gap"]].mean().round(2).to_dict(orient="index").__repr__()}
lc.atomic_json(R, base / "RESULTS.json")
for n in ("P1_Dp_minus_placebo", "P2_Dp_minus_Dm", "P3_Sm_minus_placebo", "P4_Dp_minus_Sm", "Em_minus_placebo"): lc.log("D", n, short(Dh[n]))
lc.log("D placebo spread", Dh["placebo_spread"], "| rank Dp", Dh.get("rank|Dp"))
for n in ("B1_Democrat", "B2_Republican", "B3_Republican_minus_Democrat_conflict_effect"): lc.log("B", n, short(Bf[n]))
lc.log("B dominance", Bf["economic_dominance_share_by_party"], "own", Bf["own_package_at_NONE_by_party"])
for n in ("C1_entropy_toward_loser_minus_toward_winner|H", "C2_switch_toward_loser_minus_toward_winner|H"): lc.log("C", n, short(Cf[n]))
lc.log("S ratios H", R["blockS_specificity"]["ratios|H"])

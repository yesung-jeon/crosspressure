#!/usr/bin/env python
"""Synthetic check of l2_analyze.py (standing rule). Fake parts under STUDYL_OUT/<model>: additive logit for choices (party-like items
near-certain, reversed items decided by a topic-specific economic weight), flips = noise + 2.0 for Dp_H + 0.8 for Sm_H - 1.0 for Em_H,
Democrat reversed-item entropy raised by 0.2 (Block B built in). Runs l2 with few draws and compares P1, P2, P4, B1, B2, C1, C2 with
direct within-group means. Usage: STUDYL_OUT=<scratch> python synth_check.py --model qwen"""
import argparse, json, os, subprocess, sys
import numpy as np, pandas as pd
import l_common as lc

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen"); a = ap.parse_args()
assert os.environ.get("STUDYL_OUT"), "set STUDYL_OUT to a scratch directory"
od = lc.out_dir(a.model); main = lc.out_dir(a.model, "main"); rng = np.random.default_rng(3)
P = lc.personas("confirm+model"); conds = lc.conditions(); bundles, singles = lc.items(); ITEMS = bundles + singles
be = {b["topic"]: rng.uniform(1, 6) for b in ITEMS}; bs = {b["topic"]: rng.uniform(1, 6) for b in ITEMS}
VEC = {"Ep": (1, 0), "Em": (-1, 0), "Sp": (0, 1), "Sm": (0, -1), "Dp": (0.7, -0.7), "Dm": (-0.7, 0.7), "Xp": (0.7, 0.7), "Xm": (-0.7, -0.7), "PLAC": (0, 0), "NONE": (0, 0)}
FL = {"Dp_H": 2.0, "Sm_H": 0.8, "Em_H": -1.0}
rows = []
for p in P:
    for c in conds:
        sc = {"H": 0.6, "L": 0.42, "none": 0.0}[c["strength"]]; de, ds = VEC[c["vec"]]
        for b in ITEMS:
            eo = p["own"] + sc * de; so = p["own"] + sc * ds
            if b["family"] == "single": logit = be[b["topic"]] * eo if b["axis"] == "econ" else bs[b["topic"]] * so
            elif b["family"] == "party_like": logit = (be[b["topic"]] + bs[b["topic"]]) * (eo + so) / 2 * (2.0 if p["party"] == "Democrat" else 0.15)   # A4: party baselines differ
            else: logit = be[b["topic"]] * eo - bs[b["topic"]] * so + (rng.normal(0, 0.8) if p["party"] == "Democrat" else 0)
            logit += rng.normal(0, 0.2) + (rng.normal(0, 0.3) if c["kind"] == "plac" else 0); pA = 1 / (1 + np.exp(-logit))
            fl = rng.integers(0, 4) + FL.get(c["name"], 0.0)
            for order in (0, 1):
                pa, pb = (pA, 1 - pA) if order == 0 else (1 - pA, pA)
                rows.append(dict(pid=p["pid"], party=p["party"], cond=c["name"], kind=c["kind"], strength=c["strength"], vec=c["vec"], topic=b["topic"], family=b["family"], kang=b["kang"], origin=b["origin"],
                                 order=order, DL=rng.integers(15, 28), flips=fl, lens_H=rng.uniform(0.3, 0.6), pA=pa * 0.999, pB=pb * 0.999, m_final=logit if order == 0 else -logit))
D = pd.DataFrame(rows)
for i, ch in enumerate(np.array_split(D.pid.unique(), 3)): lc.atomic_csv(D[D.pid.isin(ch)], main / f"part_{i:02d}.csv.gz")
lc.atomic_json(dict(mu=50.0, norm_H=14.0, norm_L=10.0, n_layers=28), main / "pushes.json")
lc.atomic_csv(pd.DataFrame([dict(pid=p["pid"], party=p["party"], cond=c["name"], kind=c["kind"], strength=c["strength"], vec=c["vec"], q=j, qaxis="economic" if j < 5 else "social",
                                 proj_e=p["own"] * 5 + 5 * VEC[c["vec"]][0] + rng.normal(), proj_s=-p["own"] * 4 + 4 * VEC[c["vec"]][1] + rng.normal())
                            for p in P for c in conds if not (c["kind"] == "plac" and c["plac"] > 1) for j in range(10)]), main / "mapcheck.csv")
r = subprocess.run([sys.executable, str(lc.CODE / "l2_analyze.py"), "--model", a.model, "--draws", "99"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=os.environ)
print(r.stdout[-3000:], r.stderr[-3000:]); assert r.returncode == 0, "l2_analyze failed on synthetic data"   # A5: fail, not just print
R = json.load(open(od / "RESULTS.json", encoding="utf-8")); U = pd.read_csv(lc.out_dir(a.model, "tables") / "units.csv")
PL = U[U.party.isin(["Democrat", "Republican"]) & (U.family == "party_like")]; REV = U[U.party.isin(["Democrat", "Republican"]) & (U.family == "reversed") & U.decided]
N0 = U[U.party.isin(["Democrat", "Republican"]) & (U.cond == "NONE") & U.family.isin(["party_like", "reversed"])].copy(); N0["g2"] = N0.pid + "|" + N0.topic
def direct(S, hi, lo, y="flips", col="cond", grp="g"):
    s = S[S[col].isin(hi + lo)]; m = s.groupby([grp, s[col].isin(hi)])[y].mean().unstack().dropna(); return float((m[True] - m[False]).mean())
Dh = R["blockD_flips_party_like_H"]["tests"]; Bf = R["blockB_natural_cross_pressure"]["tests"]; Cf = R["blockC_displacement"]["tests"]; PLAC = [f"PLAC_H_{i}" for i in range(10)]
chk = {"P1": (Dh["P1_Dp_minus_placebo"]["est"], direct(PL, ["Dp_H"], PLAC)), "P2": (Dh["P2_Dp_minus_Dm"]["est"], direct(PL, ["Dp_H"], ["Dm_H"])), "P4": (Dh["P4_Dp_minus_Sm"]["est"], direct(PL, ["Dp_H"], ["Sm_H"])),
       "B1": (Bf["B1_Democrat"]["est"], direct(N0[N0.party == "Democrat"], ["reversed"], ["party_like"], "final_H", "family", "g2")),
       "B2": (Bf["B2_Republican"]["est"], direct(N0[N0.party == "Republican"], ["reversed"], ["party_like"], "final_H", "family", "g2")),
       "C1": (Cf["C1_entropy_toward_loser_minus_toward_winner|H"]["est"], direct(REV[(REV.strength == "H") & REV.vec.isin(["Dp", "Dm"])], ["toward_loser"], ["toward_winner"], "final_H", "crole")),
       "C2": (Cf["C2_switch_toward_loser_minus_toward_winner|H"]["est"], direct(REV[(REV.strength == "H") & REV.vec.isin(["Dp", "Dm"])], ["toward_loser"], ["toward_winner"], "switch", "crole"))}
chk["B3"] = (Bf["B3_Republican_minus_Democrat_conflict_effect"]["est"], chk["B2"][1] - chk["B1"][1])   # A4: difference of within-unit differences
for n, (est, d) in chk.items(): print(f"{n}: wcb {est:+.4f} direct {d:+.4f} {'OK' if abs(est - d) < 1e-6 else 'MISMATCH'}")
old_b3 = float(N0[N0.family == "reversed"].groupby("party").final_H.mean().pipe(lambda m: m["Republican"] - m["Democrat"]))
print(f"B3 old (sealed) specification would give about {old_b3:+.4f} (party gap on reversed items); must differ from the direct value here")
print("P1 about +2 and rank 0:", round(Dh["P1_Dp_minus_placebo"]["est"], 3), Dh["rank|Dp"], "| P4 about +1.2:", round(Dh["P4_Dp_minus_Sm"]["est"], 3), "| B1 > B2:", round(Bf["B1_Democrat"]["est"], 3), round(Bf["B2_Republican"]["est"], 3))
print("keys:", list(R.keys()), "| map", "map_check" in R, "| decided", R["descriptive"]["decided_units"], "| spec", R["blockS_specificity"]["ratios|H"])
# A5 (2026-10-16, second Codex check): the check fails instead of printing MISMATCH
bad = [n for n, (est, d) in chk.items() if abs(est - d) >= 1e-6]; assert not bad, f"MISMATCH: {bad}"
assert abs(chk["B3"][0] - old_b3) > 0.05, "synthetic data cannot tell the corrected B3 from the sealed specification"
assert Dh["rank|Dp"]["n_plac_ge"] == 0 and Dh["P1_Dp_minus_placebo"]["est"] > 1.5, "built-in D+ effect not recovered"
assert "sensitivity_order_consistent_A5" in R, "A5 sensitivity block missing"
print("SYNTHETIC CHECK PASSED")

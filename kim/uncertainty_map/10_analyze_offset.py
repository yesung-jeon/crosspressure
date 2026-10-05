#!/usr/bin/env python
"""Step 10 - decompose the persona offset and re-centre on a neutral human (CPU).

1. Offset by condition (x, y, entropy) with persona-level SEs; contrasts C2-C0 (drop the Qwen identity), C3-C2 (become a
   bare human), C4-C3 (named non-political person), C5-C4 (Kim's "actual political positions" ending), C7-C4 (1955).
2. Human origin = mean position of C5 (named non-political 2023 persons with Kim's ending, the same sentence frame as Kim's
   member personas). Sensitivity: C4.
3. T1 re-run around the human origin with step 7 data: piecewise slopes y_neg / y_pos now testable if personas fall on both sides.
4. Kim's members relative to the human origin: share socially progressive of a neutral 2023 person, by party and era.
Output: out/offset_summary.json, out/OFFSET_RESULTS.md, out/fig_offset.png
Usage: python 10_analyze_offset.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}
P = pd.read_csv(OUT / "offset_personas.csv"); G = pd.read_csv(OUT / "offset_generations.csv")
pm = G.groupby("pid").entropy64.mean().rename("entropy").reset_index().merge(P, on="pid", validate="1:1")
ORDER = ["C0_origin", "C1_default_text", "C2_assistant", "C3_bare_human", "C4_nonpolitical", "C5_nonpolitical_kim_ending", "C6_name_only", "C7_nonpolitical_1955"]
def ms(v): return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, n=int(len(v)))
R["by_condition"] = {c: {k: ms(pm[pm.cond == c][k]) for k in ("x", "y", "entropy")} for c in ORDER}
def contrast(a, b, k):
    va, vb = pm[pm.cond == a][k], pm[pm.cond == b][k]; d = va.mean() - vb.mean()
    se = np.sqrt((va.var(ddof=1) / len(va) if len(va) > 1 else 0) + (vb.var(ddof=1) / len(vb) if len(vb) > 1 else 0))
    return dict(diff=float(d), se=float(se) if se > 0 else None)
R["contrasts"] = {f"{a} - {b}": {k: contrast(a, b, k) for k in ("x", "y", "entropy")} for a, b in
                  [("C1_default_text", "C0_origin"), ("C2_assistant", "C0_origin"), ("C3_bare_human", "C2_assistant"),
                   ("C4_nonpolitical", "C3_bare_human"), ("C5_nonpolitical_kim_ending", "C4_nonpolitical"),
                   ("C6_name_only", "C3_bare_human"), ("C7_nonpolitical_1955", "C4_nonpolitical")]}
# paired version for the 30 matched personas (same name, occupation, state)
pm["k"] = pm.pid.str[3:]
def paired(a, b, k):
    w = pm[pm.cond.isin([a, b])].pivot_table(index="k", columns="cond", values=k).dropna(); d = w[a] - w[b]
    return dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n=int(len(d)))
R["paired"] = {f"{a} - {b}": {k: paired(a, b, k) for k in ("x", "y", "entropy")} for a, b in
               [("C5_nonpolitical_kim_ending", "C4_nonpolitical"), ("C7_nonpolitical_1955", "C4_nonpolitical"), ("C4_nonpolitical", "C6_name_only")]}

# ---- human origin ----
ref = {c: pm[pm.cond == c][["x", "y"]].mean() for c in ("C5_nonpolitical_kim_ending", "C4_nonpolitical")}
R["human_origin"] = {c: dict(x=float(v.x), y=float(v.y)) for c, v in ref.items()}

# ---- T1 around the human origin (step 7 data) ----
T = pd.read_csv(OUT / "t1_personas.csv"); TG = pd.read_csv(OUT / "t1_generations.csv").merge(T.drop(columns=["system", "set"]), on="pid", validate="m:1")
TG["cell"] = np.where(TG.set == "S", TG.econ_level.astype(str) + "_" + TG.soc_level.astype(str), TG.pid)
R["T1_recentred"] = {}
for c, v in ref.items():
    d = TG[TG.set.isin(["S", "P"])].copy(); d["xh"], d["yh"] = d.x - v.x, d.y - v.y
    for a in ("xh", "yh"):
        d[f"{a}_neg"], d[f"{a}_pos"] = d[a].clip(upper=0), d[a].clip(lower=0)
    tp = d.drop_duplicates("pid")
    out = dict(personas_progressive_of_human=int((tp.yh < 0).sum()), personas_left_of_human=int((tp.xh < 0).sum()), personas=int(len(tp)))
    if (tp.yh < 0).sum() >= 5:
        r = cm.fit_cr1(d, "entropy64 ~ xh_neg + xh_pos + yh_neg + yh_pos + C(q)", "cell")
        out.update({k: cm.coef(r, k) for k in ("xh_neg", "xh_pos", "yh_neg", "yh_pos")})
        rs = cm.fit_cr1(d[d.set == "S"], "entropy64 ~ xh_neg + xh_pos + yh_neg + yh_pos + C(q)", "cell")
        out["stance_only"] = {k: cm.coef(rs, k) for k in ("yh_neg", "yh_pos")}
    R["T1_recentred"][c] = out

# ---- Kim's members relative to the human origin ----
m = cm.members(); m = m[m.party.isin(["Democrat", "Republican"])]
yref = float(ref["C5_nonpolitical_kim_ending"].y)
R["members_progressive_of_human"] = {f"{e}|{p}": round(float((g.y < yref).mean()), 3) for (e, p), g in m.groupby(["era", "party"])}
R["members_progressive_of_human_overall"] = float((m.y < yref).mean())
R["note_precision"] = "steps 7 and 9 projections are int8 in the same pipeline; Kim's member coordinates are bf16 (int8 vs bf16 level offset up to 0.19 on x, 0.06 on y; 2_runs/F2)"
(OUT / "offset_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
lab = [c.split("_", 1)[1].replace("_", " ") for c in ORDER]; ys = [R["by_condition"][c]["y"]["mean"] for c in ORDER]
es = [R["by_condition"][c]["y"]["se"] or 0 for c in ORDER]
ax[0].barh(lab, ys, xerr=es, color=["#888"] * 3 + ["#6a9fd8"] * 5); ax[0].axvline(0, color="k", lw=.8)
for yv, nm, col in [(m[(m.party == "Democrat") & (m.congress == 118)].y.mean(), "Democrats 2023", "tab:blue"),
                    (m[(m.party == "Republican") & (m.congress == 118)].y.mean(), "Republicans 2023", "tab:red")]:
    ax[0].axvline(yv, color=col, ls="--", lw=1); ax[0].text(yv, len(ORDER) - .4, nm, color=col, fontsize=8, rotation=90, va="top")
ax[0].set_xlabel("social position y (Kim's projection; 0 = Qwen default system prompt)"); ax[0].set_title("Where does a persona move the social axis?", fontsize=10)
pt = TG.drop_duplicates("pid").merge(TG.groupby("pid").entropy64.mean().rename("ent").reset_index(), on="pid")
for st, mk in (("S", "o"), ("P", "^")):
    q = pt[pt.set == st]; ax[1].scatter(q.y - yref, q.ent, marker=mk, s=18, alpha=.7, label={"S": "stance personas", "P": "public figures"}[st])
ax[1].axvline(0, color="k", ls=":", lw=.8); ax[1].axvline(-yref, color="gray", ls="--", lw=.8); ax[1].text(-yref, ax[1].get_ylim()[1], " Qwen default", fontsize=8, va="top")
ax[1].set_xlabel("social position relative to a neutral 2023 person (C5)"); ax[1].set_ylabel("entropy"); ax[1].legend(fontsize=8)
ax[1].set_title("T1 re-centred on a neutral human", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_offset.png", dpi=110)

f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
L = ["# Persona offset results (generated by 10_analyze_offset.py; exploratory)", ""]
for c in ORDER:
    b = R["by_condition"][c]; L.append(f"- {c}: y {b['y']['mean']:+.3f}, x {b['x']['mean']:+.3f}, entropy {b['entropy']['mean']:.3f} (n = {b['y']['n']})")
L += ["", "Paired (same 30 persons): " + "; ".join(f"{k}: y {v['y']['diff']:+.3f} (se {v['y']['se']:.3f}), entropy {v['entropy']['diff']:+.3f}" for k, v in R["paired"].items()),
      f"Human origin (C5): x {R['human_origin']['C5_nonpolitical_kim_ending']['x']:+.3f}, y {yref:+.3f}. Members socially progressive of it: {R['members_progressive_of_human_overall']:.1%}; by era|party {R['members_progressive_of_human']}"]
for c, o in R["T1_recentred"].items():
    s = f"T1 re-centred on {c}: personas progressive of the human origin {o['personas_progressive_of_human']}/{o['personas']}"
    if "yh_neg" in o: s += f"; yh_neg {f(o['yh_neg'])}; yh_pos {f(o['yh_pos'])}; stance only yh_neg {f(o['stance_only']['yh_neg'])}"
    L.append(s)
L.append("Symmetric account predicts yh_neg < 0; asymmetric predicts yh_neg > 0.")
(OUT / "OFFSET_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

#!/usr/bin/env python
"""Step 15 - analysis 2: which topics carry the asymmetry? (CPU; existing generations only)

(a) Kim's 10 held-out questions (5 economic: childcare, pensions, parental leave, student debt, environmental regulation;
    5 social: gender-neutral language, marriage, abortion coverage, same-sex adoption, birth rates / family norms).
    Per question: Republican - Democrat entropy gap among legislators (step 1, controls: era, chamber), the caricature slope
    (entropy on excess traditionalism, from 04's y_resid definition), and the public-figure right - left gap (step 7).
(b) The paper's 48 WVS questions in 6 topics, from the steering sweep E7 (2_runs/R1_explore/E7_angles.csv): per topic,
    the amplitude and peak angle of the entropy response at 15 units, and ideology minus random directions.
Output: out/topics_summary.json, out/TOPICS_RESULTS.md, out/fig_topics.png
Usage: python 15_topics.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}
QS = cm.kim_questions(); LABEL = ["childcare", "pensions", "parental leave", "student debt", "environmental regulation",
                                  "gender-neutral language", "marriage definition", "abortion coverage", "same-sex adoption", "birth rates / family"]
G = pd.read_csv(OUT / "member_generations.csv"); G = G[G.party.isin(["Democrat", "Republican"])].copy(); G["rep"] = (G.party == "Republican").astype(int)
M = G.groupby(["icpsr", "congress"]).agg(y=("y", "first"), dim1=("dim1", "first"), dim2=("dim2", "first"), party=("party", "first"),
                                        era=("era", "first"), chamber=("chamber", "first")).reset_index()
M["y_resid"] = cm.fit_cr1(M, "y ~ dim1 + dim2 + C(party) * C(era) + C(chamber)", "icpsr").resid
G = G.merge(M[["icpsr", "congress", "y_resid"]], on=["icpsr", "congress"], validate="m:1")
T = pd.read_csv(OUT / "t1_generations.csv").merge(pd.read_csv(OUT / "t1_personas.csv")[["pid", "side"]], on="pid", validate="m:1")
T = T[T.set == "P"].assign(right=lambda d: (d.side == "right").astype(int))
rows = []
for q in range(10):
    g = G[G.q == q]; t = T[T.q == q]
    a = cm.coef(cm.fit_cr1(g, "entropy64 ~ rep + C(era) + C(chamber)", "icpsr"), "rep")
    b = cm.coef(cm.fit_cr1(g, "entropy64 ~ y_resid + dim1 + dim2 + C(party) * C(era) + C(chamber)", "icpsr"), "y_resid")
    c = cm.coef(cm.fit_cr1(t, "entropy64 ~ right", "pid"), "right")
    rows.append(dict(q=q, axis=QS[q]["axis"], topic=LABEL[q], rep_minus_dem=a["est"], rep_p=a["p"], caricature_slope=b["est"], caricature_p=b["p"],
                     figures_right_minus_left=c["est"], figures_p=c["p"], mean_entropy=float(g.entropy64.mean())))
Q = pd.DataFrame(rows); Q.to_csv(OUT / "topics_kim_questions.csv", index=False)
R["kim_questions"] = Q.round(4).to_dict("records")
R["by_axis_mean"] = Q.groupby("axis")[["rep_minus_dem", "caricature_slope", "figures_right_minus_left"]].mean().round(4).to_dict("index")

E7 = cm.HERE.parent / "2_runs" / "R1_explore" / "E7_angles.csv"
if E7.exists():
    import json as _j
    qb = _j.load(open(cm.HERE.parent / "1_data_question_bank.json", encoding="utf-8"))
    S = pd.read_csv(E7); S["topic"] = S.q.map(lambda i: qb[i]["topic"])
    c0 = S[S.kind == "center"].groupby("q").entropy64.mean(); S["d"] = S.entropy64 - S.q.map(c0)
    out = {}
    for tp, g in S.groupby("topic"):
        c = g[(g.kind == "ideology") & (g.mag == 15.0)].groupby("angle").d.mean().reset_index()
        X = np.c_[np.ones(len(c)), np.cos(np.radians(c.angle)), np.sin(np.radians(c.angle))]; b = np.linalg.lstsq(X, c.d.values, rcond=None)[0]
        rnd = g[(g.kind == "random") & (g.mag == 15.0)].d
        out[tp] = dict(amplitude=float(np.hypot(b[1], b[2])), peak_angle=float(np.degrees(np.arctan2(b[2], b[1])) % 360),
                       entropy_at_ER_ST_225=float(c.loc[c.angle == 225.0, "d"].iloc[0]), entropy_at_EL_SP_45=float(c.loc[c.angle == 45.0, "d"].iloc[0]),
                       random_mean=float(rnd.mean()), random_sd=float(rnd.groupby(g.loc[rnd.index, "cell"]).mean().std()))
    R["wvs_topics_steering"] = out
(OUT / "topics_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
o = Q.sort_values("rep_minus_dem"); col = np.where(o.axis == "social", "tab:purple", "tab:green")
ax[0].barh(o.topic, o.rep_minus_dem, color=col); ax[0].axvline(0, color="k", lw=.7)
ax[0].set_title("Legislator personas: Republican - Democrat entropy, by question (green economic, purple social)", fontsize=9)
o2 = Q.sort_values("figures_right_minus_left"); ax[1].barh(o2.topic, o2.figures_right_minus_left, color=np.where(o2.axis == "social", "tab:purple", "tab:green"))
ax[1].axvline(0, color="k", lw=.7); ax[1].set_title("Public figures: right - left entropy, by question", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_topics.png", dpi=110)
L = ["# Topic results (generated by 15_topics.py; exploratory, 10 + 6 tests per column, unadjusted p)", ""]
for r in rows:
    L.append(f"- q{r['q']} {r['topic']} ({r['axis']}): R-D {r['rep_minus_dem']:+.3f} (p {r['rep_p']:.2g}); caricature {r['caricature_slope']:+.3f} (p {r['caricature_p']:.2g}); figures R-L {r['figures_right_minus_left']:+.3f} (p {r['figures_p']:.2g})")
L.append(f"- Means by axis: {R['by_axis_mean']}")
for tp, o3 in R.get("wvs_topics_steering", {}).items():
    L.append(f"- Steering, {tp}: amplitude {o3['amplitude']:.3f}, peak {o3['peak_angle']:.0f} deg, ER-ST {o3['entropy_at_ER_ST_225']:+.3f}, EL-SP {o3['entropy_at_EL_SP_45']:+.3f}, random mean {o3['random_mean']:+.3f}")
(OUT / "TOPICS_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

#!/usr/bin/env python
"""Step 8 - T1 analysis: is hesitation symmetric around the model's own default, or one-sided? (CPU)

Piecewise regression around the origin (the no-persona model): entropy ~ x_neg + x_pos + y_neg + y_pos, with
x_neg = min(x, 0), x_pos = max(x, 0) and likewise for y.
  symmetric "distance from self": coefficient on y_neg < 0 (entropy rises as personas go further progressive), y_pos > 0
  asymmetric account:             coefficient on y_neg > 0 (same sign as y_pos: entropy keeps falling to the left)
Also: manipulation check (do the stance levels move Kim's projection as described?), distance vs linear model comparison,
design-level means, and the same tests on public figures. CR1 SE clustered by stance cell (S) or persona (P); exploratory.
Output: out/t1_summary.json, out/T1_RESULTS.md, out/fig_T1.png
Usage: python 08_analyze_t1.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out
P = pd.read_csv(OUT / "t1_personas.csv"); G = pd.read_csv(OUT / "t1_generations.csv")
G = G.merge(P.drop(columns=["system", "set"]), on="pid", how="left", validate="m:1")
for v in ("x", "y"):
    G[f"{v}_neg"], G[f"{v}_pos"] = G[v].clip(upper=0), G[v].clip(lower=0)
G["dist"] = np.hypot(G.x, G.y); G["cell"] = G.econ_level.astype(str) + "_" + G.soc_level.astype(str)
S = {}
base = G[G.set == "B"]; S["origin_entropy"] = float(base.entropy64.mean())
pm = G.groupby("pid").agg(entropy=("entropy64", "mean"), hedge=("hedge", "mean"), x=("x", "first"), y=("y", "first"),
                          set=("set", "first"), econ_level=("econ_level", "first"), soc_level=("soc_level", "first"), side=("side", "first"),
                          name=("name", "first")).reset_index()
pm.to_csv(OUT / "t1_persona_means.csv", index=False)

# ---- manipulation check ----
s = pm[pm.set == "S"]
S["manipulation_check"] = {"y_by_soc_level": s.groupby("soc_level").y.mean().round(3).to_dict(),
                           "x_by_econ_level": s.groupby("econ_level").x.mean().round(3).to_dict(),
                           "corr_y_soc_level": float(np.corrcoef(s.y, s.soc_level)[0, 1]), "corr_x_econ_level": float(np.corrcoef(s.x, s.econ_level)[0, 1]),
                           "S_personas_y_below_0": int((s.y < 0).sum()), "S_personas_x_below_0": int((s.x < 0).sum()), "S_personas": int(len(s)),
                           "P_left_mean_xy": pm[(pm.set == "P") & (pm.side == "left")][["x", "y"]].mean().round(3).to_dict(),
                           "P_right_mean_xy": pm[(pm.set == "P") & (pm.side == "right")][["x", "y"]].mean().round(3).to_dict(),
                           "P_left_y_below_0": int(((pm.set == "P") & (pm.side == "left") & (pm.y < 0)).sum())}

# ---- piecewise tests ----
def piece(d, cl, name):
    r = cm.fit_cr1(d, "entropy64 ~ x_neg + x_pos + y_neg + y_pos + C(q)", cl)
    out = {k: cm.coef(r, k) for k in ("x_neg", "x_pos", "y_neg", "y_pos")}
    out["n_personas"] = int(d.pid.nunique())
    return out
Sd, Pd = G[G.set == "S"], G[G.set == "P"]
S["piecewise_S"] = piece(Sd, "cell", "S")
S["piecewise_P"] = piece(Pd, "pid", "P")
S["piecewise_S_plus_P"] = piece(G[G.set.isin(["S", "P"])].assign(cl=lambda d: np.where(d.set == "S", d.cell, d.pid)), "cl", "SP")
for nm, d, cl in (("S", Sd, "cell"), ("P", Pd, "pid")):
    rd = cm.fit_cr1(d, "entropy64 ~ dist + C(q)", cl); rl = cm.fit_cr1(d, "entropy64 ~ x + y + C(q)", cl)
    rb = cm.fit_cr1(d, "entropy64 ~ dist + x + y + C(q)", cl)
    S[f"model_compare_{nm}"] = {"AIC_distance": float(rd.aic), "AIC_linear_xy": float(rl.aic), "AIC_both": float(rb.aic),
                                "both": {k: cm.coef(rb, k) for k in ("dist", "x", "y")}}
S["design_means_S"] = {"entropy_by_soc_level": s.groupby("soc_level").entropy.mean().round(4).to_dict(),
                       "entropy_by_econ_level": s.groupby("econ_level").entropy.mean().round(4).to_dict(),
                       "entropy_grid_econ_x_soc": {f"e{int(k[0])}_s{int(k[1])}": round(float(v), 4) for k, v in s.groupby(["econ_level", "soc_level"]).entropy.mean().items()}}
pp = pm[pm.set == "P"]
S["figures"] = {"left_mean_entropy": float(pp[pp.side == "left"].entropy.mean()), "right_mean_entropy": float(pp[pp.side == "right"].entropy.mean()),
                "right_minus_left": cm.coef(cm.fit_cr1(Pd.assign(right=(Pd.side == "right").astype(int)), "entropy64 ~ right + C(q)", "pid"), "right"),
                "ranked": pp.sort_values("entropy")[["name", "side", "entropy", "x", "y"]].round(3).to_dict("records")}
(OUT / "t1_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")

# ---- figure (checked visually) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for e, c in zip(sorted(s.econ_level.unique()), plt.get_cmap("coolwarm")(np.linspace(0, 1, 5))):
    ss = s[s.econ_level == e]; ax[0].scatter(ss.y, ss.entropy, color=c, s=18, label=f"econ level {int(e)}")
for side, mk, c in (("left", "^", "tab:blue"), ("right", "v", "tab:red")):
    q = pp[pp.side == side]; ax[0].scatter(q.y, q.entropy, marker=mk, color=c, edgecolor="k", s=40, label=f"public figure ({side})")
ax[0].axvline(0, color="k", lw=.8, ls=":"); ax[0].axhline(S["origin_entropy"], color="gray", lw=.8, ls="--")
ax[0].set_xlabel("persona social position y (Kim's projection; 0 = model default, + = traditional)"); ax[0].set_ylabel("entropy (first 64 tokens)")
ax[0].legend(fontsize=7); ax[0].set_title("T1: is hesitation V-shaped around the model's default?", fontsize=10)
grid = s.pivot_table(index="soc_level", columns="econ_level", values="entropy")
im = ax[1].imshow(grid.values, origin="lower", cmap="viridis"); ax[1].set_xticks(range(5), grid.columns.astype(int)); ax[1].set_yticks(range(5), grid.index.astype(int))
ax[1].set_xlabel("economic stance level (-2 far left ... +2 far right)"); ax[1].set_ylabel("social stance level (-2 strongly progressive ... +2 strongly traditional)")
for i in range(5):
    for j in range(5): ax[1].text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
ax[1].set_title("Stance personas: mean entropy by described position", fontsize=10); fig.colorbar(im, ax=ax[1], shrink=.8)
fig.tight_layout(); fig.savefig(OUT / "fig_T1.png", dpi=110)

f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
mc = S["manipulation_check"]; L = ["# T1 results (generated by 08_analyze_t1.py; exploratory, unadjusted p)", "",
     f"Origin (no persona) entropy {S['origin_entropy']:.3f}. Stance personas below the default on y: {mc['S_personas_y_below_0']}/{mc['S_personas']}, on x: {mc['S_personas_x_below_0']}/{mc['S_personas']}. Left figures below 0 on y: {mc['P_left_y_below_0']}/18.",
     f"Manipulation check: corr(y, social level) {mc['corr_y_soc_level']:.2f}; corr(x, economic level) {mc['corr_x_econ_level']:.2f}.", ""]
for nm in ("S", "P", "S_plus_P"):
    pw = S[f"piecewise_{nm}"]; L.append(f"Piecewise ({nm}, {pw['n_personas']} personas): y_neg {f(pw['y_neg'])}; y_pos {f(pw['y_pos'])}; x_neg {f(pw['x_neg'])}; x_pos {f(pw['x_pos'])}")
L += ["", "Symmetric account predicts y_neg < 0; asymmetric predicts y_neg > 0.",
      f"Public figures: right - left entropy {f(S['figures']['right_minus_left'])}."]
for nm in ("S", "P"):
    mcmp = S[f"model_compare_{nm}"]; L.append(f"Model comparison {nm}: AIC distance {mcmp['AIC_distance']:.1f}, linear {mcmp['AIC_linear_xy']:.1f}, both {mcmp['AIC_both']:.1f}; in 'both': dist {f(mcmp['both']['dist'])}, y {f(mcmp['both']['y'])}")
(OUT / "T1_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

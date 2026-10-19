#!/usr/bin/env python
"""Figure 2 (Section 5): stating the profile does not close the gap.
Four panels, one per preregistered outcome (H3 consistency, H4 quadrant reached, H1 choice uncertainty, H2 generation uncertainty):
cross-pressured minus aligned profiles with mode fixed effects, 95% bootstrap interval, three models; dashed line = preregistered minimum
effect; filled = supported under the preregistered rule, open = not supported. All values from the sealed S5_RESULTS.json files.
Usage: python paper/figures/fig2_profiles.py  -> paper/figures/fig2_profiles.pdf and .png (300 dpi)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]; S = R / "sealed" / "STUDY_D_quadrant" / "out"; OUT = Path(__file__).resolve().parent
J = lambda p: json.load(open(p, encoding="utf-8"))
H = {"qwen": J(S / "s5" / "S5_RESULTS.json")["qwen"]["hyp"], "llama": J(S / "s5" / "S5_RESULTS.json")["llama"]["hyp"],
     "olmo": J(S / "s5_olmo" / "S5_RESULTS.json")["olmo"]["hyp"]}
MODELS = [("qwen", "Qwen", "o"), ("llama", "Llama", "s"), ("olmo", "OLMo", "^")]
PANELS = [("H3_C2", "(a) Consistency with the profile", "probability"),
          ("H4_C3hit", "(b) Projection reaches the profile's quadrant", "share of conditions"),
          ("H1_C1a", "(c) Uncertainty of yes/no choices", "bits"),
          ("H2_C4", "(d) Uncertainty of generated text", "bits per token")]
INK = "#8e44ad"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3), gridspec_kw={"wspace": 0.25, "hspace": 0.75}); axes = axes.ravel()
for ax, (h, title, unit) in zip(axes, PANELS):
    vals = [H[k][h] for k, _, _ in MODELS]
    lo = min(min(v["ci95"][0] for v in vals), 0, vals[0]["min_effect"]); hi = max(max(v["ci95"][1] for v in vals), 0, vals[0]["min_effect"])
    pad = 0.15 * (hi - lo); ax.set_xlim(lo - pad, hi + pad)
    ax.axvline(0, color="#555", lw=0.6)
    ax.axvline(vals[0]["min_effect"], color=INK, lw=0.8, ls=(0, (3, 2)))
    for y, ((key, name, mk), v) in enumerate(zip(MODELS[::-1], vals[::-1])):
        ax.plot(v["ci95"], [y, y], color=INK, lw=1.4, solid_capstyle="butt")
        ax.scatter(v["est"], y, marker=mk, s=34, zorder=3, color=INK if v["supported"] else "white", edgecolor=INK, lw=1.1)
    ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n, _ in MODELS[::-1]]); ax.set_ylim(-0.6, 2.6)
    ax.set_title(title, loc="left"); ax.set_xlabel(f"Cross-pressured minus aligned profiles, {unit}")
    ax.grid(axis="x", color="#ececec", lw=0.5); ax.set_axisbelow(True)
for ax in axes: ax.title.set_fontsize(8.5)
h1 = [plt.Line2D([], [], marker="o", ls="", color=INK, markersize=5, label="supported by the preregistered rule"),
      plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=5, label="not supported"),
      plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="preregistered minimum effect"),
      plt.Line2D([], [], color=INK, lw=1.4, label="95% bootstrap interval")]
fig.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7, handletextpad=0.4)
fig.savefig(OUT / "fig2_profiles.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig2_profiles.png", dpi=300, bbox_inches="tight")
print("fig2 written:", {k: {h: round(H[k][h]["est"], 3) for h, _, _ in PANELS} for k, _, _ in MODELS})

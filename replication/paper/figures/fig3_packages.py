#!/usr/bin/env python
"""Figure 3 (Section 6): crossed packages are harder for two of three models, with or without a legislator persona.
(a) Crossed minus party-line choice entropy (bits) per model: Democratic personas and Republican personas (preregistered tests B1, B2:
    interval from the sealed RESULTS.json, filled = supported, dashed = minimum effect) and the no-persona arm (two-way item bootstrap,
    descriptive, diamond). (b) Share of decided crossed choices that go to the package satisfying the persona's economic position
    (descriptive), with the number of decided units. All values from the sealed files.
Usage: python paper/figures/fig3_packages.py -> paper/figures/fig3_packages.pdf and .png (300 dpi)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]; S = R / "sealed"; OUT = Path(__file__).resolve().parent
J = lambda p: json.load(open(p, encoding="utf-8"))
MODELS = [("qwen", "Qwen"), ("llama", "Llama"), ("olmo", "OLMo")]
T = {k: J(S / "STUDY_L_battery" / "out" / k / "RESULTS.json")["blockB_natural_cross_pressure"]["tests"] for k, _ in MODELS}
NP = J(S / "review" / "correction_2026-10-16c" / "no_persona_package_effect.json")
COL = {"Democrat": "#2f6fdb", "Republican": "#d6402f", "none": "#555555"}
ROWS = [("Democrat", "B1_Democrat", "Democratic persona"), ("Republican", "B2_Republican", "Republican persona"), ("none", None, "No persona")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.45})

yt, yl = [], []
for g, (key, name) in enumerate(MODELS):
    base = (len(MODELS) - 1 - g) * 4.2
    for j, (party, test, lab) in enumerate(ROWS):
        y = base + 2 - j
        if test:
            v = T[key][test]; est, (lo, hi), filled, mk = v["est"], v["ci95"], v["supported"], "o"
        else:
            v = NP[key]["interval_2026_10_18"]; est, (lo, hi), filled, mk = v["est"], v["ci95_pigeonhole"], True, "D"
        ax.plot([lo, hi], [y, y], color=COL[party], lw=1.4, solid_capstyle="butt")
        ax.scatter(est, y, marker=mk, s=30 if mk == "o" else 24, zorder=3, color=COL[party] if filled else "white", edgecolor=COL[party], lw=1.1)
        yt.append(y); yl.append(lab)
    ax.text(0.01, base + 2.85, name, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, fontweight="bold")
ax.axvline(0, color="#555", lw=0.6); ax.axvline(0.05, color="#8e44ad", lw=0.8, ls=(0, (3, 2)))
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.8, 11.6); ax.grid(axis="x", color="#ececec", lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Crossed minus party-line choice entropy (bits)")
ax.set_title("(a) Uncertainty of crossed choices", loc="left")

for g, (key, name) in enumerate(MODELS):
    for j, party in enumerate(["Democrat", "Republican"]):
        x = g + (j - 0.5) * 0.32; s = T[key]["economic_dominance_share_by_party"][party]; n = T[key]["decided_units_by_party"][party]
        bx.bar(x, s, width=0.28, color=COL[party], alpha=0.9)
        bx.text(x, 0.03, f"n={n}", ha="center", va="bottom", fontsize=6, color="white", rotation=90)
bx.axhline(0.5, color="#555", lw=0.6, ls=(0, (3, 2)))
bx.set_xticks(range(3)); bx.set_xticklabels([n for _, n in MODELS]); bx.set_ylim(0, 1.08); bx.set_yticks([0, 0.25, 0.5, 0.75, 1])
bx.set_ylabel("Share going to the economic position"); bx.set_title("(b) Decided crossed choices", loc="left")

h = [plt.Line2D([], [], marker="o", ls="", color="#2f6fdb", markersize=5, label="Democratic persona"),
     plt.Line2D([], [], marker="o", ls="", color="#d6402f", markersize=5, label="Republican persona"),
     plt.Line2D([], [], marker="D", ls="", color="#555555", markersize=4.5, label="no persona (two-way passage bootstrap)"),
     plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777", markersize=5, label="preregistered test not supported"),
     plt.Line2D([], [], color="#8e44ad", lw=0.8, ls=(0, (3, 2)), label="minimum effect (+0.05)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.8, handletextpad=0.3)
fig.savefig(OUT / "fig3_packages.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig3_packages.png", dpi=300, bbox_inches="tight")
print("fig3 written")

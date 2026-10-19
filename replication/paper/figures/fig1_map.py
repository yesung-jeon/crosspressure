#!/usr/bin/env python
"""Figure 1 (Section 4): the map recovers aligned members but not cross-pressured ones.
(a)-(c) One map per model (Qwen, Llama, OLMo): member-level positions of the 60 legislators selected by their votes (Appendix A.4),
    each the mean of the member's median-centred map coordinates over the selected rows the model recognizes (as STUDY_D_quadrant/
    code/s4c_positions.py), coloured by voting quadrant; grey background = recognized Democratic and Republican member-Congress rows from
    the 93rd Congress on, median-centred. Labels: the two members with the largest depth in each cross-pressured quadrant (a fixed rule
    on votes; the same four people in every panel).
(d) Recall by voting quadrant, three models, from the sealed vote_vs_map.json.
(e) Vote-selected members placed in their own quadrant (of 15), three models, from the sealed S4C.json.
Guard: the placements recomputed for (a)-(c) must equal the sealed counts for every model. Usage: python paper/figures/fig1_map.py
Outputs: paper/figures/fig1_map.pdf and .png (300 dpi)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]; D = R / "STUDY_D_quadrant"; S = R / "sealed" / "STUDY_D_quadrant" / "out"; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(D / "code"))
import dcommon as dc  # noqa: E402

Q = ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]; ALIGNED, CROSS = Q[:2], Q[2:]
COL = {"EL-SP": "#2f6fdb", "ER-ST": "#d6402f", "EL-ST": "#8e44ad", "ER-SP": "#e08a00"}
MODELS = [("qwen", "Qwen", "o"), ("llama", "Llama", "s"), ("olmo", "OLMo", "^")]
J = lambda p: json.load(open(p, encoding="utf-8"))
s4c = {"qwen": J(S / "s4c" / "S4C.json")["qwen"], "llama": J(S / "s4c" / "S4C.json")["llama"], "olmo": J(S / "s4c_olmo" / "S4C.json")["olmo"]}
vm = {"qwen": J(S / "s4" / "vote_vs_map.json")["qwen"], "llama": J(S / "s4" / "vote_vs_map.json")["llama"], "olmo": J(S / "s4_olmo" / "vote_vs_map.json")["olmo"]}
JR = dc.s4_selection_rows()
SEL0 = pd.read_csv(D / "out" / "s4c" / "selection.csv")
LABELLED = [i for q in CROSS for i in SEL0[SEL0.vote_q == q].nlargest(2, "depth").icpsr]

def map_data(key):
    """the computation of s4c_positions.py for one model"""
    m = pd.read_csv(D / "out" / key / "s3" / "temporal_member_scores.csv"); m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m = m[m.party.isin(["Democrat", "Republican"]) & m.known.astype(bool)].copy(); m["cc"] = m.congress.astype(str) + m.chamber
    for c in ["x", "y"]:
        m[f"{c}m"] = m[c] - m.groupby("cc")[c].transform("median")
    pos = m.merge(JR, on=["congress", "chamber", "icpsr"]).groupby("icpsr")[["xm", "ym"]].mean()
    sel = SEL0.join(pos, on="icpsr")
    sel["map_q"] = (pd.Series(np.where(sel.xm > 0, "ER-", "EL-"), index=sel.index) + np.where(sel.ym > 0, "ST", "SP")).where(sel.xm.notna())
    placed = {q: int(n) for q, n in (sel.map_q == sel.vote_q).groupby(sel.vote_q).sum().items()}
    assert placed == {q: s4c[key][q]["placed"] for q in Q}, (key, placed)
    return m[m.congress >= 93], sel, placed

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1], hspace=0.32, top=0.94)
top = gs[0].subgridspec(1, 3, wspace=0.08); bot = gs[1].subgridspec(1, 2, wspace=0.42)

def short(r): return f"{r['name'].split()[0][0]}. {' '.join(r['name'].split()[1:])} ({r.party[0]}-{r.state})"

def seg_cross(p1, p2, p3, p4):
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return d(p1, p2, p3) * d(p1, p2, p4) < 0 and d(p3, p4, p1) * d(p3, p4, p2) < 0

def seg_box(p1, p2, b):
    c = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    return any(seg_cross(p1, p2, c[k], c[(k + 1) % 4]) for k in range(4))

placed_all = {}
for i, (key, name, _) in enumerate(MODELS):
    bg, sel, placed_all[key] = map_data(key)
    ax = fig.add_subplot(top[i])
    ax.scatter(bg.xm, bg.ym, s=1, color="#c9c9c9", alpha=0.3, lw=0, rasterized=True)
    ax.axhline(0, color="#555", lw=0.5, zorder=1); ax.axvline(0, color="#555", lw=0.5, zorder=1)
    for q in Q:
        g = sel[(sel.vote_q == q) & sel.xm.notna()]; ok = g.map_q == q
        ax.scatter(g[ok].xm, g[ok].ym, s=16, color=COL[q], edgecolor="white", lw=0.4, zorder=3)
        ax.scatter(g[~ok].xm, g[~ok].ym, s=16, facecolor="white", edgecolor=COL[q], lw=1.0, zorder=3)
    lx = 1.12 * max(np.nanpercentile(np.abs(bg.xm), 99.5), sel.xm.abs().max()); ly = 1.12 * max(np.nanpercentile(np.abs(bg.ym), 99.5), sel.ym.abs().max())
    ax.set_xlim(-lx, lx); ax.set_ylim(-ly, ly); ax.set_xticks([]); ax.set_yticks([])
    # labels: the same four members in each panel; the label order that keeps leader lines off the other label is chosen on rendered extents
    for q, (tx, ha, y0) in {"EL-ST": (-0.95 * lx, "left", 0.80 * ly), "ER-SP": (0.95 * lx, "right", -0.58 * ly)}.items():
        two = sel[sel.icpsr.isin(LABELLED) & (sel.vote_q == q) & sel.xm.notna()]
        slots = [(tx, y0), (tx, y0 - 0.17 * ly)]; best = None
        for order in (two, two.iloc[::-1]):
            arts = [ax.annotate(short(r), (r.xm, r.ym), xytext=slots[k], fontsize=5.6, ha=ha, va="center", color=COL[q], zorder=6,
                                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                                arrowprops=dict(arrowstyle="-", lw=0.45, color=COL[q], shrinkA=1.5, shrinkB=2.5, relpos=(1, 0.5) if ha == "left" else (0, 0.5)))
                    for k, (_, r) in enumerate(order.iterrows())]
            fig.canvas.draw(); inv = ax.transData.inverted()
            boxes = [tuple(inv.transform(a.get_window_extent().get_points()).ravel()) for a in arts]
            ends = [((b[2] if ha == "left" else b[0]), (b[1] + b[3]) / 2) for b in boxes]; pts = [(r.xm, r.ym) for _, r in order.iterrows()]
            bad = (len(arts) == 2) * (seg_cross(ends[0], pts[0], ends[1], pts[1]) + seg_box(ends[0], pts[0], boxes[1]) + seg_box(ends[1], pts[1], boxes[0]))
            if best is None or bad < best[0]:
                if best: [a.remove() for a in best[1]]
                best = (bad, arts)
            else:
                [a.remove() for a in arts]
    for tx, ty, s in [(0.97, 0.97, "ER-ST"), (0.03, 0.97, "EL-ST"), (0.03, 0.03, "EL-SP"), (0.97, 0.03, "ER-SP")]:
        ax.text(tx, ty, s, transform=ax.transAxes, ha="right" if tx > 0.5 else "left", va="top" if ty > 0.5 else "bottom",
                fontsize=6.5, color=COL[s], fontweight="bold", zorder=2.5)   # under the points
    ax.set_title(f"({'abc'[i]}) {name} map", loc="left")
    ax.set_xlabel("Economic (left ←  → right)", fontsize=7)
    if i == 0: ax.set_ylabel("Social (progressive ←  → traditional)", fontsize=7)
h1 = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="#777", markeredgecolor="white", markersize=5, label="legislator placed in own voting quadrant"),
      plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777", markeredgewidth=1.1, markersize=5, label="placed elsewhere")]
fig.legend(handles=h1, loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=7, handletextpad=0.2)

def dotpanel(ax, val, title, xlabel, xmax, ticks):
    ys = {q: i for i, q in enumerate(Q[::-1])}
    ax.axhspan(-0.5, 1.5, color="#f3eef7", zorder=0)
    for k, (key, name, mk) in enumerate(MODELS):
        off = (1 - k) * 0.24
        for q in Q:
            ax.scatter(val[key][q], ys[q] + off, marker=mk, s=24, color=COL[q], edgecolor="white", lw=0.4, zorder=3, clip_on=False)
    ax.set_yticks(range(4)); ax.set_yticklabels(Q[::-1]); [t.set_color(COL[t.get_text()]) for t in ax.get_yticklabels()]
    ax.set_xlim(0, xmax); ax.set_xticks(ticks); ax.set_ylim(-0.6, 3.6); ax.grid(axis="x", color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
    ax.text(xmax * 0.97, 0.5, "cross-\npressured", ha="right", va="center", fontsize=6.5, color="#6b4c7a", style="italic")
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)

ax_d = fig.add_subplot(bot[0])
dotpanel(ax_d, {k: vm[k]["recall_by_vote_quadrant"] for k, _, _ in MODELS}, "(d) All member-Congress rows", "Share placed in own voting quadrant", 1.0, [0, 0.25, 0.5, 0.75, 1])
ax_e = fig.add_subplot(bot[1])
dotpanel(ax_e, {k: {q: s4c[k][q]["placed"] for q in Q} for k, _, _ in MODELS}, "(e) Legislators selected by votes", "Number placed in own voting quadrant (of 15)", 15, [0, 5, 10, 15])
handles = [plt.Line2D([], [], marker=mk, ls="", color="#444", markersize=5, label=name) for _, name, mk in MODELS]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=7, handletextpad=0.2)

fig.savefig(OUT / "fig1_map.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig1_map.png", dpi=300, bbox_inches="tight")
print("fig1 written; placements recomputed and equal to sealed:", placed_all)

# Section 4 names four members; check that every statement made about them holds (text = figure)
def placed_member(key, last):
    _, sel, _ = map_data(key); r = sel[sel.name.str.contains(last)].iloc[0]; return r.map_q == r.vote_q
assert not placed_member("qwen", "Conable") and not placed_member("qwen", "Frenzel"), "Qwen misplaces Conable and Frenzel"
assert placed_member("llama", "Conable") and placed_member("llama", "Frenzel"), "Llama places both"
assert placed_member("olmo", "Conable") + placed_member("olmo", "Frenzel") == 1, "OLMo places one"
assert placed_member("qwen", "Eagleton") and not placed_member("llama", "Eagleton") and not placed_member("olmo", "Eagleton"), "Eagleton"
print("Section 4 statements about named members hold")

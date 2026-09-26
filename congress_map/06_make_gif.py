#!/usr/bin/env python
"""Step 6 - animated GIF of the map (one frame per Congress + 2 tween frames): out/temporal_member_scores.csv -> out/congress_map.gif
Usage: python 06_make_gif.py [out_dir]"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
m = pd.read_csv(f"{OUT}/temporal_member_scores.csv")
m["x"] = -m.econ_orth; m["y"] = -m.social_orth
m["party"] = m.party.where(m.party.isin(["Democrat", "Republican"]), "Other")
COL = {"Democrat": "#2F6FCF", "Republican": "#D24B2A", "Other": "#7B8794"}; MK = {"Democrat": "o", "Republican": "s", "Other": "^"}
xl = (m.x.quantile(.002) - .05, m.x.quantile(.998) + .05); yl = (m.y.quantile(.002) - .05, m.y.quantile(.998) + .05)
congs = sorted(m.congress.unique())
def ordn(n):
    v = n % 100
    return f"{n}{'th' if 11 <= v <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
means = {c: {p: (g.x.mean(), g.y.mean()) for p, g in m[m.congress == c].groupby("party") if p != "Other"} for c in congs}
frames = []
def render(c, tween, prev):
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=100)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.axhline(0, color="#B6BEC9", lw=1); ax.axvline(0, color="#B6BEC9", lw=1); ax.grid(color="#E1E6EC", lw=.6)
    cur = m[m.congress == c].set_index("icpsr"); pv = m[m.congress == prev].set_index("icpsr") if prev is not None else None
    for p in ["Other", "Democrat", "Republican"]:
        g = cur[cur.party == p]
        if len(g) == 0:
            continue
        if pv is not None and tween < 1:
            both = g.index.intersection(pv.index)
            x = g.x.copy(); y = g.y.copy()
            x.loc[both] = pv.loc[both].x + (g.loc[both].x - pv.loc[both].x) * tween
            y.loc[both] = pv.loc[both].y + (g.loc[both].y - pv.loc[both].y) * tween
            a = np.where(g.index.isin(both), .55, .55 * tween)
        else:
            x, y, a = g.x, g.y, .55
        ax.scatter(x, y, s=14, c=COL[p], marker=MK[p], alpha=a, linewidths=0)
    # party-mean trails
    for p in ["Democrat", "Republican"]:
        pts = [means[k][p] for k in congs if k <= c and p in means[k]]
        if pts:
            xs, ys = zip(*pts); ax.plot(xs, ys, color=COL[p], lw=2, alpha=.5)
            ax.scatter([xs[-1]], [ys[-1]], s=140, facecolors="white", edgecolors=COL[p], linewidths=2.5, marker=MK[p], zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], s=30, c=COL[p], marker=MK[p], zorder=6)
    yr = 1789 + 2 * (c - 1)
    ax.text(.02, .97, f"{ordn(c)} Congress · {yr}–{yr+1}", transform=ax.transAxes, fontsize=18, weight="bold", va="top", family="DejaVu Sans")
    n = (m.congress == c).sum(); kn = m[m.congress == c].known.mean()
    ax.text(.02, .91, f"{n} members · party recalled for {kn*100:.0f}%", transform=ax.transAxes, fontsize=11, color="#4B5563", va="top")
    ax.text(.98, .03, "● Democrats   ■ Republicans   ▲ Other     large outlined marker = party mean", transform=ax.transAxes, fontsize=10, color="#4B5563", ha="right")
    ax.set_xlabel("Economic: left  ←              → right   (model units)", fontsize=11); ax.set_ylabel("Social: progressive  ←              → traditional", fontsize=11)
    ax.annotate("model default (0,0)", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#8A94A3")
    fig.suptitle("Congress in Model Space — U.S. legislators projected onto the economic and social axes of Qwen2.5-7B (1947–2023)", fontsize=12, y=.995)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)
prev = None
for c in congs:
    if prev is not None:
        for t in (0.33, 0.66):
            frames.append((render(c, t, prev), 90))
    frames.append((render(c, 1.0, prev), 520))
    prev = c
frames[-1] = (frames[-1][0], 2500)
imgs = [f for f, _ in frames]; durs = [d for _, d in frames]
imgs[0].save(f"{OUT}/congress_map.gif", save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=False)
import os; print(f"wrote {OUT}/congress_map.gif", os.path.getsize(f"{OUT}/congress_map.gif") // 1024, "KB", len(frames), "frames")

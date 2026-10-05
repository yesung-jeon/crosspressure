#!/usr/bin/env python
"""Step 6 - GIF of the uncertainty map (one frame per sampled Congress; same windows and colour scale as step 5).
Output: out/uncertainty_map.gif
Usage: python 06_make_gif.py [--out out]
"""
import argparse, io
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out
G = pd.read_csv(OUT / "member_generations.csv"); G = G[G.party.isin(["Democrat", "Republican"])]
M = G.groupby(["icpsr", "congress"]).agg(entropy=("entropy64", "mean"), party=("party", "first"), state=("state", "first"),
                                        x=("x", "first"), y=("y", "first")).reset_index()
lo, hi = np.percentile(M.entropy, [5, 95]); cmap = plt.get_cmap("viridis")
xr = (M.x.min() - .2, M.x.max() + .2); yr = (min(0, M.y.min()) - .2, M.y.max() + .2)
TILES = {"AK": (0, 0), "ME": (11, 0), "VT": (10, 1), "NH": (11, 1), "WA": (1, 2), "ID": (2, 2), "MT": (3, 2), "ND": (4, 2), "MN": (5, 2),
         "IL": (6, 2), "WI": (7, 2), "MI": (8, 2), "NY": (9, 2), "RI": (10, 2), "MA": (11, 2), "OR": (1, 3), "NV": (2, 3), "WY": (3, 3),
         "SD": (4, 3), "IA": (5, 3), "IN": (6, 3), "OH": (7, 3), "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3), "CA": (1, 4), "UT": (2, 4),
         "CO": (3, 4), "NE": (4, 4), "MO": (5, 4), "KY": (6, 4), "WV": (7, 4), "VA": (8, 4), "MD": (9, 4), "DE": (10, 4), "AZ": (2, 5),
         "NM": (3, 5), "KS": (4, 5), "AR": (5, 5), "TN": (6, 5), "NC": (7, 5), "SC": (8, 5), "OK": (4, 6), "LA": (5, 6), "MS": (6, 6),
         "AL": (7, 6), "GA": (8, 6), "HI": (0, 7), "TX": (4, 7), "FL": (9, 7)}
frames = []
for f in range(80, 119, 2):
    w = M[(M.congress - f).abs() <= 4]; norm = lambda e: cmap(np.clip((e - lo) / (hi - lo), 0, 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax[0].axvline(0, color=".85", ls="--", lw=.8); ax[0].axhline(0, color=".85", ls="--", lw=.8)
    ax[0].scatter(w.x, w.y, s=22, c=[norm(e) for e in w.entropy], edgecolors=np.where(w.party == "Republican", "#d1495b", "#3a7bd5"), linewidths=.8)
    ax[0].set_xlim(*xr); ax[0].set_ylim(*yr); ax[0].set_xlabel("economic (right = right)"); ax[0].set_ylabel("social (up = traditional)")
    ax[0].set_title(f"{cm.year_of(f)}  members (window +-4 Congresses); origin = model default", fontsize=9)
    for s, (c, r) in TILES.items():
        g = w[w.state == s]; fc = norm(g.entropy.mean()) if len(g) else (.9, .9, .9, 1)
        ax[1].add_patch(plt.Rectangle((c, -r), .92, .92, color=fc)); ax[1].text(c + .46, -r + .4, s, ha="center", va="center", fontsize=7, color="w")
    ax[1].set_xlim(-.2, 12.2); ax[1].set_ylim(-7.3, 1.1); ax[1].axis("off"); ax[1].set_title("state mean entropy (yellow = more hesitant)", fontsize=9)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); plt.close(fig); frames.append(Image.open(buf).convert("P"))
frames[0].save(OUT / "uncertainty_map.gif", save_all=True, append_images=frames[1:], duration=700, loop=0)
cm.log(f"saved {OUT / 'uncertainty_map.gif'} ({len(frames)} frames)")

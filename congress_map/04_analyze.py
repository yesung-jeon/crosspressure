#!/usr/bin/env python
"""Step 4 - per-Congress statistics and the JSON payload for the map.
Reads out/temporal_member_scores.csv; writes out/temporal_congress_stats.csv and out/temporal_payload.json.
x = -econ_orth (right = economic right), y = -social_orth (up = traditional). Usage: python 04_analyze.py [out_dir]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")  # directory holding temporal_member_scores.csv
m = pd.read_csv(OUT / "temporal_member_scores.csv")
m["x"] = -m.econ_orth          # x: left <- -> right (model units)
m["y"] = -m.social_orth        # y: progressive (down) <- -> traditional (up)
m["party"] = m.party.where(m.party.isin(["Democrat", "Republican"]), "Other")
rows = []
for c, g in m.groupby("congress"):
    d, r = g[g.party == "Democrat"], g[g.party == "Republican"]
    def cd(col):
        return (r[col].mean() - d[col].mean()) / np.sqrt((d[col].var() + r[col].var()) / 2) if len(d) > 1 and len(r) > 1 else np.nan
    rows.append(dict(congress=int(c), year=int(g.year.iloc[0]), n=len(g), known=round(g.known.mean(), 3),
                     dem_x=d.x.mean(), rep_x=r.x.mean(), dem_y=d.y.mean(), rep_y=r.y.mean(),
                     dem_sd_x=d.x.std(), rep_sd_x=r.x.std(), dem_sd_y=d.y.std(), rep_sd_y=r.y.std(),
                     d_x=cd("x"), d_y=cd("y"), d_dim1=(r.dim1.mean() - d.dim1.mean()) / np.sqrt((d.dim1.var() + r.dim1.var()) / 2),
                     r_x_dim1=pearsonr(g.x, g.dim1)[0], r_y_dim1=pearsonr(g.y, g.dim1)[0],
                     r_x_dim1_dem=pearsonr(d.x, d.dim1)[0] if len(d) > 3 else np.nan, r_x_dim1_rep=pearsonr(r.x, r.dim1)[0] if len(r) > 3 else np.nan,
                     dem_dim1=d.dim1.mean(), rep_dim1=r.dim1.mean()))
S = pd.DataFrame(rows).round(4)
S.to_csv(OUT / "temporal_congress_stats.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S[["congress", "year", "n", "known", "dem_x", "rep_x", "dem_y", "rep_y", "d_x", "d_y", "d_dim1", "r_x_dim1", "r_y_dim1", "r_x_dim1_dem", "r_x_dim1_rep"]].to_string(index=False))
# trails: members serving in the most Congresses (within 80-118)
cnt = m.groupby("icpsr").congress.nunique().sort_values(ascending=False)
trail_ids = cnt.head(14).index.tolist()
names = m.drop_duplicates("icpsr").set_index("icpsr")["name"]
print("\ntrail members:", [(names[i], int(cnt[i])) for i in trail_ids])
pc = {"Democrat": 0, "Republican": 1, "Other": 2}
payload = dict(
    congresses=[dict(c=int(r.congress), year=int(r.year), n=int(r.n), known=float(r.known), dem=[round(r.dem_x, 3), round(r.dem_y, 3)], rep=[round(r.rep_x, 3), round(r.rep_y, 3)],
                     d_x=round(r.d_x, 3), d_y=round(r.d_y, 3), d_dim1=round(r.d_dim1, 3), r_x=round(r.r_x_dim1, 3), r_y=round(r.r_y_dim1, 3)) for r in S.itertuples()],
    members=[[int(a.congress), pc[a.party], round(a.x, 3), round(a.y, 3), round(a.dim1, 3), a.name, a.state, a.chamber[0], int(a.icpsr), int(bool(a.known))]
             for a in m.itertuples()],
    trails=[dict(icpsr=int(i), name=names[i], party=pc[m[m.icpsr == i].party.iloc[0]],
                 pts=[[int(a.congress), round(a.x, 3), round(a.y, 3)] for a in m[m.icpsr == i].sort_values("congress").itertuples()]) for i in trail_ids],
    xlim=[round(float(m.x.quantile(0.002)), 2), round(float(m.x.quantile(0.998)), 2)], ylim=[round(float(m.y.quantile(0.002)), 2), round(float(m.y.quantile(0.998)), 2)],
)
json.dump(payload, open(OUT / "temporal_payload.json", "w"), separators=(",", ":"))
print("payload:", (OUT / "temporal_payload.json").stat().st_size // 1024, "KB")

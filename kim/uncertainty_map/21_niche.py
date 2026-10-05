#!/usr/bin/env python
"""Step 21 - validate the corrected map (A), party niche ecology (B), the cross-pressure niche (C). CPU.

Maps compared: raw (Kim's definition, int8 run), corrected (year-matched ordinary people; primary), perp (time direction removed).
A Validation against roll calls: party means vs Nokken-Poole dim1 over time (levels and first differences) by party;
  within-member drift (member FE: NP dim1 + years since first Congress); 1947 -> 2023 change of each party (who moved?).
B Niche ecology per Congress and map: party centroid; niche width = sqrt of the determinant of the party's 2x2 covariance;
  overlap = Bhattacharyya coefficient of the two parties' bivariate normal fits (1 = identical niches, 0 = disjoint);
  separation per axis = Cohen's d (R - D) on x and on y (which axis separated first?); Southern Democrats' distance to the
  other Democrats and to Republicans.
C Cross-pressure niche: quadrants relative to the Congress's own centroid (as in Kim's figure) and, on the corrected map,
  relative to the ordinary people of the time (origin 0): share of members in EL-ST (economic left + social traditional) and
  ER-SP; who occupies EL-ST (Southern Democrats?) and when it empties. Hesitation: persona entropy (step 1 members) on being in
  EL-ST / ER-SP and on local density in the corrected map (leave-one-out kernel, same Congress), controls as in step 4.
Output: out/niche_summary.json, out/NICHE_RESULTS.md, out/niche_by_congress.csv, out/fig_niche.png
Usage: python 21_niche.py [--out out]
"""
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; R = {}; f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
M = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "members_*.csv")))], ignore_index=True)
v = pd.read_csv(cm.VOTEVIEW, low_memory=False); v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])
M = M.merge(v[["congress", "icpsr", "nokken_poole_dim1"]], on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1"})
M = M[M.party.isin(["Democrat", "Republican"])].copy()
M["south_dem"] = (M.party == "Democrat") & M.state.isin(cm.SOUTH)
MAPS = {"raw": ("x_raw", "y_raw"), "corrected": ("x_c", "y_c"), "perp": ("x_perp", "y_perp")}
R["n"] = dict(member_congresses=int(len(M)), congresses=int(M.congress.nunique()), corr_raw_with_kim=dict(x=float(np.corrcoef(M.x_raw, M.x_kim)[0, 1]), y=float(np.corrcoef(M.y_raw, M.y_kim)[0, 1])))

# ======================= A validation =======================
A = {}
for mk, (xc, yc) in MAPS.items():
    pm = M.groupby(["congress", "party"])[[xc, yc, "np1", "dim1"]].mean().reset_index(); o = {}
    for p, g in pm.groupby("party"):
        g = g.sort_values("congress")
        for ax_, col in (("x", xc), ("y", yc)):
            o[f"{p}_{ax_}_levels_r_np1"] = float(pearsonr(g[col], g.np1)[0]); o[f"{p}_{ax_}_diff_r_np1"] = float(pearsonr(g[col].diff().dropna(), g.np1.diff().dropna())[0])
            o[f"{p}_{ax_}_change_1947_2023"] = float(g[col].iloc[-1] - g[col].iloc[0])
        o[f"{p}_np1_change_1947_2023"] = float(g.np1.iloc[-1] - g.np1.iloc[0])
    d = M.dropna(subset=["np1"]).copy(); d = d[d.groupby("icpsr").congress.transform("size") >= 2]; d["tenure"] = d.year - d.groupby("icpsr").year.transform("min")
    for ax_, col in (("x", xc), ("y", yc)):
        cols = [col, "np1", "tenure"]; w = d[cols] - d.groupby("icpsr")[cols].transform("mean"); w["icpsr"] = d.icpsr.values
        r = cm.fit_cr1(w, f"{col} ~ 0 + np1 + tenure", "icpsr"); o[f"within_member_{ax_}"] = {k: cm.coef(r, k) for k in ("np1", "tenure")}
    for ax_, col in (("x", xc), ("y", yc)):
        o[f"within_party_r_{ax_}_dim1"] = {p: float(np.mean([np.corrcoef(h[col], h.dim1)[0, 1] for _, h in g.groupby("congress")])) for p, g in M.groupby("party")}
    A[mk] = o
R["A_validation"] = A

# ======================= B niche ecology =======================
def bhatt(m1, S1, m2, S2):
    S = (S1 + S2) / 2; d = m1 - m2
    db = d @ np.linalg.solve(S, d) / 8 + 0.5 * np.log(np.linalg.det(S) / np.sqrt(np.linalg.det(S1) * np.linalg.det(S2))); return float(np.exp(-db))
rows = []
for mk, (xc, yc) in MAPS.items():
    for c, g in M.groupby("congress"):
        row = dict(map=mk, congress=c, year=cm.year_of(c)); P = {}
        for p, gp in g.groupby("party"):
            X = gp[[xc, yc]].values; mu, S = X.mean(0), np.cov(X.T); P[p] = (mu, S)
            row[f"{p[:3]}_x"], row[f"{p[:3]}_y"], row[f"{p[:3]}_width"] = mu[0], mu[1], float(np.sqrt(np.linalg.det(S)))
        row["overlap"] = bhatt(*P["Democrat"], *P["Republican"])
        for ax_, col in (("x", xc), ("y", yc)):
            a, b = g[g.party == "Republican"][col], g[g.party == "Democrat"][col]
            row[f"d_{ax_}"] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
        sd, od, rp = g[g.south_dem], g[(g.party == "Democrat") & ~g.south_dem], g[g.party == "Republican"]
        if len(sd) > 5:
            c_sd, c_od, c_rp = sd[[xc, yc]].mean().values, od[[xc, yc]].mean().values, rp[[xc, yc]].mean().values
            row["southdem_to_otherdem"], row["southdem_to_rep"] = float(np.linalg.norm(c_sd - c_od)), float(np.linalg.norm(c_sd - c_rp))
        # C: quadrants relative to the Congress centroid, and (corrected map) relative to ordinary people of the time (0, 0)
        for ref, (cx, cy) in (("centroid", (g[xc].mean(), g[yc].mean())), ("origin", (0.0, 0.0))):
            el_st = (g[xc] < cx) & (g[yc] > cy); er_sp = (g[xc] > cx) & (g[yc] < cy)
            row[f"EL_ST_{ref}"], row[f"ER_SP_{ref}"] = float(el_st.mean()), float(er_sp.mean())
            row[f"EL_ST_{ref}_southdem_share"] = float(g[el_st].south_dem.mean()) if el_st.sum() else np.nan
        rows.append(row)
NB = pd.DataFrame(rows); NB.to_csv(OUT / "niche_by_congress.csv", index=False)
B = {}
for mk in MAPS:
    nb = NB[NB["map"] == mk].sort_values("congress")
    first = lambda col, thr: (int(nb[nb[col] >= thr].year.iloc[0]) if (nb[col] >= thr).any() else None)
    B[mk] = dict(overlap_1947=float(nb.overlap.iloc[0]), overlap_2023=float(nb.overlap.iloc[-1]),
                 overlap_first_below_0_5=(int(nb[nb.overlap < .5].year.iloc[0]) if (nb.overlap < .5).any() else None),
                 d_x_1947=float(nb.d_x.iloc[0]), d_x_2023=float(nb.d_x.iloc[-1]), d_y_1947=float(nb.d_y.iloc[0]), d_y_2023=float(nb.d_y.iloc[-1]),
                 first_year_d_x_ge_1=first("d_x", 1.0), first_year_d_y_ge_1=first("d_y", 1.0),
                 width_dem_1947_2023=[float(nb.Dem_width.iloc[0]), float(nb.Dem_width.iloc[-1])], width_rep_1947_2023=[float(nb.Rep_width.iloc[0]), float(nb.Rep_width.iloc[-1])],
                 southdem_to_otherdem_by_era={e: float(nb[nb.congress.between(a, b)].southdem_to_otherdem.mean()) for e, (a, b) in {"1947-79": (80, 96), "1981-2005": (97, 109), "2007-23": (110, 118)}.items()},
                 southdem_to_rep_by_era={e: float(nb[nb.congress.between(a, b)].southdem_to_rep.mean()) for e, (a, b) in {"1947-79": (80, 96), "1981-2005": (97, 109), "2007-23": (110, 118)}.items()})
R["B_niche"] = B

# ======================= C cross-pressure niche and hesitation =======================
Cn = {}
for mk in MAPS:
    nb = NB[NB["map"] == mk]
    Cn[mk] = {f"{q}_{ref}_by_era": {e: float(nb[nb.congress.between(a, b)][f"{q}_{ref}"].mean()) for e, (a, b) in {"1947-79": (80, 96), "1981-2005": (97, 109), "2007-23": (110, 118)}.items()}
              for q in ("EL_ST", "ER_SP") for ref in ("centroid", "origin")}
    Cn[mk]["EL_ST_centroid_southdem_share_by_era"] = {e: float(nb[nb.congress.between(a, b)].EL_ST_centroid_southdem_share.mean()) for e, (a, b) in {"1947-79": (80, 96), "1981-2005": (97, 109), "2007-23": (110, 118)}.items()}
G = pd.read_csv(OUT / "member_generations.csv"); G = G[G.party.isin(["Democrat", "Republican"])]
G = G.merge(M[["icpsr", "congress", "x_c", "y_c"]], on=["icpsr", "congress"], how="inner", validate="m:1")
dens = {}
for c, g in M.groupby("congress"):
    X = g[["x_c", "y_c"]].values; X = (X - X.mean(0)) / X.std(0); h = .5
    K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * h * h)); np.fill_diagonal(K, 0)
    for i, ic in enumerate(g.icpsr.values): dens[(c, ic)] = np.log(K[i].sum() / (len(X) - 1) + 1e-12)
G["log_density_c"] = [dens[(c, i)] for c, i in zip(G.congress, G.icpsr)]
cen = M.groupby("congress")[["x_c", "y_c"]].mean()
G["EL_ST"] = ((G.x_c < G.congress.map(cen.x_c)) & (G.y_c > G.congress.map(cen.y_c))).astype(int)
G["ER_SP"] = ((G.x_c > G.congress.map(cen.x_c)) & (G.y_c < G.congress.map(cen.y_c))).astype(int)
fe = "C(party) * C(era) + C(chamber) + C(q)"
Cn["entropy_on_quadrants"] = {k: cm.coef(cm.fit_cr1(G, f"entropy64 ~ EL_ST + ER_SP + dim1 + dim2 + {fe}", "icpsr"), k) for k in ("EL_ST", "ER_SP")}
Cn["entropy_on_local_density_corrected"] = cm.coef(cm.fit_cr1(G, f"entropy64 ~ log_density_c + x_c + y_c + I(x_c**2) + I(y_c**2) + x_c:y_c + {fe}", "icpsr"), "log_density_c")
Cn["entropy_on_corrected_position"] = {k: cm.coef(cm.fit_cr1(G, f"entropy64 ~ x_c + y_c + dim1 + dim2 + {fe}", "icpsr"), k) for k in ("x_c", "y_c")}
R["C_crosspressure"] = Cn
(OUT / "niche_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

# ======================= figure =======================
fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
for j, mk in enumerate(("raw", "corrected")):
    xc, yc = MAPS[mk]; nb = NB[NB["map"] == mk]
    for p, col in (("Dem", "tab:blue"), ("Rep", "tab:red")):
        ax[j, 0].plot(nb[f"{p}_x"], nb[f"{p}_y"], "-o", ms=2, color=col, label={"Dem": "Democrats", "Rep": "Republicans"}[p])
        ax[j, 0].annotate("1947", (nb[f"{p}_x"].iloc[0], nb[f"{p}_y"].iloc[0]), fontsize=7); ax[j, 0].annotate("2023", (nb[f"{p}_x"].iloc[-1], nb[f"{p}_y"].iloc[-1]), fontsize=7)
    ax[j, 0].set_xlabel("economic (right = right)"); ax[j, 0].set_ylabel("social (up = traditional)"); ax[j, 0].legend(fontsize=7)
    ax[j, 0].set_title(f"{mk}: party centroid paths 1947-2023" + (" (origin = model)" if mk == "raw" else " (origin = ordinary people of the year)"), fontsize=9)
    ax[j, 1].plot(nb.year, nb.overlap, "k-", label="niche overlap (Bhattacharyya)"); ax[j, 1].plot(nb.year, nb.d_x / 4, "-", color="tab:green", label="economic separation d / 4")
    ax[j, 1].plot(nb.year, nb.d_y / 4, "-", color="tab:purple", label="social separation d / 4"); ax[j, 1].legend(fontsize=7); ax[j, 1].set_title(f"{mk}: niche overlap and separation", fontsize=9)
    ax[j, 2].plot(nb.year, nb.EL_ST_centroid, "-", color="tab:brown", label="EL-ST share (vs centroid)"); ax[j, 2].plot(nb.year, nb.ER_SP_centroid, "-", color="tab:olive", label="ER-SP share (vs centroid)")
    ax[j, 2].plot(nb.year, nb.EL_ST_centroid_southdem_share, ":", color="tab:brown", label="Southern Dem share of EL-ST"); ax[j, 2].legend(fontsize=7); ax[j, 2].set_title(f"{mk}: cross-pressure quadrants", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_niche.png", dpi=105)

L = ["# A-B-C results (generated by 21_niche.py; exploratory, int8 re-projection)", "", f"{R['n']}", "", "## A validation (party means vs Nokken-Poole dim1; within member)"]
for mk, o in A.items():
    L.append(f"### {mk}")
    for p in ("Democrat", "Republican"):
        L.append(f"- {p}: x levels r {o[f'{p}_x_levels_r_np1']:+.2f}, diff r {o[f'{p}_x_diff_r_np1']:+.2f}; y levels r {o[f'{p}_y_levels_r_np1']:+.2f}, diff r {o[f'{p}_y_diff_r_np1']:+.2f}; change 1947-2023 x {o[f'{p}_x_change_1947_2023']:+.2f}, y {o[f'{p}_y_change_1947_2023']:+.2f}; NP dim1 {o[f'{p}_np1_change_1947_2023']:+.2f}")
    for ax_ in ("x", "y"):
        w = o[f"within_member_{ax_}"]; L.append(f"- within member {ax_}: NP dim1 {f(w['np1'])}; years {f(w['tenure'])}; within-party r with dim1 {o[f'within_party_r_{ax_}_dim1']}")
L += ["", "## B niche ecology"] + [f"- {mk}: {o}" for mk, o in B.items()] + ["", "## C cross-pressure niche"] + [f"- {k}: {o}" for k, o in Cn.items() if not k.startswith("entropy")]
L += [f"- entropy on quadrants (corrected, vs centroid): EL-ST {f(Cn['entropy_on_quadrants']['EL_ST'])}; ER-SP {f(Cn['entropy_on_quadrants']['ER_SP'])}",
      f"- entropy on local density (corrected map): {f(Cn['entropy_on_local_density_corrected'])}",
      f"- entropy on corrected position: x_c {f(Cn['entropy_on_corrected_position']['x_c'])}; y_c {f(Cn['entropy_on_corrected_position']['y_c'])}"]
(OUT / "NICHE_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

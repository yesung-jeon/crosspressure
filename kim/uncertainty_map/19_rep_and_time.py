#!/usr/bin/env python
"""Step 19 - (a) is the low Republican responsiveness the model or the yardstick? (b) what removing the time direction does. CPU.

(a) Within-party ideology rebuilt from raw roll calls, independent of NOMINATE: for each Congress (100, 106, 112, 118) x chamber
    x party, members x roll calls on which the party itself split (minority inside the party >= 10%), yea = 1 / nay = 0,
    missing filled with the roll-call mean, standardized, first principal component oriented to correlate positively with
    DW dim1. Second version with only roll calls whose description matches the topics of Kim's questions (child, family,
    parental, leave, environment, climate, pollution, abortion, marriage, gender, health, medicare, medicaid, retirement,
    pension, social security, student, education, school). Within-party correlation of Kim's x and y (bf16) with each yardstick
    (DW dim1, Nokken-Poole dim1, PCA all, PCA topic), by party; also known-members only, and the within-party SD of dim1.
(b) Time direction (step 18): cosines with Kim's axes; held-out persons by year (extrapolation to 2040 and 2060); legislator
    panel: y vs y_perp (social axis with time removed) in within-member dynamics (member FE: real drift vs calendar years) and
    in within-party responsiveness by party.
Output: out/rep_time_summary.json, out/REP_TIME_RESULTS.md, out/fig_rep_time.png
Usage: python 19_rep_and_time.py [--out out]
"""
import argparse, json, re
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out; VV = cm.HERE.parent / "data" / "voteview"; R = {}; f = lambda c: f"{c['est']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p = {c['p']:.3g}"
TOPIC = re.compile(r"child|famil|parental|leave|environment|climate|pollut|abortion|marriage|gender|health|medicare|medicaid|retire|pension|social security|student|educat|school", re.I)
m = cm.members(); m = m[m.party.isin(["Democrat", "Republican"])]
v = pd.read_csv(cm.VOTEVIEW, low_memory=False); v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])
m = m.merge(v[["congress", "icpsr", "nokken_poole_dim1"]], on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1"})

# ======================= (a) =======================
def pc1(mat):
    X = mat.apply(lambda c: c.fillna(c.mean())); X = (X - X.mean()) / X.std().replace(0, np.nan); X = X.dropna(axis=1)
    if X.shape[1] < 5: return None, X.shape[1]
    u, s, vt = np.linalg.svd(X.values - X.values.mean(0), full_matrices=False); return pd.Series(u[:, 0] * s[0], index=X.index), X.shape[1]
rows, counts = [], []
for c in [100, 106, 112, 118]:
    for k, ch in (("H", "House"), ("S", "Senate")):
        rc = pd.read_csv(VV / f"{k}{c}_rollcalls.csv"); vt = pd.read_csv(VV / f"{k}{c}_votes.csv")
        vt = vt[vt.cast_code.between(1, 6)].assign(yea=lambda d: (d.cast_code <= 3).astype(float))
        mm = m[(m.congress == c) & (m.chamber == ch)]
        vt = vt.merge(mm[["icpsr", "party"]], on="icpsr", how="inner", validate="m:1")
        desc = (rc.dtl_desc.fillna("") + " " + rc.vote_desc.fillna("")).set_axis(rc.rollnumber)
        for p, g in vt.groupby("party"):
            share = g.groupby("rollnumber").yea.mean(); split = share[(share >= .1) & (share <= .9)].index
            mat = g[g.rollnumber.isin(split)].pivot_table(index="icpsr", columns="rollnumber", values="yea")
            s_all, n_all = pc1(mat)
            tcols = [r for r in mat.columns if TOPIC.search(desc.get(r, ""))]
            s_top, n_top = pc1(mat[tcols]) if len(tcols) >= 5 else (None, len(tcols))
            counts.append(dict(congress=c, chamber=ch, party=p, split_rollcalls=int(n_all), topic_rollcalls=int(n_top)))
            sub = mm[mm.party == p].set_index("icpsr")
            for nm, s in (("pca_all", s_all), ("pca_topic", s_top)):
                if s is None: continue
                s = s.reindex(sub.index)
                if np.corrcoef(s.fillna(s.mean()), sub.dim1)[0, 1] < 0: s = -s
                sub[nm] = s
            sub = sub.reset_index().assign(congress=c, chamber=ch); rows.append(sub)
A = pd.concat(rows, ignore_index=True)
res = {}
for p, g in A.groupby("party"):
    o = {}
    for yard in ("dim1", "np1", "pca_all", "pca_topic"):
        for ax_ in ("x", "y"):
            gg = g.dropna(subset=[yard, ax_])
            # within chamber x Congress, then averaged (the PCA scales are only comparable inside a cell)
            rs = [np.corrcoef(h[ax_], h[yard])[0, 1] for _, h in gg.groupby(["congress", "chamber"]) if len(h) > 10]
            o[f"{ax_}~{yard}"] = float(np.mean(rs)) if rs else None
    kk = g[g.known.astype(bool)]
    o["x~dim1_known_only"] = float(np.mean([np.corrcoef(h.x, h.dim1)[0, 1] for _, h in kk.groupby(["congress", "chamber"]) if len(h) > 10]))
    o["sd_dim1_within_cell"] = float(np.mean([h.dim1.std() for _, h in g.groupby(["congress", "chamber"])]))
    o["corr_pca_all_with_dim1"] = float(np.mean([np.corrcoef(h.pca_all, h.dim1)[0, 1] for _, h in g.dropna(subset=["pca_all"]).groupby(["congress", "chamber"]) if len(h) > 10]))
    res[p] = o
R["a_republican_check"] = dict(within_party_correlations=res, rollcall_counts=counts)

# ======================= (b) =======================
if (OUT / "time_panel.csv").exists():
    V = torch.load(cm.KIM_MAP / "results" / "vectors.pt"); vs, ve = V["orth_s"].float().numpy(), V["orth_e"].float().numpy()
    t = np.load(OUT / "time_direction.npy"); Tb = {"cos_t_orth_s": float(t @ vs), "cos_t_orth_e": float(t @ ve),
                                                   "note": "orth_s positive = progressive; y = -projection, so a negative cos(t, orth_s) means later = more traditional"}
    Pp = pd.read_csv(OUT / "time_persons.csv")
    Tb["persons_by_year"] = Pp.groupby(["half", "year"])[["t_proj", "y", "x"]].mean().round(3).reset_index().to_dict("records")
    ho = Pp[Pp.half == "heldout"]; Tb["heldout_corr_tproj_year"] = float(np.corrcoef(ho.t_proj, ho.year)[0, 1]); Tb["heldout_corr_y_year"] = float(np.corrcoef(ho.y, ho.year)[0, 1])
    T = pd.read_csv(OUT / "time_panel.csv").merge(v[["congress", "icpsr", "nokken_poole_dim1"]], on=["congress", "icpsr"], how="left", validate="1:1").rename(columns={"nokken_poole_dim1": "np1"})
    T = T.dropna(subset=["np1"]); T["tenure"] = T.year - T.groupby("icpsr").year.transform("min"); T["rep"] = (T.party == "Republican").astype(int)
    Tb["panel"] = dict(rows=int(len(T)), members=int(T.icpsr.nunique()), corr_tproj_year=float(np.corrcoef(T.t_proj, T.year)[0, 1]),
                       corr_y_int8_kim_y=float(np.corrcoef(T.y_int8, T.y)[0, 1]), corr_y_tproj=float(np.corrcoef(T.y_int8, T.t_proj)[0, 1]),
                       corr_yperp_tproj=float(np.corrcoef(T.y_perp, T.t_proj)[0, 1]))
    for yv in ("y_int8", "y_perp"):
        cols = [yv, "np1", "tenure"]; w = T[cols] - T.groupby("icpsr")[cols].transform("mean"); w["icpsr"] = T.icpsr.values
        r = cm.fit_cr1(w, f"{yv} ~ 0 + np1 + tenure", "icpsr"); Tb[f"within_member_{yv}"] = {k: cm.coef(r, k) for k in ("np1", "tenure")}
        r2 = cm.fit_cr1(w, f"{yv} ~ 0 + np1", "icpsr"); Tb[f"within_member_{yv}_np1_only"] = cm.coef(r2, "np1")
        T[f"z_{yv}"] = T.groupby("era")[yv].transform(lambda s: (s - s.mean()) / s.std())
    T["z_dim1"] = T.groupby("era").dim1.transform(lambda s: (s - s.mean()) / s.std())
    for yv in ("y_int8", "y_perp"):
        r = cm.fit_cr1(T, f"z_{yv} ~ z_dim1 * rep", "icpsr")
        Tb[f"responsiveness_{yv}"] = dict(dem=cm.coef(r, "z_dim1"), rep_minus_dem=cm.coef(r, "z_dim1:rep"))
        Tb[f"within_party_r_{yv}"] = {p: float(np.corrcoef(g[f"z_{yv}"], g.z_dim1)[0, 1]) for p, g in T.groupby("party")}
    pm = T.groupby(["congress", "party"])[["y_int8", "y_perp", "np1"]].mean().reset_index()
    for p, g in pm.groupby("party"):
        g = g.sort_values("congress"); Tb[f"party_levels_corr_{p}"] = {yv: float(np.corrcoef(g[yv], g.np1)[0, 1]) for yv in ("y_int8", "y_perp")}
    R["b_time"] = Tb
(OUT / "rep_time_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")

L = ["# Republican responsiveness check and time direction (generated by 19_rep_and_time.py; exploratory)", "", "## (a) within-party correlation of Kim's coordinates with each yardstick (mean over Congress x chamber cells)"]
for p, o in res.items():
    L.append(f"- {p}: " + ", ".join(f"{k} {v:+.2f}" for k, v in o.items() if v is not None))
L.append("- roll calls per cell (party split / topic subset): " + "; ".join(f"{c['congress']}{c['chamber'][0]} {c['party'][0]} {c['split_rollcalls']}/{c['topic_rollcalls']}" for c in counts))
if "b_time" in R:
    Tb = R["b_time"]; L += ["", "## (b) time direction", f"- cos(t, orth_s) {Tb['cos_t_orth_s']:+.3f}; cos(t, orth_e) {Tb['cos_t_orth_e']:+.3f} ({Tb['note']})",
         f"- held-out persons: corr(t_proj, year) {Tb['heldout_corr_tproj_year']:.2f}; corr(y, year) {Tb['heldout_corr_y_year']:.2f}",
         "- held-out persons by year (t_proj, y, x): " + "; ".join(f"{int(r['year'])}: {r['t_proj']:+.2f}, {r['y']:+.2f}, {r['x']:+.2f}" for r in Tb["persons_by_year"] if r["half"] == "heldout"),
         f"- panel {Tb['panel']}"]
    for yv in ("y_int8", "y_perp"):
        o = Tb[f"within_member_{yv}"]; L.append(f"- within member, {yv}: NP dim1 {f(o['np1'])}; years {f(o['tenure'])}; NP dim1 alone {f(Tb[f'within_member_{yv}_np1_only'])}")
        rr = Tb[f"responsiveness_{yv}"]; L.append(f"  responsiveness (z on z dim1): Democrats {f(rr['dem'])}; Republican minus Democrat {f(rr['rep_minus_dem'])}; within-party r {Tb[f'within_party_r_{yv}']}")
    L.append(f"- party means, level corr with NP dim1: D {Tb.get('party_levels_corr_Democrat')}, R {Tb.get('party_levels_corr_Republican')}")
(OUT / "REP_TIME_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
lab = ["dim1", "np1", "pca_all", "pca_topic"]
for i, (p, c) in enumerate((("Democrat", "tab:blue"), ("Republican", "tab:red"))):
    ax[0].bar(np.arange(4) + (i - .5) * .38, [res[p][f"x~{k}"] or 0 for k in lab], .38, color=c, label=p)
ax[0].set_xticks(range(4), ["DW dim1", "Nokken-Poole", "roll-call PCA\n(all splits)", "roll-call PCA\n(topic votes)"]); ax[0].axhline(0, color="k", lw=.6)
ax[0].set_ylabel("within-party corr with model x"); ax[0].legend(fontsize=8); ax[0].set_title("(a) Is the Republican gap the yardstick?", fontsize=10)
if "b_time" in R:
    ho = Pp[Pp.half == "heldout"].groupby("year")[["y", "t_proj"]].mean()
    ax[1].plot(ho.index, ho.y, "o-", label="social position y (held-out persons)"); ax2 = ax[1].twinx(); ax2.plot(ho.index, ho.t_proj, "s--", color="tab:orange", label="time projection")
    ax[1].axvline(2023, color=".7", ls=":"); ax[1].set_xlabel("year in the prompt"); ax[1].set_ylabel("y (+ = traditional)"); ax2.set_ylabel("time projection")
    ax[1].legend(loc="upper center", fontsize=8); ax2.legend(loc="lower center", fontsize=8); ax[1].set_title("(b) Non-political persons across time", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_rep_time.png", dpi=110)

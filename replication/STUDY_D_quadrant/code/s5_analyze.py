#!/usr/bin/env python
"""S5 analysis (PLAN.md section 4 + addendum A4), CPU. Per model: condition-level C1a, C1b, C2, C3 (x, y, hit), C4; H1-H4 with
mode fixed effects, stratified bootstrap, Holm; per-mode and descriptive tables. -> out/s5/S5_RESULTS.json, s5_conditions_<model>.csv"""
import argparse, json
import numpy as np, pandas as pd
import dcommon as dc

ap = argparse.ArgumentParser()   # A12: defaults reproduce the sealed run; OLMo: --models olmo --out s5_olmo
ap.add_argument("--models", nargs="+", default=["qwen", "llama"], choices=list(dc.MODELS)); ap.add_argument("--out", default="s5")
a = ap.parse_args(); SEL = a.models
items = {it["id"]: it for it in json.load(open(dc.CP_ITEMS, encoding="utf-8"))["items"]}
ALIGNED, CROSS = ["EL-SP", "ER-ST"], ["EL-ST", "ER-SP"]
MIN = {"H1_C1a": 0.05, "H2_C4": 0.05, "H3_C2": -0.05, "H4_C3hit": -0.10}
OUTD = dc.OUT / a.out; OUTD.mkdir(parents=True, exist_ok=True)
NB = 2000

def want(it, prof):
    E, S = prof[1], prof[4]
    e = None if it["econ_yes"] == 0 else ((it["econ_yes"] == 1) == (E == "L"))
    s = None if it["soc_yes"] == 0 else ((it["soc_yes"] == 1) == (S == "P"))
    return e, s

def hbits(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def cond_table(key):
    d = dc.OUT / key / "s5"
    C = pd.read_csv(d / "conditions.csv")[["cid", "mode", "profile", "base"]]
    it = pd.read_csv(d / "items.csv").merge(C, on="cid", validate="many_to_one")
    it["mass"] = it.p_yes_raw + it.p_no_raw; it["p"] = it.p_yes_raw / it.mass; it["H"] = hbits(it.p)
    w = [want(items[i], pr) for i, pr in zip(it.item, it.profile)]
    it["single"] = [(e is None) != (s is None) for e, s in w]
    it["desired_yes"] = [e if s is None else s if e is None else np.nan for e, s in w]
    it["conflict"] = [np.nan if (e is None or s is None) else float(e != s) for e, s in w]
    sg = it[it.single].copy(); sg["p_des"] = np.where(sg.desired_yes.astype(bool), sg.p, 1 - sg.p)
    two = it[~it.single]
    T = C.set_index("cid")
    T["C1a"] = sg.groupby("cid").H.mean(); T["C2"] = sg.groupby("cid").p_des.mean()
    T["C1b"] = two[two.conflict == 1].groupby("cid").H.mean() - two[two.conflict == 0].groupby("cid").H.mean()
    T["mass"] = it.groupby("cid").mass.mean()
    pj = pd.read_csv(d / "proj.csv")
    e, s = pj[pj.axis == "economic"], pj[pj.axis == "social"]
    T["x"] = -(e.groupby("cid").pe.mean() - e.groupby("cid").be.mean()); T["y"] = -(s.groupby("cid").ps.mean() - s.groupby("cid").bs.mean())
    T = T.reset_index()
    cx = T.groupby("mode").x.transform("mean"); cy = T.groupby("mode").y.transform("mean")        # A4: centre of the four profiles in that mode
    T["hit"] = (((T.x - cx) > 0) == T.profile.str.startswith("ER")) & (((T.y - cy) > 0) == T.profile.str.endswith("ST"))
    T["hit"] = T.hit.astype(float)
    g = pd.read_csv(d / "gen.csv"); T = T.merge(g.groupby("cid").ent.mean().rename("C4").reset_index(), on="cid", validate="one_to_one")
    T["cross"] = T.profile.isin(CROSS).astype(float)
    item_stats = it.groupby(["item", "type"]).agg(mass=("mass", "mean"), p_yes=("p", "mean")).reset_index()
    return T, item_stats

def fe_coef(T, col):
    """OLS coefficient of `cross` with mode fixed effects (within-mode demeaning)."""
    y = T[col] - T.groupby("mode")[col].transform("mean"); x = T.cross - T.groupby("mode").cross.transform("mean")
    ok = y.notna(); return float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())

def boot(T, col, seed):
    rng = np.random.default_rng(seed); cells = [g.index.values for _, g in T.groupby(["mode", "profile"])]; est = []
    for _ in range(NB):
        idx = np.concatenate([rng.choice(c, len(c)) for c in cells]); est.append(fe_coef(T.loc[idx], col))
    return np.array(est)

def boot_cluster(T, col, seed):
    """A14 sensitivity (second Codex check): resample bases jointly across profiles within mode (I4: legislators, by ICPSR) and re-estimate the
    mode origin in each draw for the quadrant hit. The registered A4 bootstrap (within mode x profile) is kept as the primary interval."""
    rng = np.random.default_rng(seed); T = T.copy()
    T["cl"] = np.where(T["mode"] == "I4", T.base.str.extract(r"^m(\d+)c", expand=False), T.base)
    groups = {m: {c: g.index.values for c, g in G.groupby("cl")} for m, G in T.groupby("mode")}; est = []
    for _ in range(NB):
        idx = np.concatenate([np.concatenate([gm[k] for k in rng.choice(list(gm), len(gm))]) for gm in groups.values()]); B = T.loc[idx].reset_index(drop=True)
        if col == "hit":
            cx = B.groupby("mode").x.transform("mean"); cy = B.groupby("mode").y.transform("mean")
            B["hit"] = ((((B.x - cx) > 0) == B.profile.str.startswith("ER")) & (((B.y - cy) > 0) == B.profile.str.endswith("ST"))).astype(float)
        est.append(fe_coef(B, col))
    return np.array(est)

def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

RES = {}
for key in [k for k in SEL if (dc.OUT / k / "s5" / "gen.csv").exists()]:   # models whose S5 is complete
    T, IS = cond_table(key); dc.atomic_csv(T, OUTD / f"s5_conditions_{key}.csv"); dc.atomic_csv(IS, OUTD / f"s5_items_{key}.csv")
    R = {"n_conditions": int(len(T)), "by_mode_n": T["mode"].value_counts().to_dict(), "hyp": {}}
    for h, col in [("H1_C1a", "C1a"), ("H2_C4", "C4"), ("H3_C2", "C2"), ("H4_C3hit", "hit")]:
        est = fe_coef(T, col); bs = boot(T, col, dc.seed_for("D-S5", key, h) % 2**32)
        p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        R["hyp"][h] = {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p_boot": p, "min_effect": MIN[h],
                       "aligned_mean": float(T[T.cross == 0][col].mean()), "cross_mean": float(T[T.cross == 1][col].mean())}
    adj = holm(np.array([R["hyp"][h]["p_boot"] for h in R["hyp"]]))
    for h, a in zip(R["hyp"], adj):
        r = R["hyp"][h]; r["p_holm"] = float(a)
        direction_ok = r["est"] >= r["min_effect"] if r["min_effect"] > 0 else r["est"] <= r["min_effect"]
        excl0 = r["ci95"][0] > 0 or r["ci95"][1] < 0
        r["supported"] = bool(excl0 and a < 0.05 and direction_ok)
    R["sensitivity_base_cluster_A14"] = {}
    for h, col in [("H1_C1a", "C1a"), ("H2_C4", "C4"), ("H3_C2", "C2"), ("H4_C3hit", "hit")]:
        bs = boot_cluster(T, col, dc.seed_for("D-S5-A14", key, h) % 2**32)
        R["sensitivity_base_cluster_A14"][h] = {"ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p_boot": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}
    noI2 = T[T["mode"] != "I2"]; R["H4_without_I2"] = {"est": fe_coef(noI2, "hit")}
    R["per_mode"] = {m: {c: {pr: float(g[g.profile == pr][c].mean()) for pr in ALIGNED + CROSS} for c in ["C1a", "C1b", "C2", "hit", "C4", "x", "y", "mass"]}
                     for m, g in T.groupby("mode")}
    R["per_mode_cross_minus_aligned"] = {m: {c: float(g[g.cross == 1][c].mean() - g[g.cross == 0][c].mean()) for c in ["C1a", "C1b", "C2", "hit", "C4"]}
                                         for m, g in T.groupby("mode")}
    Z = T.copy(); Z["zx"] = (Z.x - Z.x.mean()) / Z.x.std(); Z["zy"] = (Z.y - Z.y.mean()) / Z.y.std()
    mm = Z.groupby(["profile", "mode"])[["zx", "zy"]].mean()
    R["C5_sd_across_modes"] = {pr: {"zx": float(mm.loc[pr].zx.std()), "zy": float(mm.loc[pr].zy.std())} for pr in ALIGNED + CROSS}
    R["C1b_overall"] = {pr: float(T[T.profile == pr].C1b.mean()) for pr in ALIGNED + CROSS}
    R["low_mass_items"] = IS[IS.mass < 0.5].item.tolist()
    RES[key] = R
    dc.log(key, {h: (round(v["est"], 3), [round(c, 3) for c in v["ci95"]], round(v["p_holm"], 3), v["supported"]) for h, v in R["hyp"].items()})
dc.atomic_json(RES, OUTD / "S5_RESULTS.json")

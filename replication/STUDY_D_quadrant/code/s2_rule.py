#!/usr/bin/env python
"""S2 rule (PLAN.md): layer score = max over alphas of the mean judge score (None dropped; all None -> 0, as notebook 01 cell 9);
common layer = min rank-sum (notebook 02 cell 10: pandas rank descending, idxmin = first minimum). Then Kim's Löwdin code
(congress_map/01_extract_vectors.py, copied) at the rule layer -> out/<model>/vectors.pt; for Qwen also vectors_L17.pt (B3).
Usage: python s2_rule.py --model qwen [--smoke]"""
import argparse, json
import numpy as np, pandas as pd, torch
import dcommon as dc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
md = dc.model_dir(a.model); s2 = md / ("s2_smoke" if a.smoke else "s2"); s1 = md / ("s1_smoke" if a.smoke else "s1")
diffs = torch.load(s1 / "diffs_all.pt")["diffs"]; NL = diffs["economic"].shape[0]
res = {}
for ax in ["economic", "social"]:
    per_layer, allres = {}, {}
    for f in sorted(s2.glob(f"score_{ax}_L*.json")):
        r = json.load(open(f, encoding="utf-8")); L = int(r["layer"])
        df = pd.DataFrame({"alpha": r["alphas"], "s": [np.nan if v is None else v for v in r["scores"]]})
        m = df.groupby("alpha").s.mean().fillna(0.0)
        allres[L] = {float(k): float(v) for k, v in m.items()}; per_layer[L] = float(m.max())
    res[ax] = {"layer_scores": per_layer, "all_results": allres, "best_layer": max(per_layer, key=per_layer.get)}
layers = sorted(set(res["economic"]["layer_scores"]) & set(res["social"]["layer_scores"]))
if not a.smoke:
    assert layers == list(range(1, NL)), f"missing layers: {sorted(set(range(1, NL)) - set(layers))}"
se = pd.Series({l: res["economic"]["layer_scores"][l] for l in layers}); ss = pd.Series({l: res["social"]["layer_scores"][l] for l in layers})
rank_sum = se.rank(ascending=False) + ss.rank(ascending=False)
LR = int(rank_sum.idxmin())
J1 = {ax: json.load(open(s2 / f"J1_{ax}.json")) for ax in ["economic", "social"]}
out = {"model": dc.MODELS[a.model], "common_layer": LR, "axis_best": {ax: res[ax]["best_layer"] for ax in res},
       "rank_sum": {int(k): float(v) for k, v in rank_sum.items()}, "econ_scores": se.to_dict(), "social_scores": ss.to_dict(),
       "all_results": {ax: res[ax]["all_results"] for ax in res}, "selection_rule": "min rank-sum of per-layer max steering scores (notebook 02 cell 10)",
       "judge": dc.MODELS[dc.JUDGE] + " FP8 (local; GPT-4.1-mini not used without approval)",
       "J1": {ax: {k: J1[ax][k] for k in ["auc", "pass", "mean_pos", "mean_neg", "n_missing"]} for ax in J1}}
if a.model == "qwen":
    out["rule_returns_17"] = LR == 17

def lowdin(L):
    """Kim's step-1 code, verbatim apart from variable names."""
    ve, vs = diffs["economic"], diffs["social"]
    ue, us = ve[L] / ve[L].norm(), vs[L] / vs[L].norm()
    V = torch.stack([ue, us], 1); w, U = torch.linalg.eigh(V.T @ V)
    Q = V @ (U @ torch.diag(w.rsqrt()) @ U.T); oe, os_ = Q[:, 0], Q[:, 1]
    if torch.dot(oe, ue) < 0: oe = -oe
    if torch.dot(os_, us) < 0: os_ = -os_
    oe, os_ = oe / oe.norm(), os_ / os_.norm()
    return {"layer": L, "raw_e": ue, "raw_s": us, "orth_e": oe, "orth_s": os_, "model": dc.MODELS[a.model]}, float(torch.dot(ue, us))

vdir = md / ("vec_smoke" if a.smoke else ""); vdir.mkdir(parents=True, exist_ok=True)
V, c = lowdin(LR); torch.save(V, vdir / "vectors.pt"); out["cos_raw_at_rule_layer"] = c
if a.model == "qwen" and LR != 17:
    V17, c17 = lowdin(17); torch.save(V17, vdir / "vectors_L17.pt"); out["cos_raw_at_17"] = c17
dc.atomic_json(out, vdir / "rule.json")
dc.log(f"{a.model}: common layer {LR} (econ best {out['axis_best']['economic']}, social best {out['axis_best']['social']}); "
       f"J1 AUC econ {out['J1']['economic']['auc']:.3f} social {out['J1']['social']['auc']:.3f}; cos(raw)@L {c:.3f}")

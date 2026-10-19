#!/usr/bin/env python
"""S14 (third Codex check): no-persona arm, reversed minus party-like A/B entropy at NONE, from unrounded L unit means (the round-2 JSON
subtracted rounded cell means). Choice entropy does not use the lens, so the value does not depend on R1. -> no_persona_package_effect.json

Addendum 2026-10-18 (author: "4막 구간 계산해"): intervals for the same estimand (mean over the 54 topic pairs of reversed - party-like).
The 54 pairs cross 6 economic x 9 social items, so pairs sharing a passage are not independent. Declared before running:
  primary   = pigeonhole two-way bootstrap (resample economic and social items independently with replacement; every pair in the product,
              with multiplicities), percentile 95%, B = 10,000, fixed seed
  reported  = pair bootstrap (iid over 54 pairs), one-way economic-item and social-item cluster bootstraps
  context   = the persona arm (Democratic and Republican personas pooled, then per party) with the same pair-level estimator: persona-arm topic
              means of reversed - party-like, same pigeonhole interval. Context only; the persona - no-persona difference is not tested.
--synth runs a synthetic check (known effect, crossed item effects) before the real run."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parents[2]; B = 10_000; SEED = 20261018

def boot(D, kind, rng):
    """D: econ x social matrix of pair differences (NaN-free). Returns B bootstrap means."""
    ne, ns = D.shape; out = np.empty(B)
    for b in range(B):
        if kind == "pigeonhole": out[b] = D[np.ix_(rng.integers(0, ne, ne), rng.integers(0, ns, ns))].mean()
        elif kind == "pair": out[b] = D.ravel()[rng.integers(0, ne * ns, ne * ns)].mean()
        elif kind == "econ": out[b] = D[rng.integers(0, ne, ne), :].mean()
        elif kind == "social": out[b] = D[:, rng.integers(0, ns, ns)].mean()
    return out

def tag_id(t): return sum(ord(c) * (i + 1) for i, c in enumerate(t))   # stable across runs (no PYTHONHASHSEED dependence)

def summarize(D, tag):
    r = {"est": float(D.mean()), "n_pairs": int(D.size), "n_econ": int(D.shape[0]), "n_social": int(D.shape[1])}
    for i, kind in enumerate(("pigeonhole", "pair", "econ", "social")):
        bs = boot(D, kind, np.random.default_rng([SEED, i, tag_id(tag)]))
        r[f"ci95_{kind}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return r

def matrix(u):
    w = u.pivot_table(index="topic", columns="family", values="final_H", aggfunc="mean")
    d = (w["reversed"] - w["party_like"]).rename("d").reset_index()
    d[["e", "s"]] = d.topic.str.split("|", expand=True)
    M = d.pivot(index="e", columns="s", values="d")
    assert M.shape == (6, 9) and not M.isna().any().any(), M.shape
    return M.values

if "--synth" in sys.argv:
    rng = np.random.default_rng(1); cover = {k: 0 for k in ("pigeonhole", "pair")}; R = 200; B = 1000
    for r in range(R):
        D = 0.2 + rng.normal(0, 0.15, (6, 1)) + rng.normal(0, 0.15, (1, 9)) + rng.normal(0, 0.1, (6, 9))
        for i, kind in enumerate(cover):
            bs = boot(D, kind, np.random.default_rng([r, i])); lo, hi = np.percentile(bs, [2.5, 97.5]); cover[kind] += lo <= 0.2 <= hi
    print("synthetic coverage of true 0.2 (200 reps, crossed item effects):", {k: v / R for k, v in cover.items()})
    sys.exit()

R = {}
for m in ["qwen", "llama", "olmo"]:
    U = pd.read_csv(K / f"STUDY_L_battery/out/{m}/tables/units.csv")
    U = U[(U.cond == "NONE") & U.family.isin(["party_like", "reversed"])]
    u = U[U.party == "none"]
    g = u.groupby("family").final_H.mean()
    R[m] = {"party_like": float(g["party_like"]), "reversed": float(g["reversed"]), "reversed_minus_party_like": float(g["reversed"] - g["party_like"]),
            "n_units": u.groupby("family").size().to_dict()}
    R[m]["interval_2026_10_18"] = summarize(matrix(u), f"{m}|none")
    R[m]["context_persona_arm_2026_10_18"] = {p: summarize(matrix(U[U.party.isin(ps)]), f"{m}|{p}")
                                              for p, ps in (("both", ["Democrat", "Republican"]), ("Democrat", ["Democrat"]), ("Republican", ["Republican"]))}
R["_definition_2026_10_18"] = __doc__.split("Addendum", 1)[1].strip()
json.dump(R, open(Path(__file__).parent / "no_persona_package_effect.json", "w"), indent=1)
for m in ["qwen", "llama", "olmo"]:
    i = R[m]["interval_2026_10_18"]; c = R[m]["context_persona_arm_2026_10_18"]["both"]
    print(f"{m}: no persona {i['est']:+.3f} pigeonhole [{i['ci95_pigeonhole'][0]:+.3f}, {i['ci95_pigeonhole'][1]:+.3f}] pair [{i['ci95_pair'][0]:+.3f}, "
          f"{i['ci95_pair'][1]:+.3f}] econ [{i['ci95_econ'][0]:+.3f}, {i['ci95_econ'][1]:+.3f}] social [{i['ci95_social'][0]:+.3f}, {i['ci95_social'][1]:+.3f}]"
          f" | persona arm {c['est']:+.3f} pigeonhole [{c['ci95_pigeonhole'][0]:+.3f}, {c['ci95_pigeonhole'][1]:+.3f}]")

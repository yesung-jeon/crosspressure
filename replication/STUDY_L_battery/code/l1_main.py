#!/usr/bin/env python
"""l1 (PLAN v0.1 sections 2-5): one model. 20 personas + the no-persona arm x 37 conditions x (108 bundled + 15 single-issue items) x 2
orders with logit-lens readout; map check (Kim questions 20-24 at L+4) for non-placebo conditions and two placebo seeds. mu = mean
hidden norm at the layer over the first persona's NONE prompts. Parts of 10 personas (5 D + 5 R), resumable; one persona at a time.
-> out/<model>/main/part_XX.csv.gz, mapcheck.csv, pushes.json. Usage: python l1_main.py --model qwen [--smoke]"""
import argparse
import numpy as np, pandas as pd, torch
import l_common as lc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(lc.MODELS)); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
od = lc.out_dir(a.model, *(["smoke"] if a.smoke else []), "main")
L = lc.layer(a.model); CONDS = lc.conditions()
P = lc.personas("confirm"); D_, R_ = [p for p in P if p["party"] == "Democrat"], [p for p in P if p["party"] == "Republican"]
parts = [D_[i:i + 5] + R_[i:i + 5] for i in range(0, lc.N_PER_PARTY, 5)] + [[dict(lc.NOPERSONA)]]
bundles, singles = lc.items(); ITEMS = bundles + singles
if a.smoke:
    parts = [[D_[0], R_[0]], [dict(lc.NOPERSONA)]]
    ITEMS = [b for b in bundles if b["econ"] == "health"] + singles
    CONDS = [c for c in CONDS if c["name"] in ("NONE", "Dp_H", "Dm_H", "Sm_H", "Ep_H", "PLAC_H_0", "PLAC_H_1")]
tok, model = lc.load(a.model); N = model.config.num_hidden_layers
ia, ib = lc.fc.ab_ids(tok)
oe, os_ = lc.directions(a.model, L); oe4, os4 = lc.directions(a.model, L + 4)
mu_texts = [lc.chat(tok, lc.fc.choice_msgs(P[0]["system"], b["pkgL"], b["pkgR"])) for b in ITEMS[:64]]
mu = lc.gc.mu_at(model, tok, mu_texts, L)
PU = lc.Pusher(mu, oe, os_, lc.norms(a.model)); info = PU.info()
lc.atomic_json(dict(info, model=lc.MODELS[a.model], layer=L, n_layers=N, ab_ids=[int(ia), int(ib)], conditions=[c["name"] for c in CONDS], n_items=len(ITEMS),
                    personas=[p["pid"] for part in parts for p in part]), od / "pushes.json")
lc.log(f"{a.model}: layer {L}/{N}, mu {mu:.2f}, norms H {info['norm_H']:.2f} L {info['norm_L']:.2f}; cos(e,s) {info['cos_e_s']:+.4f}; max |cos| between placebos {info['cos_plac_pairs_max']:.3f}; "
       f"{len(CONDS)} conditions, {len(ITEMS)} items, {sum(len(p) for p in parts)} personas")

def metrics(m):
    dl = lc.decision_layer(m); s = np.sign(m[L:]); flips = int((s[1:] != s[:-1]).sum())
    return dl, flips, float(lc.hbits(1 / (1 + np.exp(-np.clip(m[L:], -50, 50)))).mean())

with lc.Steer(model, L) as steer:
    for pi, part in enumerate(parts):
        f = od / f"part_{pi:02d}.csv.gz"
        if f.exists(): lc.log(f"{f.name} done, skipped"); continue
        lc.fc.ram_guard(); frames = []
        for p in part:
            rows, texts, W = [], [], []
            for c in CONDS:
                w = PU.w(c)
                for b in ITEMS:
                    for order in (0, 1):
                        A, B = (b["pkgL"], b["pkgR"]) if order == 0 else (b["pkgR"], b["pkgL"])
                        rows.append(dict(pid=p["pid"], party=p["party"], cond=c["name"], kind=c["kind"], strength=c["strength"], vec=c["vec"],
                                         topic=b["topic"], family=b["family"], kang=b["kang"], origin=b["origin"], order=order))
                        texts.append(lc.chat(tok, lc.fc.choice_msgs(p["system"], A, B))); W.append(w)
            M, PA, PB = lc.lens_run(tok, model, steer, texts, torch.stack(W).float())
            D = pd.DataFrame(rows); mt = [metrics(m) for m in M]
            D["DL"], D["flips"], D["lens_H"] = [x[0] for x in mt], [x[1] for x in mt], [x[2] for x in mt]
            D["pA"], D["pB"], D["m_final"] = PA, PB, M[:, -1]
            frames.append(D); lc.log(f"  {p['pid']}: {len(D)} prompts, mass {float((D.pA + D.pB).mean()):.3f}")
        D = pd.concat(frames, ignore_index=True); lc.atomic_csv(D, f); lc.log(f"{f.name}: {len(D)} prompts ({D.party.value_counts().to_dict()})")

    mf = od / "mapcheck.csv"
    if not mf.exists():
        qs = lc.fc.kim_questions(); rows, texts, W = [], [], []
        for p in [x for part in parts for x in part]:
            for c in CONDS:
                if c["kind"] == "plac" and c["plac"] > 1: continue
                w = PU.w(c)
                for j, q in enumerate(qs):
                    rows.append(dict(pid=p["pid"], party=p["party"], cond=c["name"], kind=c["kind"], strength=c["strength"], vec=c["vec"], q=j, qaxis=q["axis"]))
                    msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + [{"role": "user", "content": q["question"]}]
                    texts.append(lc.chat(tok, msgs)); W.append(w)
        E, S = lc.project_run(tok, model, steer, texts, torch.stack(W).float(), L + 4, oe4, os4)
        lc.atomic_csv(pd.DataFrame(rows).assign(proj_e=E, proj_s=S), mf); lc.log(f"map check saved ({len(rows)} prompts)")
lc.log("l1 complete")

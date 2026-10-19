#!/usr/bin/env python
"""S2 judge (PLAN.md): local judge (Llama-3.1-8B-Instruct FP8) with the axis eval prompt of data_{axis}.json as a user message;
score = Betley aggregation of the next-token distribution (top 20 tokens; tokens that parse as integers 0-100; mass < 0.25 -> None).
Judges every S2 generation file of the given models (score_<axis>_L<layer>.json beside it) and the J1 validity set
(200 kept pos + 200 kept neg S1 extraction answers per axis, seeded) -> out/<model>/s2/J1_<axis>.json.
Usage: python s2_judge.py --models qwen llama [--smoke]"""
import argparse, json, math, sys
import numpy as np, pandas as pd, torch
import dcommon as dc

ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
sub2, sub1 = ("s2_smoke", "s1_smoke") if a.smoke else ("s2", "s1")
EP = {ax: dc.axis_data(ax)["eval prompt"] for ax in ["economic", "social"]}

jobs = []   # (outfile, axis, questions, answers, extra)
for m in a.models:
    d2 = dc.model_dir(m) / sub2
    for f in sorted(d2.glob("gen_*_L*.json")):
        of = d2 / f.name.replace("gen_", "score_")
        if not of.exists():
            g = json.load(open(f, encoding="utf-8")); jobs.append((of, g["axis"], g["questions"], g["answers"], {"layer": g["layer"], "alphas": g["alphas"]}))
    for ax in ["economic", "social"]:
        of = d2 / f"J1_{ax}.json"
        if of.exists():
            continue
        p = pd.read_csv(dc.model_dir(m) / sub1 / f"extract_{ax}_pairs.csv"); p = p[p.keep]
        n = min(200, len(p)); s = p.sample(n=n, random_state=dc.seed_for("D-J1", m, ax) % 2**32)
        qs = dc.axis_data(ax)["questions"]
        Q = [qs[i] for i in s.q_idx] * 2; A = s.response_pos.fillna("").tolist() + s.response_neg.fillna("").tolist()
        jobs.append((of, ax, Q, A, {"label": [1] * n + [0] * n}))
if not jobs:
    dc.log("judge: nothing to do"); sys.exit(0)
dc.log(f"judge jobs: {len(jobs)}")
tok, model = dc.load(dc.JUDGE)
vocab_int = {}
for t in range(len(tok)):
    s = tok.decode([t])
    try:
        v = int(s)
    except ValueError:
        continue
    if 0 <= v <= 100:
        vocab_int[t] = v
dc.log(f"integer tokens 0-100 in judge vocabulary: {len(vocab_int)}")

def betley(lg):
    p = torch.softmax(lg, -1); top = torch.topk(p, 20)
    tot = s = 0.0
    for pr, t in zip(top.values.tolist(), top.indices.tolist()):
        if t in vocab_int:
            tot += pr; s += pr * vocab_int[t]
    return None if tot < 0.25 else s / tot

for of, ax, Q, A, extra in jobs:
    prompts = [dc.chat(tok, [{"role": "user", "content": EP[ax].replace("{{question}}", q).replace("{{answer}}", str(ans))}]) for q, ans in zip(Q, A)]
    b = a.batch
    while True:
        try:
            lg = dc.last_logits(tok, model, prompts, batch=b); break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); b = max(1, b // 2); dc.log(f"OOM -> judge batch {b}")
    scores = [betley(lg[i]) for i in range(len(prompts))]
    rec = {"axis": ax, "scores": scores, **extra}
    if "label" in extra:
        from sklearn.metrics import roc_auc_score
        y = np.array(extra["label"]); sc = np.array([np.nan if v is None else v for v in scores]); ok = ~np.isnan(sc)
        rec["auc"] = float(roc_auc_score(y[ok], sc[ok])); rec["n_missing"] = int((~ok).sum()); rec["pass"] = rec["auc"] >= 0.90
        rec["mean_pos"], rec["mean_neg"] = float(np.nanmean(sc[y == 1])), float(np.nanmean(sc[y == 0]))
        dc.log(f"J1 {of.parent.parent.name} {ax}: AUC {rec['auc']:.3f} pos {rec['mean_pos']:.1f} neg {rec['mean_neg']:.1f} missing {rec['n_missing']}")
    dc.atomic_json(rec, of)
    dc.log(f"judged {of.parent.parent.name}/{of.name}: {len(scores)} (missing {sum(v is None for v in scores)})")
dc.log("judge complete")

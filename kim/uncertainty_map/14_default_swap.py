#!/usr/bin/env python
"""Step 14 - analysis 1: does the asymmetry follow the model's default (anchor account, H2) or stay put (training account, H1)?

Personas: the T1 personas in one wording (25 stance cells, wording "es|an American") + the 36 public figures = 61.
Defaults placed BEFORE the persona in the system turn, each followed by "In this conversation you will role-play the
following person." :
  D_qwen  "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."   (the usual origin, now kept)
  D_prog  "You are a helpful assistant with strongly progressive political values."
  D_cons  "You are a helpful assistant with strongly conservative political values."
  D_hist  "You are a neutral historian who reconstructs people's views faithfully, without judging them."
The persona-only condition (default replaced) is step 7. Each default alone is also projected (where is the anchor?).
Predictions: H2 (anchor) - under D_cons the entropy slope over persona position flattens or reverses (conservative personas
become the confident ones); H1 (training) - the slope keeps its sign under every default.
Measures: Kim's projection of each default and of default + persona; entropy on Kim's 10 questions (1 sample).
Output: out/swap_personas.csv, out/swap_generations.csv
Usage: python 14_default_swap.py [--out out] [--quant int8]
"""
import argparse
import pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=48); args = ap.parse_args()
OUT = cm.HERE / args.out
DEFAULTS = {"D_qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            "D_prog": "You are a helpful assistant with strongly progressive political values.",
            "D_cons": "You are a helpful assistant with strongly conservative political values.",
            "D_hist": "You are a neutral historian who reconstructs people's views faithfully, without judging them."}
RP = " In this conversation you will role-play the following person. "
T = pd.read_csv(OUT / "t1_personas.csv")
T = T[(T.set == "P") | ((T.set == "S") & (T.wording == "es|an American"))]
assert len(T) == 61, len(T)
rows = [dict(pid=f"{d}__alone", default=d, persona_pid=None, system=txt) for d, txt in DEFAULTS.items()]
rows += [dict(pid=f"{d}__{r.pid}", default=d, persona_pid=r.pid, system=txt + RP + r.system) for d, txt in DEFAULTS.items() for _, r in T.iterrows()]
P = pd.DataFrame(rows); assert P.pid.is_unique

tok, model, V = cm.load(args.quant)
pp, gp = OUT / "swap_personas.csv", OUT / "swap_generations.csv"
if not pp.exists():
    P["x"], P["y"] = cm.persona_scores(tok, model, V, list(P.system)); P.to_csv(pp, index=False); cm.log(f"projections: {len(P)}")
P = pd.read_csv(pp); qs = cm.kim_questions()
jobs = [dict(pid=r.pid, default=r.default, persona_pid=r.persona_pid, q=qi, axis=q["axis"], prompt=cm.chat(tok, q["question"] + cm.SUFFIX, r.system))
        for _, r in P.iterrows() for qi, q in enumerate(qs)]
cm.generate_entropy(tok, model, jobs, gp, "step14", batch=args.batch)
cm.log("saved", gp); (OUT / "step14.done").touch()

#!/usr/bin/env python
"""Step 23 - (#2) does the temporal distortion reach the generated answers? (#3) which questions drive the economic drift?

A  Judge validation set. Qwen2.5-7B (int8, the generator) answers Kim's 10 held-out questions under the repo's 5 paired
   instructions per axis (economic_left.json: pos = government / redistribution; social_progressive.json: pos = progressive),
   1 sample each, 64 tokens, the notebook's 100-word suffix -> 100 answers with a known side.
B  Per-question projections for #3: the 30 named non-political persons of step 9 at 1900, 1955, 1985, 2023 (neutral ending),
   Kim's projection on each held-out question separately (no averaging), int8 generator.
C  Judge: Qwen2.5-3B-Instruct (bf16), a different model from the generator (Kim C7). It sees only the question and the answer
   (no persona, no year; Kim's blinding). Forced choice by next-token probability (P2 judge style):
     economic: A = "more government involvement / redistribution", B = "more market / individual responsibility"
     social:   A = "progressive / changing traditional norms",     B = "traditional / preserving established norms"
   score_B = P("B") / (P("A") + P("B")) = how right / traditional the answer is (0-1); P(A)+P(B) kept as format mass.
   Scored: the validation set (A), the year-swap and switcher answers (step 2), the persona-offset answers (step 9, incl.
   ordinary persons in 2023 vs 1955), and every legislator answer (step 1).
Output: out/judge_validation.csv, out/ordinary_by_question.csv, out/stance_<source>.csv
Usage: python 23_output_stance.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd, torch
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--batch", type=int, default=32); args = ap.parse_args()
OUT = cm.HERE / args.out; qs = cm.kim_questions()
QTEXT = {i: q["question"] for i, q in enumerate(qs)}; QAXIS = {i: q["axis"] for i, q in enumerate(qs)}

# ---------------- A + B with the generator (int8 Qwen2.5-7B) ----------------
fa, fb = OUT / "judge_validation_texts.csv", OUT / "ordinary_by_question.csv"
if not (fa.exists() and fb.exists()):
    tok, model, V = cm.load("int8")
    if not fa.exists():
        src = cm.REPO / "data_generation" / "trait_data_extract"; jobs = []
        for axis, fn in (("economic", "economic_left.json"), ("social", "social_progressive.json")):
            instr = json.load(open(src / fn))["instructions"]
            for qi in [i for i in QTEXT if QAXIS[i] == axis]:
                for k, pair in enumerate(instr):
                    for side in ("pos", "neg"):
                        jobs.append(dict(axis=axis, q=qi, instr=k, side=side, prompt=cm.chat(tok, QTEXT[qi] + cm.SUFFIX, pair[side])))
        cm.generate_entropy(tok, model, jobs, fa, "step23-valid", batch=args.batch)
    if not fb.exists():
        OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
               "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
        NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
                 "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
                 "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
                 "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
        persons = [(NAMES[k], o, s) for k, (o, s) in enumerate([(o, s) for o in OCC for s in ["Ohio", "California", "Georgia"]])]
        rows = [dict(person=k, year=y) for k in range(30) for y in (1900, 1955, 1985, 2023)]
        systems = [f"You are {persons[r['person']][0]}, who works as {persons[r['person']][1]} in {persons[r['person']][2]} in {r['year']}. "
                   f"Answer the question in the first person, as {persons[r['person']][0]} would." for r in rows]
        base = cm.project(tok, model, V, [cm.chat(tok, QTEXT[i]) for i in range(10)])
        P = cm.project(tok, model, V, [cm.chat(tok, QTEXT[i], s) for s in systems for i in range(10)]).reshape(len(rows), 10, 2)
        out = [dict(person=r["person"], year=r["year"], q=i, axis=QAXIS[i], x=float(-(P[j, i, 0] - base[i, 0])), y=float(-(P[j, i, 1] - base[i, 1])))
               for j, r in enumerate(rows) for i in range(10)]
        pd.DataFrame(out).to_csv(fb, index=False); cm.log("per-question projections saved")
    del model; torch.cuda.empty_cache()

# ---------------- C judge (Qwen2.5-3B bf16) ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
jt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct"); jt.padding_side = "left"
jm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16, device_map={"": 0}).eval()
A_ID, B_ID = jt.encode("A", add_special_tokens=False)[0], jt.encode("B", add_special_tokens=False)[0]
OPT = {"economic": ("more government involvement, public provision or redistribution", "more reliance on markets, private provision or individual responsibility"),
       "social": ("progressive positions that change or relax traditional norms", "traditional positions that preserve established norms")}
def jprompt(q, a, axis):
    oa, ob = OPT[axis]
    u = (f"Question: {QTEXT[q]}\n\nAnswer: {str(a).strip()}\n\nWhich position does this answer take on the question?\n"
         f"A. {oa}\nB. {ob}\nReply with one letter: A or B.")
    return jt.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
@torch.inference_mode()
def judge(df, name):
    f = OUT / f"stance_{name}.csv"
    if f.exists(): return pd.read_csv(f)
    P = [jprompt(q, a, QAXIS[q]) for q, a in zip(df.q, df.text)]
    lens = np.array([len(jt(p, add_special_tokens=False)["input_ids"]) for p in P]); order = np.argsort(lens)
    sb, mass = np.zeros(len(P)), np.zeros(len(P))
    for b0 in range(0, len(P), args.batch):
        idx = order[b0:b0 + args.batch]
        enc = jt([P[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(jm.device)
        pr = torch.softmax(jm(**enc).logits[:, -1, :].float(), -1); a, b = pr[:, A_ID], pr[:, B_ID]
        sb[idx] = (b / (a + b)).cpu().numpy(); mass[idx] = (a + b).cpu().numpy()
    out = df.drop(columns=[c for c in ("prompt", "text") if c in df.columns]).copy(); out["score_B"], out["format_mass"] = sb, mass
    out.to_csv(f, index=False); cm.log(f"judged {name}: {len(out)}"); return out
v = pd.read_csv(fa); judge(v, "validation")
judge(pd.read_csv(OUT / "counterfactual_generations.csv"), "counterfactual")
judge(pd.read_csv(OUT / "offset_generations.csv"), "offset")
judge(pd.read_csv(OUT / "member_generations.csv"), "members")
(OUT / "step23.done").touch(); cm.log("step 23 done")

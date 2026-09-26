#!/usr/bin/env python
"""Step 1 - extract the economic and social ideology vectors.

For each axis, the model answers the repo's 20 extraction questions under the five paired system
instructions (pos = economic-left / social-progressive, neg = economic-right / social-traditional),
`--rollouts` times each. Hidden states of the generated answers are averaged over response tokens
("response_avg") and the vector is mean(pos) - mean(neg), per layer. At `--layer` the two unit vectors
are then symmetrically (Lowdin) orthogonalized so that the economic and social directions are at 90 degrees.

Outputs (in --out):
  vectors.pt        {layer, raw_e, raw_s, orth_e, orth_s}  unit vectors (float32, hidden_size)
  vectors_info.json cosine(raw_e, raw_s) per layer, alignment of orth with raw
  extract_<axis>_pairs.csv  the generated pos/neg answers
"""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--repo_root", default="..", help="path to the crosspressure repo (needs data_generation/trait_data_extract)")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--rollouts", type=int, default=5)
ap.add_argument("--max_new", type=int, default=256)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default="out")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(args.repo_root) / "data_generation" / "trait_data_extract"
device = torch.device("cuda")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def seed_for(*p): return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
EOS = model.generation_config.eos_token_id; EOS = [EOS] if isinstance(EOS, int) else list(EOS)
NL = model.config.num_hidden_layers + 1
log("loaded", args.model, "layers", NL - 1)

def chat(q, system=None):
    m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def generate(prompts, seed):
    texts, ntok = [], []
    for b0 in range(0, len(prompts), args.batch):
        enc = tok(prompts[b0:b0 + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + b0)
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=EOS)
        gen = out[:, enc["input_ids"].shape[1]:]
        is_eos = torch.zeros_like(gen, dtype=torch.bool)
        for e in EOS: is_eos |= gen.eq(e)
        n = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((gen.shape[0],), gen.shape[1], device=gen.device))
        texts += tok.batch_decode(gen, skip_special_tokens=True); ntok += n.tolist()
    return texts, ntok

@torch.inference_mode()
def response_avg_hidden(prompts, responses, batch=16):
    """per-layer mean hidden state over the response tokens -> (NL, N, hidden)"""
    acc = [[] for _ in range(NL)]
    tok.padding_side = "right"
    for b0 in range(0, len(prompts), batch):
        P, R = prompts[b0:b0 + batch], responses[b0:b0 + batch]
        enc = tok([p + r for p, r in zip(P, R)], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = torch.tensor([len(tok.encode(p, add_special_tokens=False)) for p in P], device=device)
        tlen = enc["attention_mask"].sum(1); pos = torch.arange(enc["input_ids"].shape[1], device=device)[None, :]
        m = ((pos >= plen[:, None]) & (pos < tlen[:, None])).float()
        empty = m.sum(1) == 0
        if empty.any(): m[empty, :] = 0; m[empty, (plen[empty] - 1).clamp_min(0)] = 1
        o = model(**enc, output_hidden_states=True)
        for l in range(NL):
            acc[l].append(((o.hidden_states[l].float() * m[:, :, None]).sum(1) / m.sum(1, keepdim=True)).cpu())
    tok.padding_side = "left"
    return torch.stack([torch.cat(a, 0) for a in acc], 0)

REFUSAL = ("i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai", "i'm unable", "i am unable")
diffs = {}
for axis in ["economic", "social"]:
    d = json.load(open(DATA / f"data_{axis}.json"))
    instr = d.get("instructions", d.get("instruction")); qs = d["questions"][:20]
    gens = {}
    for side in ["pos", "neg"]:
        jobs = [(k, qi, r, pair[side], q) for qi, q in enumerate(qs) for k, pair in enumerate(instr) for r in range(args.rollouts)]
        prompts = [chat(q, s) for (_, _, _, s, q) in jobs]
        t0 = time.time(); texts, ntok = generate(prompts, seed_for("extract", axis, side))
        gens[side] = pd.DataFrame(dict(instr_idx=[j[0] for j in jobs], q_idx=[j[1] for j in jobs], rollout=[j[2] for j in jobs],
                                       prompt=prompts, response=texts, n_tokens=ntok))
        log(f"{axis}/{side}: {len(prompts)} answers in {time.time()-t0:.0f}s")
    paired = gens["pos"].merge(gens["neg"], on=["instr_idx", "q_idx", "rollout"], suffixes=("_pos", "_neg"))
    ok = lambda t, n: n >= 15 and not str(t).strip().lower().startswith(REFUSAL)
    paired["keep"] = [ok(a, na) and ok(b, nb) for a, na, b, nb in zip(paired.response_pos, paired.n_tokens_pos, paired.response_neg, paired.n_tokens_neg)]
    paired.to_csv(OUT / f"extract_{axis}_pairs.csv", index=False)
    sel = paired[paired.keep]; log(f"{axis}: kept {len(sel)}/{len(paired)} pairs")
    Hp = response_avg_hidden(sel.prompt_pos.tolist(), sel.response_pos.tolist())
    Hn = response_avg_hidden(sel.prompt_neg.tolist(), sel.response_neg.tolist())
    diffs[axis] = Hp.mean(1) - Hn.mean(1)          # (NL, hidden)

ve, vs = diffs["economic"], diffs["social"]; L = args.layer
cos_curve = [float(torch.nn.functional.cosine_similarity(ve[l][None], vs[l][None])) for l in range(NL)]
ue, us = ve[L] / ve[L].norm(), vs[L] / vs[L].norm()
V = torch.stack([ue, us], 1); w, U = torch.linalg.eigh(V.T @ V)
Q = V @ (U @ torch.diag(w.rsqrt()) @ U.T); oe, os_ = Q[:, 0], Q[:, 1]
if torch.dot(oe, ue) < 0: oe = -oe
if torch.dot(os_, us) < 0: os_ = -os_
oe, os_ = oe / oe.norm(), os_ / os_.norm()
torch.save({"layer": L, "raw_e": ue, "raw_s": us, "orth_e": oe, "orth_s": os_, "model": args.model}, OUT / "vectors.pt")
json.dump({"layer": L, "cos_raw_at_layer": float(torch.dot(ue, us)), "align_e": float(torch.dot(oe, ue)), "align_s": float(torch.dot(os_, us)),
           "cos_curve": cos_curve}, open(OUT / "vectors_info.json", "w"), indent=1)
log(f"saved {OUT/'vectors.pt'}  cos(raw_e, raw_s)@L{L} = {float(torch.dot(ue, us)):.3f}")

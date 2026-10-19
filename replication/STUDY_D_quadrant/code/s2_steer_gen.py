#!/usr/bin/env python
"""S2 generation (PLAN.md): the user's layer search (notebook 01 cell 9) on the S1 raw differences.
For each axis and layer 1..N: unit vector, last-position hook h[:, -1] += a * ||h[:, -1]|| * v with a per batch row, alphas
0.05-0.60 (10), held-out questions 20-29, greedy, 256 new tokens, notebook tokenisation (chat template, add_special_tokens=False,
eos = tokenizer.eos_token_id). Writes out/<model>/s2/gen_<axis>_L<layer>.json (resumable per file).
Deviation recorded in PLAN addendum: no torch.autocast (the FP8 model already runs in bf16).
Usage: python s2_steer_gen.py --model qwen [--smoke]"""
import argparse, sys
import torch
from transformers import GenerationConfig
import dcommon as dc

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(dc.MODELS)); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--rows", type=int, default=100, help="batch rows (alpha x question); halved on OOM")
a = ap.parse_args()
md = dc.model_dir(a.model); src = md / ("s1_smoke" if a.smoke else "s1") / "diffs_all.pt"
out = md / ("s2_smoke" if a.smoke else "s2"); out.mkdir(parents=True, exist_ok=True)
diffs = torch.load(src)["diffs"]; NL = diffs["economic"].shape[0]
layers = list(range(1, NL)) if not a.smoke else [1, NL // 2, NL - 1]
todo = [(ax, L) for ax in ["economic", "social"] for L in layers if not (out / f"gen_{ax}_L{L}.json").exists()]
if not todo:
    dc.log("S2 generation done, skipped"); sys.exit(0)
tok, model = dc.load(a.model)
assert model.config.num_hidden_layers + 1 == NL
max_new = 24 if a.smoke else 256

class Steer:
    def __init__(self, vec, alphas, layer):
        self.v = (vec / vec.norm().clamp_min(1e-12)).float(); self.al = torch.as_tensor(alphas, dtype=torch.float32); self.layer = layer
    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        if h.shape[0] != len(self.al):
            raise RuntimeError(f"batch mismatch {h.shape[0]} vs {len(self.al)}")
        v = self.v.to(h.device, h.dtype); al = self.al.to(h.device, h.dtype).view(-1, 1)
        n = h[:, -1, :].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h[:, -1, :] += al * n * v.view(1, -1)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    def __enter__(self):
        self.hd = model.model.layers[self.layer - 1].register_forward_hook(self.hook); return self
    def __exit__(self, *e):
        self.hd.remove()

@torch.inference_mode()
def gen_rows(qs, alphas, vec, layer):
    prompts = [dc.chat(tok, [{"role": "user", "content": q}]) for q in qs]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to("cuda")
    cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    with Steer(vec, alphas, layer):
        o = model.generate(**enc, generation_config=cfg)
    return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

rows = a.rows
for ax, L in todo:
    qs10 = dc.axis_data(ax)["questions"][20:30]
    Q = [q for al in ALPHAS for q in qs10]; A = [al for al in ALPHAS for q in qs10]
    ans = []; i = 0
    while i < len(Q):
        try:
            ans += gen_rows(Q[i:i + rows], A[i:i + rows], diffs[ax][L], L); i += rows
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); rows = max(10, rows // 2); dc.log(f"OOM -> rows {rows}")
    torch.cuda.empty_cache()
    dc.atomic_json({"axis": ax, "layer": L, "alphas": A, "questions": Q, "answers": ans, "max_new": max_new}, out / f"gen_{ax}_L{L}.json")
    dc.log(f"{ax} L{L}: {len(ans)} answers | peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
dc.log("S2 generation complete")

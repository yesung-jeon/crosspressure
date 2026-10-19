#!/usr/bin/env python
"""Pre-run load test for a model (standing rule): FP8 load, tokenizer A/B single tokens, chat template with a system prompt, one lens
batch with zero steering, mu at the layer, Steer hook on a random unit direction. Usage: python test_load.py --model olmo"""
import argparse
import numpy as np, torch
import l_common as lc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
tok, model = lc.load(a.model); N = model.config.num_hidden_layers; d = model.config.hidden_size
ia, ib = lc.fc.ab_ids(tok); lc.log("A/B ids", ia, ib, "pad", tok.pad_token, "eos", tok.eos_token)
P = lc.personas("confirm"); b = lc.items()[0][0]
texts = [lc.chat(tok, lc.fc.choice_msgs(P[0]["system"], b["pkgL"], b["pkgR"])), lc.chat(tok, lc.fc.choice_msgs(None, b["pkgL"], b["pkgR"]))]
print(texts[0][:600])
L = 14 if a.model != "olmo" else 16
g = torch.Generator().manual_seed(0); u = torch.randn(d, generator=g); u = u / u.norm(); v = torch.randn(d, generator=g); v = v - (v @ u) * u; v = v / v.norm()
mu = lc.gc.mu_at(model, tok, texts, L); lc.log("mu at layer", L, round(mu, 2))
with lc.Steer(model, L) as steer:
    M, PA, PB = lc.lens_run(tok, model, steer, texts * 4, torch.zeros(8, d))
    lc.log("zero steer: P(A)", np.round(PA[:2], 3), "P(B)", np.round(PB[:2], 3), "mass", np.round((PA + PB)[:2], 3), "margins shape", M.shape)
    M2, PA2, PB2 = lc.lens_run(tok, model, steer, texts * 4, (0.28 * mu * u).repeat(8, 1))
    lc.log("random push 0.28 mu: P(A)", np.round(PA2[:2], 3), "mass", np.round((PA2 + PB2)[:2], 3))
    E, S = lc.project_run(tok, model, steer, texts, torch.zeros(2, d), L + 4, u, v); lc.log("projection ok", np.round(E, 2), np.round(S, 2))
lc.log("test_load ok", a.model, f"GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GB peak")

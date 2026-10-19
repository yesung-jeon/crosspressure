"""Study G shared helpers (PLAN v0.2 + A1). Reuses the checked Study F helpers (f_common): loading, items, chat prompts, Steer hook,
logit lens, decision layer, Löwdin. Adds anonymous party personas, per-axis pushes, prompt_last directions, a final-logit scorer."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

CODE = Path(__file__).resolve().parent
G = CODE.parent
sys.path.insert(0, str(G.parent / "STUDY_F_steer" / "code"))
import f_common as fc                       # noqa: E402
OUT = Path(os.environ.get("STUDYG_OUT", G / "out")); KEY = "qwen"
log, seed_for, atomic_csv, atomic_json, chat, items, lens_run, decision_layer, hbits, lowdin = (
    fc.log, fc.seed_for, fc.atomic_csv, fc.atomic_json, fc.chat, fc.items, fc.lens_run, fc.decision_layer, fc.hbits, fc.lowdin)
CONF_STATES, CONF_YEARS = ["CA", "TX", "NY", "FL", "OH", "GA", "PA", "MI", "AL", "MA"], [2000, 2010, 2020]
CAL_STATES, CAL_YEARS = ["WA", "IL", "NC", "AZ", "WI"], [2004, 2016]

def out_dir(*parts):
    p = OUT.joinpath(KEY, *parts); p.mkdir(parents=True, exist_ok=True); return p

def persona(party, state, year):
    return (f"You are a {party} member of the U.S. House of Representatives representing {fc.STATES[state]} in {year}. "
            f"Answer the question in the first person, as this member would, reflecting this person's actual political positions at that time.")

def personas(kind):
    S, Y = (CONF_STATES, CONF_YEARS) if kind == "confirm" else (CAL_STATES, CAL_YEARS)
    return [dict(pid=f"{p[0]}-{s}-{y}", party=p, state=s, year=y, own=1.0 if p == "Democrat" else -1.0, system=persona(p, s, y))
            for p in ["Democrat", "Republican"] for s in S for y in Y]

def load():
    return fc.load(KEY)

def directions(model, tok, vtype, L):
    """unit orth_e (+ = economic left), orth_s (+ = socially progressive) at layer L for the vector type"""
    if vtype == "response_avg":
        diffs = torch.load(fc.DOUT / KEY / "s1" / "diffs_all.pt")["diffs"]
    else:
        diffs = prompt_last_diffs(model, tok)
    return fc.lowdin(diffs, L)

def prompt_last_diffs(model, tok):
    f = out_dir() / "prompt_last_diffs.pt"
    if f.exists(): return torch.load(f)
    N, d = model.config.num_hidden_layers, model.config.hidden_size; diffs = {}
    for ax in ["economic", "social"]:
        D = json.load(open(fc.KIM.parent / "data_generation" / "trait_data_extract" / f"data_{ax}.json", encoding="utf-8"))
        ins, qs = D.get("instruction", D.get("instructions")), D["questions"][:20]; means = {}
        for side in ["pos", "neg"]:
            texts = [chat(tok, [{"role": "system", "content": p[side]}, {"role": "user", "content": q}]) for p in ins for q in qs]; acc = None
            with torch.inference_mode():
                for b0 in range(0, len(texts), 32):
                    enc = tok(texts[b0:b0 + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                    with fc.Capture(model) as cap:
                        model.model(**enc)
                    H = torch.stack([cap.store[l] for l in range(1, N + 1)], 0).cpu(); acc = H.sum(1) if acc is None else acc + H.sum(1)
            means[side] = acc / len(texts)
        diffs[ax] = torch.cat([torch.zeros(1, d), means["pos"] - means["neg"]], 0)
    torch.save(diffs, f); log("prompt_last directions extracted"); return diffs

def placebo(oe, os_, i):
    return fc.placebo_dirs(oe, os_, n=3)[i]

def push(de, ds, own, oe, os_, ke, ks, mu, plac=None):
    """de, ds in {+1 toward own side, -1 against, 0}; plac = (index, norm_scale) for a placebo of the same norm as (de*ke, ds*ks)"""
    if plac is not None:
        i, scale = plac; return scale * mu * placebo(oe, os_, i)
    return mu * (de * ke * own * oe + ds * ks * own * os_)

def mu_at(model, tok, texts, L):
    with torch.inference_mode(), fc.Capture(model, [L]) as cap:
        ns = []
        for b0 in range(0, len(texts), 32):
            enc = tok(texts[b0:b0 + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda"); model.model(**enc)
            ns.append(cap.store[L].norm(dim=1).cpu())
    return float(torch.cat(ns).mean())

@torch.inference_mode()
def score_final(tok, model, steer, texts, W, batch=48):
    """full-softmax P(A), P(B) of the first answer token with per-row steering vectors W (n, d); OOM-safe, length-sorted"""
    ia, ib = fc.ab_ids(tok)
    lens = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]); order = np.argsort(lens)
    PA, PB = np.zeros(len(texts)), np.zeros(len(texts)); b0 = 0
    while b0 < len(texts):
        idx = order[b0:b0 + batch]
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            steer.W = W[idx]
            h = model.model(**enc).last_hidden_state[:, -1, :]
            p = torch.softmax(model.lm_head(h).float(), -1)
            PA[idx], PB[idx] = p[:, ia].cpu().numpy(), p[:, ib].cpu().numpy(); b0 += len(idx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(2, batch // 2); log(f"OOM -> batch {batch}")
        finally:
            steer.W = None
    return PA, PB

Steer, Capture = fc.Steer, fc.Capture

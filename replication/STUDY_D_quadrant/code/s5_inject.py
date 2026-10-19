#!/usr/bin/env python
"""S5 (PLAN.md): inject the four profiles in five modes and measure C1 (forced-choice entropy), C2 (profile consistency),
C3 (Kim projection), C4 (greedy generation entropy). Forward passes and short greedy generations only; no steering.
Writes out/<model>/s5/{items.csv, proj.csv, gen.csv, conditions.csv} (each stage skipped when its file exists).
Usage: python s5_inject.py --model qwen [--smoke]"""
import argparse, json, sys
import numpy as np, pandas as pd, torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
import dcommon as dc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--redo-mode", default=None, help="A14: recompute only this mode's rows (e.g. I1) and replace them in the existing s5 files")
a = ap.parse_args()
md = dc.model_dir(a.model); O = md / ("s5_smoke" if a.smoke else "s5"); O.mkdir(parents=True, exist_ok=True)
PROFILES = ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]
EP = {"L": "expanding Social Security, public health insurance and union rights", "R": "cutting taxes, reducing government spending and deregulating business"}
SP = {"P": "abortion rights and same-sex marriage", "T": "restrictions on abortion and a traditional definition of marriage"}
STATES = ["CA", "TX", "NY", "FL", "OH", "GA", "PA", "MI", "AL", "MA"]; YEARS = [1975, 1995, 2015]       # E40 V2 bases
FMT = " Answer with Yes or No."
items = json.load(open(dc.CP_ITEMS, encoding="utf-8"))["items"]
single = [it for it in items if it["type"] in ("econ_only", "soc_only")]
ins = {ax: dc.axis_data(ax)["instruction"] for ax in ["economic", "social"]}
if a.smoke:      # smoke only: stand-in members from the S3 smoke file, profiles assigned in turn
    i4 = pd.read_csv(md / "s3_smoke" / "temporal_member_scores.csv").head(8); i4["profile"] = [PROFILES[i % 4] for i in range(len(i4))]
else:
    i4 = pd.read_csv(dc.OUT / "s5" / "i4_members.csv")

def want_yes(it, prof):
    """desired answer of a single-axis item, or (econ_wants_yes, soc_wants_yes) for two-dimensional items"""
    E, S = prof[1], prof[4]
    e = None if it["econ_yes"] == 0 else ((it["econ_yes"] == 1) == (E == "L"))
    s = None if it["soc_yes"] == 0 else ((it["soc_yes"] == 1) == (S == "P"))
    return e, s

def i1_system(st, yr, E, S):
    # A14 (2026-10-16): the sealed version split at the first ". " (inside "U.S."), giving "You are a member of the U.S. This member supports ...
    # House of Representatives ...". Now the full House sentence is kept and the positions sentence follows it.
    persona = dc.anonymous_persona("House", st, yr); cut = persona.index(" Answer the question")
    first, rest = persona[:cut], persona[cut + 1:]
    assert first.endswith(".") and "House of Representatives" in first, first
    return f"{first} This member supports {EP[E]} and {SP[S]}. {rest}"

conds = []   # each: id, mode, profile, base, prefix (messages before the final user turn), upre (user prefix), exclude (item ids)
for prof in PROFILES:
    E, S = prof[1], prof[4]
    for st in STATES:
        for yr in YEARS:
            conds.append(dict(mode="I1", profile=prof, base=f"{st}{yr}", prefix=[{"role": "system", "content": i1_system(st, yr, E, S)}], upre="", exclude=[]))
            conds.append(dict(mode="I5", profile=prof, base=f"{st}{yr}", prefix=[],
                              upre=f"Answer as a member of the U.S. House of Representatives from {dc.STATES[st]} in {yr} who supports {EP[E]} and {SP[S]}.\n\n", exclude=[]))
    for k in range(5):
        e_txt = ins["economic"][k]["pos" if E == "L" else "neg"]; s_txt = ins["social"][k]["pos" if S == "P" else "neg"]
        for order, txt in [(0, f"{e_txt} {s_txt}"), (1, f"{s_txt} {e_txt}")]:
            conds.append(dict(mode="I2", profile=prof, base=f"k{k}o{order}", prefix=[{"role": "system", "content": txt}], upre="", exclude=[]))
    for b in range(30):
        rng = np.random.default_rng(dc.seed_for("D-S5-I3", prof, b) % 2**32)
        ec = [it for it in single if it["type"] == "econ_only"]; so = [it for it in single if it["type"] == "soc_only"]
        shown = [ec[i] for i in rng.choice(len(ec), 2, replace=False)] + [so[i] for i in rng.choice(len(so), 2, replace=False)]
        rng.shuffle(shown); pre = []
        for it in shown:
            e, s = want_yes(it, prof); y = e if e is not None else s
            pre += [{"role": "user", "content": it["q"] + FMT}, {"role": "assistant", "content": "Yes." if y else "No."}]
        conds.append(dict(mode="I3", profile=prof, base=f"fs{b}", prefix=pre, upre="", exclude=[it["id"] for it in shown]))
    for _, r in i4[i4.profile == prof].iterrows():
        conds.append(dict(mode="I4", profile=prof, base=f"m{int(r.icpsr)}c{int(r.congress)}",
                          prefix=[{"role": "system", "content": dc.kim_persona(r["name"], r.chamber, r.state, int(r.year))}], upre="", exclude=[]))
for i, c in enumerate(conds): c["cid"] = i
REDO = None
if a.redo_mode:   # A14: keep every other mode's rows; recompute this mode into a temporary folder, then merge by cid
    REDO = {c["cid"] for c in conds if c["mode"] == a.redo_mode}; assert REDO, a.redo_mode
    FINAL = O; O = md / f"s5_redo_{a.redo_mode}{'_smoke' if a.smoke else ''}"; O.mkdir(parents=True, exist_ok=True)   # A15: smoke never writes the real redo folder
if a.smoke:
    conds = [c for c in conds if c["base"] in ("CA1975", "k0o0", "fs0") or c["cid"] in {x["cid"] for x in conds if x["mode"] == "I4"}][:40]
dc.log(f"conditions: {len(conds)}; by mode {pd.Series([c['mode'] for c in conds]).value_counts().to_dict()}")
pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in c.items()} for c in conds]).to_csv(O / "conditions.csv", index=False)
ALL = conds
if REDO is not None: conds = [c for c in conds if c["cid"] in REDO]; REDO = {c["cid"] for c in conds}; dc.log(f"A14 redo {a.redo_mode}: {len(conds)} conditions")   # after the smoke subset

tok, model = dc.load(a.model)
YES, NO = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
assert len(YES) == 1 and len(NO) == 1, (YES, NO); YES, NO = YES[0], NO[0]
EOS = tok.eos_token_id

def build(c, q):
    return dc.chat(tok, c["prefix"] + [{"role": "user", "content": c["upre"] + q}])

@torch.inference_mode()
def yes_no(prompts, batch=32):
    """full-softmax P(Yes), P(No) of the first answer token"""
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    res = np.zeros((len(prompts), 2)); b0 = 0
    while b0 < len(prompts):
        idx = order[b0:b0 + batch]
        try:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            h = model.model(**enc).last_hidden_state[:, -1, :]
            p = torch.softmax(model.lm_head(h).float(), -1)
            res[idx] = torch.stack([p[:, YES], p[:, NO]], 1).cpu().numpy(); b0 += batch
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(1, batch // 2); dc.log(f"OOM -> yes/no batch {batch}")
    return res

# ---- C1 / C2: forced choice on all 64 items ----
f_items = O / "items.csv"
if not f_items.exists():
    rows, prompts = [], []
    for c in conds:
        for it in items:
            if it["id"] in c["exclude"]:
                continue
            rows.append(dict(cid=c["cid"], item=it["id"], type=it["type"])); prompts.append(build(c, it["q"] + FMT))
    dc.log(f"C1: {len(prompts)} forced-choice prompts")
    P = yes_no(prompts)
    df = pd.DataFrame(rows); df["p_yes_raw"], df["p_no_raw"] = P[:, 0], P[:, 1]
    dc.atomic_csv(df, f_items); dc.log("C1 saved")

# ---- C3: Kim projection ----
V = torch.load(md / "vectors.pt") if not a.smoke else torch.load(md / "vec_smoke" / "vectors.pt")
kq = dc.kim_questions()
f_proj = O / "proj.csv"
if not f_proj.exists():
    base = dc.project(tok, model, V, [dc.chat(tok, [{"role": "user", "content": q["question"]}]) for q in kq])
    rows, prompts = [], []
    for c in conds:
        for j, q in enumerate(kq):
            rows.append(dict(cid=c["cid"], q=j, axis=q["axis"])); prompts.append(build(c, q["question"]))
    P = dc.project(tok, model, V, prompts); torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df["pe"], df["ps"] = P[:, 0], P[:, 1]
    df["be"] = [base[j, 0] for j in df.q]; df["bs"] = [base[j, 1] for j in df.q]
    dc.atomic_csv(df, f_proj); dc.log("C3 saved", V["layer"])

# ---- C4: greedy generation entropy (first 64 tokens) ----
class EntRec(LogitsProcessor):
    def __init__(self): self.ent = []
    def __call__(self, input_ids, scores):
        lp = torch.log_softmax(scores.float(), -1); self.ent.append(-(lp.exp() * lp).sum(-1).cpu()); return scores

f_gen = O / "gen.csv"
if not f_gen.exists():
    rows, prompts = [], []
    for c in conds:
        for j, q in enumerate(kq):
            rows.append(dict(cid=c["cid"], q=j, axis=q["axis"])); prompts.append(build(c, q["question"]))
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    ent = np.full(len(prompts), np.nan); ntok = np.zeros(len(prompts), int); texts = [""] * len(prompts)
    cfg = GenerationConfig(max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=EOS)
    batch = 64; b0 = 0
    while b0 < len(prompts):
        idx = order[b0:b0 + batch]
        try:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            rec = EntRec()
            with torch.inference_mode():
                o = model.generate(**enc, generation_config=cfg, logits_processor=LogitsProcessorList([rec]))
            gen = o[:, enc["input_ids"].shape[1]:].cpu(); E = torch.stack(rec.ent, 1).numpy()      # (B, steps)
            for j, i in enumerate(idx):
                g = gen[j].tolist(); n = g.index(EOS) + 1 if EOS in g else len(g)
                ent[i] = float(E[j, :n].mean()); ntok[i] = n; texts[i] = tok.decode(g[:n], skip_special_tokens=True)
            b0 += batch
            if (b0 // batch) % 20 == 0: dc.log(f"  C4 {b0}/{len(prompts)}")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(4, batch // 2); dc.log(f"OOM -> gen batch {batch}")
    df = pd.DataFrame(rows); df["ent"], df["ntok"], df["text"] = ent, ntok, texts
    dc.atomic_csv(df, f_gen); dc.log("C4 saved")
if REDO is not None:   # merge: replace the redone mode's rows in the final files (old files are kept in s5_pre_A14 by the chain)
    for f in ("items.csv", "proj.csv", "gen.csv"):
        old = pd.read_csv(FINAL / f, dtype=str, keep_default_na=False); new = pd.read_csv(O / f, dtype=str, keep_default_na=False)   # text kept: untouched rows stay byte-identical
        assert set(new.cid.astype(int)) == REDO and list(old.columns) == list(new.columns)
        m = pd.concat([old[~old.cid.astype(int).isin(REDO)], new], ignore_index=True); m = m.iloc[m.cid.astype(int).argsort(kind="stable")]   # stable by cid
        assert len(m) == len(old), (f, len(m), len(old)); tmp = FINAL / (f + ".tmp"); m.to_csv(tmp, index=False); import os; os.replace(tmp, FINAL / f)
    import shutil; shutil.copy(O / "conditions.csv", FINAL / "conditions.csv"); dc.log(f"A14 merged {a.redo_mode} into {FINAL}")
dc.log("S5 complete")

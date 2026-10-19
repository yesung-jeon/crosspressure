"""Shared definitions for uncertainty_map (one place for keys, prompts, measurement).

Measurement follows the repository's notebook 03 (rev1) and Junsol Kim's congress_map:
- persona wording, state names and name cleaning are read from congress_map/03_project_legislators.py by AST
  (his file is not modified or copied by hand);
- projection: last prompt token at layer 17 (forward stops there), dotted with orth_e / orth_s from
  congress_map/results/vectors.pt, minus the same quantity with no persona (origin = the model itself);
  x = -econ_orth (right = economic right), y = -social_orth (up = traditional);
- generation entropy: top-40 renormalised, bits, mean over steps up to and including the first EOS, first 64 steps;
- sampling: do_sample, temperature 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0 (explicit; Kim K14);
- seeds: seed_for(labels) = first 8 hex digits of SHA-256 of the labels (Kim K13).
"""
import ast, hashlib, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # crosspressure/
KIM_MAP = REPO / "congress_map"
VOTEVIEW = HERE.parent / "data" / "HSall_members.csv"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SUFFIX = "\n\nAnswer in no more than 100 words."     # notebook 03 PROMPT_SUFFIX (generation prompts only)
TOPK, MAX_NEW = 40, 64
SOUTH = {"AL", "AR", "FL", "GA", "LA", "MS", "NC", "SC", "TN", "TX", "VA"}
HEDGING_WORDS = ["however", "although", "though", "nevertheless", "nonetheless", "on the other hand", "that said", "depends",
                 "it depends", "complex", "nuanced", "complicated", "multifaceted", "balance", "balanced", "both sides",
                 "multiple perspectives", "arguably", "debatable", "controversial", "some argue", "others argue", "some believe",
                 "others believe", "pros and cons", "trade-off", "tradeoff", "not necessarily", "it's important to consider",
                 "important to note", "whereas"]                       # notebook 03 rev1, verbatim
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def seed_for(*labels): return int(hashlib.sha256("|".join(map(str, labels)).encode()).hexdigest()[:8], 16)
def year_of(c): return 1789 + 2 * (int(c) - 1)

# ---- Kim's persona wording (AST import from his script) ----
_ns = {"re": re}
for node in ast.parse((KIM_MAP / "03_project_legislators.py").read_text(encoding="utf-8")).body:
    if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "STATES") or \
       (isinstance(node, ast.FunctionDef) and node.name == "nice"):
        exec(compile(ast.Module([node], []), "kim03", "exec"), _ns)
STATES, nice = _ns["STATES"], _ns["nice"]
CHAMBER_LONG = {"House": "House of Representatives", "Senate": "Senate"}

def persona(name, chamber, state, year):
    """Kim's wording, verbatim."""
    return (f"You are {name}, who serves in the U.S. {CHAMBER_LONG[chamber]} representing {STATES.get(state, state)} in {year}. "
            f"Answer the question in the first person, as {name} would, reflecting this person's actual political positions at that time.")

def anonymous_persona(chamber, state, year):
    """Kim's wording with the name removed (step 3)."""
    return (f"You are a member of the U.S. {CHAMBER_LONG[chamber]} representing {STATES.get(state, state)} in {year}. "
            f"Answer the question in the first person, as this member would, reflecting this person's actual political positions at that time.")

def kim_questions():
    """Held-out questions 20-24 per axis, exactly as congress_map step 3."""
    qs = []
    for ax in ["economic", "social"]:
        ev = json.load(open(REPO / "data_generation" / "trait_data_extract" / f"data_{ax}.json"))["questions"][20:25]
        qs += [dict(question=q, axis=ax) for q in ev]
    return qs

def members():
    """Kim's member scores joined to Voteview (key: congress x icpsr, 1:1)."""
    m = pd.read_csv(KIM_MAP / "results" / "temporal_member_scores.csv")
    v = pd.read_csv(VOTEVIEW, low_memory=False)[["congress", "icpsr", "chamber", "party_code", "nominate_geo_mean_probability"]]
    v = v[v.chamber.isin(["House", "Senate"])].drop_duplicates(["congress", "icpsr"])
    m = m.merge(v[["congress", "icpsr", "nominate_geo_mean_probability"]], on=["congress", "icpsr"], how="left", validate="1:1")
    m = m.rename(columns={"nominate_geo_mean_probability": "gmp"})
    m["x"], m["y"] = -m.econ_orth, -m.social_orth
    m["group"] = np.where(m.party == "Republican", "Rep", np.where((m.party == "Democrat") & m.state.isin(SOUTH), "SouthDem",
                          np.where(m.party == "Democrat", "OtherDem", "Other")))
    m["era"] = pd.cut(m.congress, [79, 96, 109, 118], labels=["1947-79", "1981-2005", "2007-23"]).astype(str)
    return m

# ---- model ----
def load(quant="int8"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    kw = dict(device_map={"": 0})
    if quant == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else: kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    V = torch.load(KIM_MAP / "results" / "vectors.pt")
    return tok, model, V

def chat(tok, q, system=None):
    msg = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

class _Stop(Exception):
    pass

@torch.inference_mode()
def project(tok, model, V, prompts, batch=64):
    """(n, 2) projections on orth_e, orth_s of the last-token state at layer V['layer'] (Kim K15: stop after that layer)."""
    L = int(V["layer"]); ve, vs = V["orth_e"].to(model.device).float(), V["orth_s"].to(model.device).float(); cap = {}
    def grab(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; cap["h"] = h[:, -1, :].float(); raise _Stop
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    res = np.zeros((len(prompts), 2), dtype=np.float32); hk = model.model.layers[L - 1].register_forward_hook(grab)
    try:
        for b0 in range(0, len(prompts), batch):
            idx = order[b0:b0 + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            try: model.model(**enc)
            except _Stop: pass
            h = cap.pop("h"); res[idx] = torch.stack([h @ ve, h @ vs], 1).cpu().numpy()
            nb = b0 // batch + 1
            if nb % 100 == 0: log(f"  project {min(b0 + batch, len(prompts))}/{len(prompts)}")
    finally:
        hk.remove()
    return res

@torch.inference_mode()
def states(tok, model, V, prompts, batch=64):
    """(n, hidden) float32 last-token states at layer V['layer'] (same early exit as project())."""
    L = int(V["layer"]); cap = {}
    def grab(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; cap["h"] = h[:, -1, :].float(); raise _Stop
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    res = None; hk = model.model.layers[L - 1].register_forward_hook(grab)
    try:
        for b0 in range(0, len(prompts), batch):
            idx = order[b0:b0 + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            try: model.model(**enc)
            except _Stop: pass
            h = cap.pop("h").cpu().numpy()
            if res is None: res = np.zeros((len(prompts), h.shape[1]), dtype=np.float32)
            res[idx] = h
    finally:
        hk.remove()
    return res

@torch.inference_mode()
def project_dirs(tok, model, V, prompts, dirs, batch=64):
    """(n, k) projections of layer-V['layer'] last-token states on the rows of dirs (k, hidden); states are not kept in memory."""
    L = int(V["layer"]); D = torch.tensor(np.asarray(dirs), dtype=torch.float32, device=model.device).T; cap = {}
    def grab(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; cap["h"] = h[:, -1, :].float(); raise _Stop
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    res = np.zeros((len(prompts), D.shape[1]), dtype=np.float32); hk = model.model.layers[L - 1].register_forward_hook(grab)
    try:
        for b0 in range(0, len(prompts), batch):
            idx = order[b0:b0 + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            try: model.model(**enc)
            except _Stop: pass
            res[idx] = (cap.pop("h") @ D).cpu().numpy()
            if (b0 // batch + 1) % 100 == 0: log(f"  project_dirs {min(b0 + batch, len(prompts))}/{len(prompts)}")
    finally:
        hk.remove()
    return res

def persona_scores(tok, model, V, systems):
    """Kim's score for each system prompt (None = no persona): x = -(econ proj on economic qs - base), y = -(social proj on social qs - base)."""
    qs = kim_questions(); E = [j for j, q in enumerate(qs) if q["axis"] == "economic"]; S = [j for j, q in enumerate(qs) if q["axis"] == "social"]
    base = project(tok, model, V, [chat(tok, q["question"]) for q in qs])
    P = project(tok, model, V, [chat(tok, q["question"], s) for s in systems for q in qs]).reshape(len(systems), len(qs), 2)
    return -(P[:, E, 0].mean(1) - base[E, 0].mean()), -(P[:, S, 1].mean(1) - base[S, 1].mean())

class EntropyRecorder:
    def __init__(self): self.steps = []
    def __call__(self, input_ids, scores):
        p = torch.softmax(torch.topk(scores.float(), k=TOPK, dim=-1).values, dim=-1)
        self.steps.append(-(p * torch.log2(p.clamp_min(1e-12))).sum(-1)); return scores

def hedging_density(text):
    t, cnt = str(text).lower(), 0
    for w in sorted(HEDGING_WORDS, key=len, reverse=True):
        pat = r"\b" + re.escape(w) + r"\b"; cnt += len(re.findall(pat, t)); t = re.sub(pat, " ", t)
    n = len(str(text).split()); return cnt / n * 100 if n else 0.0

class Steer:
    """Adds ||h_last|| * dirs[row] at the last position of decoder block `layer`-1 output on every forward step (notebook 03)."""
    def __init__(self, model, layer, dirs): self.model, self.layer, self.dirs, self.h = model, layer, dirs, None
    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; last = h[:, -1, :]; h = h.clone()
        h[:, -1, :] = last + last.float().norm(dim=-1, keepdim=True).clamp_min(1e-8).to(h.dtype) * self.dirs
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    def __enter__(self): self.h = self.model.model.layers[self.layer - 1].register_forward_hook(self._hook); return self
    def __exit__(self, *a): self.h.remove()

@torch.inference_mode()
def generate_entropy(tok, model, jobs, out_csv, label, batch=96, steer_layer=None):
    """jobs: dicts with 'prompt' plus id fields (and optionally 'steer': a hidden-size vector, already scaled, applied with Steer
    at `steer_layer`). Appends one row per job to out_csv; resumes by batch id."""
    from transformers import LogitsProcessorList
    out_csv = Path(out_csv); done = pd.read_csv(out_csv) if out_csv.exists() else None
    start = 0 if done is None else int(done.batch_id.max()) + 1
    lens = [len(tok(j["prompt"], add_special_tokens=False)["input_ids"]) for j in jobs]
    order = np.argsort(lens, kind="stable"); batches = [order[i:i + batch] for i in range(0, len(order), batch)]
    log(f"{label}: {len(jobs)} sequences, {len(batches)} batches, resume at {start}"); t0 = time.time()
    for b, idx in enumerate(batches):
        if b < start: continue
        js = [jobs[i] for i in idx]
        enc = tok([j["prompt"] for j in js], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        rec = EntropyRecorder(); torch.manual_seed(seed_for(label, b))
        import contextlib
        ctx = contextlib.nullcontext()
        if steer_layer is not None:
            D = torch.tensor(np.stack([np.asarray(j.get("steer", np.zeros(model.config.hidden_size)), dtype=np.float32) for j in js]),
                             device=model.device, dtype=torch.bfloat16)
            ctx = Steer(model, steer_layer, D)
        with ctx:
            seq = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                 logits_processor=LogitsProcessorList([rec]), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = seq[:, enc["input_ids"].shape[1]:]; eos = gen.eq(tok.eos_token_id)
        L = torch.where(eos.any(1), eos.float().argmax(1).long() + 1, torch.full_like(eos[:, 0], gen.shape[1], dtype=torch.long))
        Em = torch.stack(rec.steps, 1); pos = torch.arange(Em.shape[1], device=Em.device)[None]
        m = ((pos < L[:, None]) & (pos < MAX_NEW)).float(); e64 = ((Em * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy()
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        rows = [{k: v for k, v in j.items() if k not in ("prompt", "steer")} | dict(batch_id=b, entropy64=float(e64[i]), n_tokens=int(L[i]),
                hedge=hedging_density(texts[i]), text=texts[i]) for i, j in enumerate(js)]
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        el = time.time() - t0; nb = b - start + 1
        if nb % 10 == 0 or b == len(batches) - 1:
            log(f"  {label} batch {b+1}/{len(batches)}  {el/nb:.1f}s/batch  eta {el/nb*(len(batches)-b-1)/60:.1f} min")

# ---- estimation: OLS with CR1 clustered SE, reconstructed independently (kim-estimate rule) ----
def fit_cr1(df, formula, cluster, tol=1e-6):
    import statsmodels.formula.api as smf
    mod = smf.ols(formula, data=df); used = df.loc[mod.data.row_labels]
    g = pd.factorize(used[cluster].astype(str))[0]
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": True, "df_correction": True})
    X, y = mod.exog, mod.endog; beta = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ beta
    N, K = X.shape; G = len(np.unique(g)); XtXi = np.linalg.pinv(X.T @ X); meat = np.zeros((K, K))
    for k in np.unique(g):
        s = X[g == k].T @ u[g == k]; meat += np.outer(s, s)
    Vr = (G / (G - 1)) * ((N - 1) / (N - K)) * XtXi @ meat @ XtXi
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(b)))
    if rel(beta, res.params.values) > tol or rel(Vr, res.cov_params().values) > tol:
        raise RuntimeError(f"CR1 reconstruction mismatch: {formula}")
    return res

def coef(res, term):
    lo, hi = res.conf_int().loc[term]
    return dict(est=float(res.params[term]), se=float(res.bse[term]), ci95=[float(lo), float(hi)], p=float(res.pvalues[term]), n=int(res.nobs))

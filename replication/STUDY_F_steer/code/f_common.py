"""Study F shared helpers (PLAN v0.3): paths, models, Kim persona, chat prompts, items, two-axis steering hook, per-layer last-token
capture (logit lens and projections), decision layer."""
import hashlib, itertools, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

CODE = Path(__file__).resolve().parent
F = CODE.parent
KIM = F.parent
OUT = Path(os.environ.get("STUDYF_OUT", F / "out"))
DOUT = KIM / "STUDY_D_quadrant" / "out"
ITEMS = KIM / "STUDY_E_alignment" / "items_draft.json"
ITEMS_SHA = "56a9c75740130d07363e7eee66072c18672ab67759041b8f603cb693c8c04a87"
MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
GPU_FRACTION = 0.80
CHAMBER_LONG = {"House": "House of Representatives", "Senate": "Senate"}
STATES = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"the District of Columbia","PR":"Puerto Rico","VI":"the U.S. Virgin Islands","GU":"Guam","AS":"American Samoa","MP":"the Northern Mariana Islands"}
CELLS = {"NONE": (0, 0), "TT": (1, 1), "TA": (1, -1), "AT": (-1, 1), "AA": (-1, -1)}      # (econ, social): +1 toward own side

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def seed_for(*p): return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)
def model_out(key):
    p = OUT / key; p.mkdir(parents=True, exist_ok=True); return p
def atomic_csv(df, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip" if path.name.endswith(".gz") else None); os.replace(tmp, path)   # compression from the final name
def atomic_json(obj, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float); os.replace(tmp, path)

def ram_guard(need=3.0, wait=3 * 3600):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available / 2**30 < need:
        if time.time() - t0 > wait: raise SystemExit("RAM guard timeout")
        log("RAM guard: waiting"); time.sleep(60)

def load(key):
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
    sys.path.insert(0, str(KIM / "STUDY_D_quadrant" / "code"))
    from fp8_loader import load_fp8
    from transformers import AutoTokenizer
    ram_guard()
    tok = AutoTokenizer.from_pretrained(MODELS[key]); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = load_fp8(MODELS[key])
    log("loaded", MODELS[key], f"FP8 | layers {model.config.num_hidden_layers} | GPU {torch.cuda.memory_allocated()/2**30:.1f} GB")
    return tok, model

def lowdin(diffs, L):
    """Kim's step-1 orthogonalization (congress_map/01_extract_vectors.py), unit orth_e, orth_s at layer L"""
    ve, vs = diffs["economic"], diffs["social"]
    ue, us = ve[L] / ve[L].norm(), vs[L] / vs[L].norm()
    V = torch.stack([ue, us], 1); w, U = torch.linalg.eigh(V.T @ V)
    Q = V @ (U @ torch.diag(w.rsqrt()) @ U.T); oe, os_ = Q[:, 0], Q[:, 1]
    if torch.dot(oe, ue) < 0: oe = -oe
    if torch.dot(os_, us) < 0: os_ = -os_
    return oe / oe.norm(), os_ / os_.norm()

def vectors(key):
    """(L, orth_e, orth_s) at the Study D rule layer, and (L+4, orth_e, orth_s) for calibration readout"""
    V = torch.load(DOUT / key / "vectors.pt"); L = int(V["layer"])
    diffs = torch.load(DOUT / key / "s1" / "diffs_all.pt")["diffs"]
    oe, os_ = lowdin(diffs, L)
    assert torch.allclose(oe, V["orth_e"].float(), atol=1e-5) and torch.allclose(os_, V["orth_s"].float(), atol=1e-5)
    oe4, os4 = lowdin(diffs, L + 4)
    return L, oe, os_, oe4, os4

def placebo_dirs(oe, os_, n=3):
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed_for("F-plac", i) % 2**31)
        u = torch.randn(oe.shape[0], generator=g)
        for v in (oe, os_): u = u - (u @ v) * v
        out.append(u / u.norm())
    return out

def kim_persona(name, chamber, state, year):
    return (f"You are {name}, who serves in the U.S. {CHAMBER_LONG[chamber]} representing {STATES.get(state, state)} in {year}. "
            f"Answer the question in the first person, as {name} would, reflecting this person's actual political positions at that time.")

def kim_questions(n_q=5):
    qs = []
    for ax in ["economic", "social"]:
        d = json.load(open(KIM.parent / "data_generation" / "trait_data_extract" / f"data_{ax}.json", encoding="utf-8"))
        qs += [dict(question=q, axis=ax) for q in d["questions"][20:20 + n_q]]
    return qs

def items():
    assert hashlib.sha256(open(ITEMS, "rb").read()).hexdigest() == ITEMS_SHA, "Study E items changed after approval"
    I = json.load(open(ITEMS, encoding="utf-8")); E, S = I["economic"], I["social"]
    cap = lambda x: x[0].upper() + x[1:]
    bundles = []
    for econ, soc, fam in itertools.product(E, S, ["party_like", "reversed"]):
        sl, sr = ("P", "T") if fam == "party_like" else ("T", "P")
        bundles.append(dict(topic=f"{econ}|{soc}", econ=econ, soc=soc, family=fam, kang=S[soc]["kang_split"],
                            pkgL=cap(f"{E[econ]['L']} and {S[soc][sl]}"), pkgR=cap(f"{E[econ]['R']} and {S[soc][sr]}")))
    manip = [dict(axis="econ", issue=k, first=cap(v["L"]), second=cap(v["R"])) for k, v in E.items()] + \
            [dict(axis="soc", issue=k, first=cap(v["P"]), second=cap(v["T"])) for k, v in S.items()]
    return bundles, manip

def choice_msgs(system, a, b):
    m = [{"role": "system", "content": system}] if system else []
    return m + [{"role": "user", "content": f"You must choose between two policy packages.\nA) {a}\nB) {b}\nWhich do you choose? Answer with A or B only."}]

def open_msgs(system, a, b):
    m = [{"role": "system", "content": system}] if system else []
    return m + [{"role": "user", "content": f"You must choose between two policy packages. Package A would {a[0].lower() + a[1:]}. Package B would {b[0].lower() + b[1:]}. Which do you choose, and why?"}]

def chat(tok, msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def ab_ids(tok):
    a, b = tok.encode("A", add_special_tokens=False), tok.encode("B", add_special_tokens=False)
    assert len(a) == 1 and len(b) == 1, (a, b); return a[0], b[0]

class Steer:
    """adds W[i] (one vector per batch row) to the output of decoder layer L (1-based) at every position"""
    def __init__(self, model, L): self.model, self.L, self.W, self.h = model, L, None, None
    def hook(self, mod, inp, out):
        if self.W is None: return None
        x = out[0] if isinstance(out, tuple) else out
        x = x + self.W.to(x.device, x.dtype)[:, None, :]
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    def __enter__(self): self.h = self.model.model.layers[self.L - 1].register_forward_hook(self.hook); return self
    def __exit__(self, *e): self.h.remove()

class Capture:
    """captures the last-position output of every decoder layer (registered after Steer, so it sees steered states)"""
    def __init__(self, model, layers=None):
        self.model = model; self.layers = layers or list(range(1, model.config.num_hidden_layers + 1)); self.h = []; self.store = {}
    def make(self, l):
        def f(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out; self.store[l] = x[:, -1, :].float()
        return f
    def __enter__(self):
        self.h = [self.model.model.layers[l - 1].register_forward_hook(self.make(l)) for l in self.layers]; return self
    def __exit__(self, *e):
        for h in self.h: h.remove()

@torch.inference_mode()
def lens_batch(tok, model, steer, texts, W):
    """returns margins (B, N) = logit_A - logit_B per layer via final norm + lm_head rows, and final P(A), P(B) (full softmax)"""
    ia, ib = ab_ids(tok); N = model.config.num_hidden_layers
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    steer.W = W
    with Capture(model) as cap:
        model.model(**enc)
    steer.W = None
    H = torch.stack([cap.store[l] for l in range(1, N + 1)], 1)              # (B, N, d), float32
    Hn = model.model.norm(H.to(model.lm_head.weight.dtype))                     # choice path, unchanged (sealed P(A), P(B))
    full = model.lm_head(Hn[:, -1, :]).float(); p = torch.softmax(full, -1)
    # R3 / F A9 (2026-10-16): lens in float32. The sealed lens used a bf16 two-row product, whose last-layer sign could differ from the
    # choice and whose exact zeros added spurious flips. The last layer's margin is the choice's own A/B logit difference unless it is
    # exactly zero, then the float32 lens margin breaks the tie.
    Hn32 = model.model.norm(H.float()).float()
    rows = model.lm_head.weight[[ia, ib]].float()                              # (2, d)
    lg = Hn32 @ rows.T                                                          # (B, N, 2) float32
    m = lg[..., 0] - lg[..., 1]
    dfull = full[:, ia] - full[:, ib]
    m[:, -1] = torch.where(dfull != 0, dfull, m[:, -1])
    return m.cpu().numpy(), p[:, ia].cpu().numpy(), p[:, ib].cpu().numpy()

def decision_layer(m):
    """smallest l (1-based) with sign(m_l') == sign(m_N) for all l' >= l"""
    s = np.sign(m[-1]); ok = np.sign(m) == s
    bad = np.where(~ok)[0]
    return int(bad.max() + 2) if len(bad) else 1

def hbits(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def lens_run(tok, model, steer, texts, W, batch=32):
    """batched lens_batch with OOM halving; W is a (n, d) tensor (zeros for unsteered rows); length-sorted"""
    N = model.config.num_hidden_layers
    lens = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]); order = np.argsort(lens)
    M = np.zeros((len(texts), N)); PA = np.zeros(len(texts)); PB = np.zeros(len(texts)); b0 = 0
    while b0 < len(texts):
        idx = order[b0:b0 + batch]
        try:
            m, pa, pb = lens_batch(tok, model, steer, [texts[i] for i in idx], W[idx])
            M[idx], PA[idx], PB[idx] = m, pa, pb; b0 += len(idx)
            if (b0 // batch) % 200 == 0: log(f"  lens {b0}/{len(texts)}")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(2, batch // 2); log(f"OOM -> lens batch {batch}")
    return M, PA, PB

STR = {1: "leans slightly toward supporting", 2: "generally supports", 3: "is firmly and passionately committed to"}
ECONT = {"L": "expanding public health insurance, Social Security and union rights", "R": "cutting taxes, reducing government spending and deregulating business"}
SCONT = {"P": "abortion rights and same-sex marriage", "T": "restrictions on abortion and a traditional definition of marriage"}

def push(cell, own, oe, os_, k, mu, plac=None):
    """steering vector for one row: own = +1 Democrat / -1 Republican (orth_e + = left, orth_s + = progressive)"""
    if cell == "NONE": return torch.zeros_like(oe)
    if cell.startswith("PLAC"): return k * mu * np.sqrt(2) * plac[int(cell[-1]) - 1]
    de, ds = CELLS[cell]; return k * mu * (de * own * oe + ds * own * os_)

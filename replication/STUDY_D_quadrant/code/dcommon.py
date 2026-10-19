"""Study D shared helpers (PLAN.md). Paths, models, FP8 loading, RAM guard, Kim's persona wording (copied from congress_map step 3)."""
import hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
D = CODE.parent                                   # kim/STUDY_D_quadrant
KIM = D.parent                                    # kim/
REPO = KIM.parent                                 # crosspressure/
CMAP = REPO / "congress_map"
OUT = Path(os.environ.get("STUDYD_OUT", D / "out"))   # override only for CPU smoke tests
MEMBERS = KIM / "data" / "HSall_members.csv"
ISSUE_SCALES = KIM / "STUDY_B_polarization" / "out" / "data" / "issue_scales.csv"
CP_ITEMS = KIM / "STUDY_B_polarization" / "issue_data" / "cp3_items.json"
MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct",
          "olmo": "allenai/OLMo-2-1124-7B-Instruct"}   # A11 (2026-10-15): added for Study L's third model; Qwen/Llama paths unchanged
JUDGE = "llama"

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def seed_for(*p):
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS(); s.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 2**30

def ram_guard(need_gb=3.0, max_wait_s=3 * 3600):
    t0 = time.time()
    while free_ram_gb() < need_gb:
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"RAM guard: < {need_gb} GB free for {max_wait_s}s")
        log(f"RAM guard: {free_ram_gb():.1f} GB free, waiting"); time.sleep(60)

def model_dir(key):
    p = OUT / key; p.mkdir(parents=True, exist_ok=True); return p

GPU_FRACTION = 0.80   # PyTorch may use at most 80% of VRAM, so an over-large batch raises OOM instead of spilling into system RAM

def cap_gpu():
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)

def load(key):
    """FP8 model + tokenizer (left padding, pad = eos when missing)."""
    cap_gpu()
    sys.path.insert(0, str(CODE))
    from fp8_loader import load_fp8
    from transformers import AutoTokenizer
    ram_guard()
    tok = AutoTokenizer.from_pretrained(MODELS[key]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_fp8(MODELS[key])
    log("loaded", MODELS[key], "FP8 | layers", model.config.num_hidden_layers, f"| GPU {torch.cuda.memory_allocated()/2**30:.1f} GB")
    return tok, model

def patch_from_pretrained():
    """Make transformers' AutoModelForCausalLM.from_pretrained load FP8 (used to run Kim's scripts unchanged via runpy)."""
    sys.path.insert(0, str(CODE))
    from fp8_loader import load_fp8
    import transformers
    cap_gpu()
    transformers.AutoModelForCausalLM.from_pretrained = lambda model_id, *a, **kw: _chunk_hidden(load_fp8(model_id))

def _chunk_hidden(model, chunk=2):
    """Addendum A5: when a caller asks for output_hidden_states (Kim step 1, response_avg_hidden, batch 16), run the base model in
    chunks of `chunk` rows and return only hidden_states (no full-sequence logits). Rows are independent, so the values Kim reads
    are the same; generation and every other call go through the original forward."""
    from types import SimpleNamespace
    orig = model.forward
    def fwd(*args, **kw):
        if not kw.get("output_hidden_states"):
            return orig(*args, **kw)
        ids, am = kw["input_ids"], kw.get("attention_mask")
        hs = None
        torch.cuda.empty_cache()
        for i in range(0, ids.shape[0], chunk):
            o = model.model(input_ids=ids[i:i + chunk], attention_mask=None if am is None else am[i:i + chunk], output_hidden_states=True)
            if hs is None:      # preallocate once; chunks are copied in place (no concatenation copies)
                hs = [torch.empty((ids.shape[0],) + t.shape[1:], dtype=t.dtype, device=t.device) for t in o.hidden_states]
            for dst, t in zip(hs, o.hidden_states):
                dst[i:i + t.shape[0]] = t
            del o
        return SimpleNamespace(hidden_states=tuple(hs))
    model.forward = fwd
    return model

def chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp"); json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False); os.replace(tmp, path)

def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)

def axis_data(axis):
    return json.load(open(REPO / "data_generation" / "trait_data_extract" / f"data_{axis}.json", encoding="utf-8"))

# ---- Kim's congress_map step 3 wording (copied verbatim) ----
STATES = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"the District of Columbia","PR":"Puerto Rico","VI":"the U.S. Virgin Islands","GU":"Guam","AS":"American Samoa","MP":"the Northern Mariana Islands"}
CHAMBER_LONG = {"House": "House of Representatives", "Senate": "Senate"}

def nice(bioname):
    last, _, rest = bioname.partition(", ")
    nick = re.search(r"\((.*?)\)", rest); rest = re.sub(r"\s*\(.*?\)", "", rest)
    toks = [t.strip(",") for t in rest.replace(",", " ").split()]
    suffix = [t for t in toks if t.upper().rstrip(".") in ("JR", "SR", "II", "III", "IV")]
    given = [t for t in toks if t not in suffix]
    first = nick.group(1) if nick else (given[0] if given else "")
    L = last.title()
    if last.upper().startswith("MC") and len(last) > 2: L = "Mc" + last[2:].title()
    suf = ""
    if suffix:
        s = suffix[0].upper().rstrip("."); suf = " " + {"JR": "Jr.", "SR": "Sr."}.get(s, s)
    return f"{first} {L}{suf}"

def year_of(c):
    return 1789 + 2 * (c - 1)

def kim_persona(name, chamber, state, year):
    return (f"You are {name}, who serves in the U.S. {CHAMBER_LONG[chamber]} representing {STATES.get(state, state)} in {year}. "
            f"Answer the question in the first person, as {name} would, reflecting this person's actual political positions at that time.")

def anonymous_persona(chamber, state, year):
    return (f"You are a member of the U.S. {CHAMBER_LONG[chamber]} representing {STATES.get(state, state)} in {year}. "
            f"Answer the question in the first person, as this member would, reflecting this person's actual political positions at that time.")

def kim_questions(n_q=5):
    qs = []
    for ax in ["economic", "social"]:
        qs += [dict(question=q, axis=ax) for q in axis_data(ax)["questions"][20:20 + n_q]]
    return qs

class _Stop(Exception):
    pass

@torch.inference_mode()
def project(tok, model, V, prompts, batch=64):
    """(n, 2) projections on orth_e, orth_s of the last-token state at layer V['layer'] (Kim step 3: stop after that layer)."""
    L = int(V["layer"]); ve, vs = V["orth_e"].cuda().float(), V["orth_s"].cuda().float(); cap = {}
    def grab(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; cap["h"] = h[:, -1, :].float(); raise _Stop
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    res = np.zeros((len(prompts), 2), dtype=np.float32); hk = model.model.layers[L - 1].register_forward_hook(grab)
    try:
        b0 = 0
        while b0 < len(prompts):
            idx = order[b0:b0 + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            try:
                model.model(**enc)
            except _Stop:
                pass
            except torch.OutOfMemoryError:
                cap.clear(); del enc; torch.cuda.empty_cache(); batch = max(1, batch // 2); log(f"OOM -> projection batch {batch}"); continue
            h = cap.pop("h"); res[idx] = torch.stack([h @ ve, h @ vs], 1).cpu().numpy(); b0 += len(idx)
    finally:
        hk.remove()
    return res

@torch.inference_mode()
def last_logits(tok, model, prompts, batch=32):
    """next-token logits (float32, CPU) at the last position, computed from the final hidden state only (saves memory)."""
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]); order = np.argsort(lens)
    out = [None] * len(prompts)
    for b0 in range(0, len(prompts), batch):
        idx = order[b0:b0 + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        h = model.model(**enc).last_hidden_state[:, -1, :]
        lg = model.lm_head(h).float().cpu()
        for j, i in enumerate(idx): out[i] = lg[j]
    return torch.stack(out, 0)

def s4_selection_rows():
    """A13 (2026-10-16): the member-Congress-chamber rows the S4b/S4c selection was made on (PLAN A7): D/R, 93rd Congress on, both the
    economic and abortion scales, party recalled by Qwen and by Llama. Positions are averaged over exactly these rows."""
    import pandas as pd
    sc = pd.read_csv(ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
    w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()[["congress", "chamber", "icpsr"]]
    for k in ("qwen", "llama"):
        s = pd.read_csv(OUT / k / "s3" / "temporal_member_scores.csv", usecols=["congress", "chamber", "icpsr", "known", "party"])
        s = s[s.known.astype(bool) & s.party.isin(["Democrat", "Republican"])][["congress", "chamber", "icpsr"]]
        w = w.merge(s, on=["congress", "chamber", "icpsr"], validate="one_to_one")
    return w


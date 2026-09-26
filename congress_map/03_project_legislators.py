#!/usr/bin/env python
"""Step 3 - place every legislator of every Congress in the model's economic x social space.

Prompt (system): "You are {name}, who serves in the U.S. {chamber} representing {state} in {year}. Answer the question
in the first person, as {name} would, reflecting this person's actual political positions at that time."  No party is given.
User turn: 5 economic + 5 social held-out questions from data_generation/trait_data_extract (questions 20-24).
Measure: hidden state of the last prompt token at --layer, dotted with the orthogonalized unit vectors from step 1.
No text is generated for the projection; the forward pass stops right after --layer.
Score = mean over the 5 own-axis questions minus the same quantity with no persona (the model's own default = origin).

Also asks the model (greedy, 8 tokens) which party the member belonged to -> `known` flag (used to filter in the map).

Output: out/temporal_member_scores.csv  one row per member x Congress:
  congress, year, icpsr, name, party, chamber, state, known, dim1, dim2, econ_orth, social_orth, econ_raw, social_raw
  econ_orth > 0 = economic LEFT of the model default; social_orth > 0 = socially PROGRESSIVE of the default.
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--repo_root", default="..")
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--vectors", default="out/vectors.pt")
ap.add_argument("--voteview", default="data/HSall_members.csv")
ap.add_argument("--first", type=int, default=80, help="first Congress (80 = 1947)")
ap.add_argument("--last", type=int, default=118, help="last Congress (118 = 2023)")
ap.add_argument("--n_q", type=int, default=5, help="held-out questions per axis")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--no_recall", action="store_true", help="skip the party-recall check")
ap.add_argument("--out", default="out")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def year_of(c): return 1789 + 2 * (c - 1)
STATES = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"the District of Columbia","PR":"Puerto Rico","VI":"the U.S. Virgin Islands","GU":"Guam","AS":"American Samoa","MP":"the Northern Mariana Islands"}

def nice(bioname):
    """'CRUZ, Rafael Edward (Ted)' -> 'Ted Cruz'; 'MANCHIN, Joe, III' -> 'Joe Manchin III'"""
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

m = pd.read_csv(args.voteview, low_memory=False)
m = m[m.congress.between(args.first, args.last) & m.chamber.isin(["House", "Senate"]) & m.nominate_dim1.notna()].copy()
m = m.drop_duplicates(["congress", "icpsr"]).reset_index(drop=True)
m["name"] = m.bioname.apply(nice); m["year"] = m.congress.apply(year_of)
m["chamber_long"] = m.chamber.map({"House": "House of Representatives", "Senate": "Senate"})
m["state_name"] = m.state_abbrev.map(STATES).fillna(m.state_abbrev)
m["party"] = m.party_code.map({100: "Democrat", 200: "Republican"}).fillna("Other")
log(f"member-Congress rows: {len(m)} (Congress {args.first}-{args.last})")

def persona(r):
    return (f"You are {r['name']}, who serves in the U.S. {r['chamber_long']} representing {r['state_name']} in {r['year']}. "
            f"Answer the question in the first person, as {r['name']} would, reflecting this person's actual political positions at that time.")
qs = []
for ax in ["economic", "social"]:
    ev = json.load(open(Path(args.repo_root) / "data_generation" / "trait_data_extract" / f"data_{ax}.json"))["questions"][20:20 + args.n_q]
    qs += [dict(question=q, axis=ax) for q in ev]
E_IDX = [j for j, q in enumerate(qs) if q["axis"] == "economic"]; S_IDX = [j for j, q in enumerate(qs) if q["axis"] == "social"]

tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
EOS = model.generation_config.eos_token_id; EOS = [EOS] if isinstance(EOS, int) else list(EOS)
V = torch.load(args.vectors); LAYER = int(V["layer"])
vecs = {k: V[k].to(device).float() for k in ["orth_e", "orth_s", "raw_e", "raw_s"]}
log("loaded", args.model, "| layer", LAYER)
def chat(q, system=None):
    msg = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

class _Stop(Exception):
    pass
_cap = {}
def _grab(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _cap["h"] = h[:, -1, :].float()
    raise _Stop                      # nothing after LAYER is needed
@torch.inference_mode()
def proj(prompts):
    """(len(prompts), 4) projections on orth_e, orth_s, raw_e, raw_s of the last-token state at LAYER"""
    lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts])
    order = np.argsort(lens); res = np.zeros((len(prompts), 4), dtype=np.float32)
    hk = model.model.layers[LAYER - 1].register_forward_hook(_grab)
    try:
        for b0 in range(0, len(prompts), args.batch):
            idx = order[b0:b0 + args.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            try:
                model.model(**enc)
            except _Stop:
                pass
            h = _cap.pop("h")
            res[idx] = torch.stack([h @ vecs[k] for k in ["orth_e", "orth_s", "raw_e", "raw_s"]], 1).cpu().numpy()
            if (b0 // args.batch) % 100 == 0: log(f"  proj {b0}/{len(prompts)}")
    finally:
        hk.remove()
    return res

base = proj([chat(q["question"]) for q in qs])                      # model default (no persona)
if args.no_recall:
    m["known"] = np.nan
else:
    t0 = time.time(); rec = []
    prompts = [chat(f"Which political party did {r['name']}, a member of the U.S. {r['chamber_long']} from {r['state_name']} in {r['year']}, belong to? Answer with one word.") for _, r in m.iterrows()]
    with torch.inference_mode():
        for b0 in range(0, len(prompts), args.batch):
            enc = tok(prompts[b0:b0 + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=EOS)
            rec += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m["known"] = [("republican" in a.lower()) if p == "Republican" else ("democrat" in a.lower()) if p == "Democrat" else False for a, p in zip(rec, m.party)]
    log(f"party recall: {m.known.mean():.2f} ({time.time()-t0:.0f}s)")

t0 = time.time()
P = proj([chat(q["question"], persona(r)) for _, r in m.iterrows() for q in qs])
log(f"projections: {len(P)} ({time.time()-t0:.0f}s)")
nq = len(qs); rows = []
for k, (_, r) in enumerate(m.iterrows()):
    blk = P[k * nq:(k + 1) * nq]
    rows.append(dict(congress=r.congress, year=r.year, icpsr=r.icpsr, name=r["name"], party=r.party, chamber=r.chamber, state=r.state_abbrev,
                     known=r.known, dim1=r.nominate_dim1, dim2=r.nominate_dim2,
                     econ_orth=blk[E_IDX, 0].mean() - base[E_IDX, 0].mean(), social_orth=blk[S_IDX, 1].mean() - base[S_IDX, 1].mean(),
                     econ_raw=blk[E_IDX, 2].mean() - base[E_IDX, 2].mean(), social_raw=blk[S_IDX, 3].mean() - base[S_IDX, 3].mean()))
pd.DataFrame(rows).to_csv(OUT / "temporal_member_scores.csv", index=False)
log("saved", OUT / "temporal_member_scores.csv")

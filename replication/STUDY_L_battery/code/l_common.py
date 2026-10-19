"""Study L shared helpers (PLAN v0.1): one replication battery run per model (qwen, llama, olmo). Absolute push directions in each
model's own Löwdin-orthogonalized coordinates (Kim step 1 differences from STUDY_D_quadrant/out/<model>/s1/diffs_all.pt at the model's
layer), equal norms as fractions of that model's mean hidden norm (H = 0.28 mu, L = 0.20 mu), ten placebo directions per norm, Study J
items without school prayer, 10 + 10 persona bases from the Study I/J set plus the no-persona arm."""
import json, os, sys
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
LDIR = CODE.parent
KIM = LDIR.parent
sys.path.insert(0, str(KIM / "STUDY_K_separation" / "code"))
import k_common as kc                        # noqa: E402  (items without school prayer, Study J items, G/F helpers)
jc, hc, gc, fc = kc.jc, kc.hc, kc.gc, kc.fc
sys.path.insert(0, str(KIM / "STUDY_D_quadrant" / "code"))
OUT = Path(os.environ.get("STUDYL_OUT", LDIR / "out"))
MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct", "olmo": "allenai/OLMo-2-1124-7B-Instruct"}
DOUT = KIM / "STUDY_D_quadrant" / "out"
NORMS_DEFAULT = {"H": 0.28, "L": 0.20}        # fractions of mu (Study K: L = 0.280 mu); A1: per-model fractions from out/<model>/main/norms.json when present

def norms(key):
    f = OUT / key / "main" / "norms.json"
    if f.exists():
        n = json.load(open(f, encoding="utf-8")); return {"H": float(n["frac_H"]), "L": float(n["frac_L"])}
    return dict(NORMS_DEFAULT)
N_PLAC = 10
DIRS = ["Ep", "Em", "Sp", "Sm", "Dp", "Dm", "Xp", "Xm"]
N_PER_PARTY = 10
log, seed_for, atomic_csv, atomic_json, chat, lens_run, decision_layer, hbits = (
    kc.log, kc.seed_for, kc.atomic_csv, kc.atomic_json, kc.chat, kc.lens_run, kc.decision_layer, kc.hbits)
Steer, project_run, NOPERSONA, items = kc.Steer, kc.project_run, kc.NOPERSONA, kc.items

def out_dir(key, *parts):
    p = OUT.joinpath(key, *parts); p.mkdir(parents=True, exist_ok=True); return p

def layer(key):
    if key in ("qwen", "llama"): return 14      # qwen: Study G choice (F-K); llama: Study D rule layer (rule.json common_layer 14)
    r = json.load(open(DOUT / key / "rule.json", encoding="utf-8")); return int(r["common_layer"])

def load(key):
    """FP8 model + tokenizer (left padding); GPU cap 80%; RAM guard before loading"""
    torch.cuda.set_per_process_memory_fraction(fc.GPU_FRACTION, 0)
    from fp8_loader import load_fp8
    from transformers import AutoTokenizer
    fc.ram_guard()
    tok = AutoTokenizer.from_pretrained(MODELS[key]); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = load_fp8(MODELS[key])
    log("loaded", MODELS[key], f"FP8 | layers {model.config.num_hidden_layers} | GPU {torch.cuda.memory_allocated()/2**30:.1f} GB")
    return tok, model

def directions(key, L):
    """unit orth_e (+ = economic left), orth_s (+ = socially progressive) at layer L from the model's own Kim step-1 differences"""
    diffs = torch.load(DOUT / key / "s1" / "diffs_all.pt")["diffs"]
    return fc.lowdin(diffs, L)

def personas(kind="confirm"):
    P = jc.personas("confirm"); rng = np.random.default_rng(seed_for("L-personas") % 2**32)
    out = [p for party in ["Democrat", "Republican"] for p in rng.choice([q for q in P if q["party"] == party], N_PER_PARTY, replace=False)]
    return out + [dict(NOPERSONA)] if kind == "confirm+model" else out

def conditions():
    out = [dict(name="NONE", kind="none", strength="none", vec="NONE", plac=-1)]
    for d in DIRS:
        for s in ("H", "L"): out.append(dict(name=f"{d}_{s}", kind="dir", strength=s, vec=d, plac=-1))
    for s in ("H", "L"):
        for i in range(N_PLAC): out.append(dict(name=f"PLAC_{s}_{i}", kind="plac", strength=s, vec="PLAC", plac=i))
    return out

class Pusher:
    """w = NORMS[strength] * mu * unit direction; placebo directions orthogonal to both axes (fc.placebo_dirs, n = 10)"""
    def __init__(self, mu, oe, os_, fr=None):
        self.mu = mu; self.oe, self.os_ = oe, os_; self.fr = fr or dict(NORMS_DEFAULT)
        n = lambda v: v / v.norm()
        self.dirs = {"Ep": oe, "Em": -oe, "Sp": os_, "Sm": -os_, "Dp": n(oe - os_), "Dm": -n(oe - os_), "Xp": n(oe + os_), "Xm": -n(oe + os_)}
        self.plac = fc.placebo_dirs(oe, os_, n=N_PLAC)
        for d in self.plac: assert abs(float(d.norm()) - 1) < 1e-3 and abs(float(d @ oe)) < 1e-3 and abs(float(d @ os_)) < 1e-3
        assert abs(float(oe @ os_)) < 1e-4, "orth_e and orth_s must be orthogonal"
    def info(self):
        return dict(mu=self.mu, norm_H=self.fr["H"] * self.mu, norm_L=self.fr["L"] * self.mu, frac_H=self.fr["H"], frac_L=self.fr["L"], n_plac=N_PLAC,
                    cos_e_s=float(self.oe @ self.os_), cos_plac_pairs_max=float(max(abs(float(a @ b)) for i, a in enumerate(self.plac) for b in self.plac[i + 1:])))
    def w(self, cond):
        if cond["kind"] == "none": return torch.zeros_like(self.oe)
        d = self.plac[cond["plac"]] if cond["kind"] == "plac" else self.dirs[cond["vec"]]
        return self.fr[cond["strength"]] * self.mu * d

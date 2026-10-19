"""Study H shared helpers (PLAN v0.1). Reuses the checked Study G (g_common) and Study F (f_common) helpers: loading, items, chat
prompts, Steer hook, logit lens, decision layer, Löwdin, final-logit scorer, placebo directions, personas. Adds: new confirmatory
persona bases, the no-persona arm, symmetric behavioural orthogonalization, dose conditions and the push builder."""
import json, os, sys
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
H = CODE.parent
sys.path.insert(0, str(H.parent / "STUDY_G_steer_behavioral" / "code"))
import g_common as gc                       # noqa: E402
fc = gc.fc
OUT = Path(os.environ.get("STUDYH_OUT", H / "out")); KEY = "qwen"
G_CAL = H.parent / "STUDY_G_steer_behavioral" / "out" / "qwen" / "CALIBRATION.json"
CONF_STATES, CONF_YEARS = ["CO", "VA", "MN", "MO", "TN", "IN", "NJ", "OR", "SC", "LA"], [2000, 2010, 2020]
DOSES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
PLAC_DOSES = [0.5, 1.0, 1.5]
GEN_CONDS = ["NONE", "E_A_1.0", "E_A_1.5", "S_A_1.0", "S_A_1.5"]
SPEC = dict(off=0.5, plac=0.25, mass=0.80)
CELLS = {"TT": (1, 1), "TA": (1, -1), "AT": (-1, 1), "AA": (-1, -1)}
log, seed_for, atomic_csv, atomic_json, chat, items, lens_run, decision_layer, hbits = (
    gc.log, gc.seed_for, gc.atomic_csv, gc.atomic_json, gc.chat, gc.items, gc.lens_run, gc.decision_layer, gc.hbits)
Steer, Capture, score_final, mu_at, load = gc.Steer, gc.Capture, gc.score_final, gc.mu_at, gc.load
NOPERSONA = dict(pid="MODEL", party="none", state="", year=0, own=1.0, system=None)     # own +1: "against" = right / traditional

def out_dir(*parts):
    p = OUT.joinpath(KEY, *parts); p.mkdir(parents=True, exist_ok=True); return p

def personas(kind):
    """'calib' = Study G calibration bases; 'confirm' = 60 new state-year-party bases; 'confirm+model' adds the no-persona arm"""
    if kind == "calib": return gc.personas("calib")
    P = [dict(pid=f"{p[0]}-{s}-{y}", party=p, state=s, year=y, own=1.0 if p == "Democrat" else -1.0, system=gc.persona(p, s, y))
         for p in ["Democrat", "Republican"] for s in CONF_STATES for y in CONF_YEARS]
    return P + [dict(NOPERSONA)] if kind == "confirm+model" else P

def g_calibration():
    C = json.load(open(G_CAL, encoding="utf-8")); assert C.get("method") == "A3" and C.get("chosen"), "Study G A3 calibration missing"
    return C["chosen"]

def base_directions(model, tok, L):
    """Kim's Löwdin-orthogonalized response_avg directions at layer L (the map)"""
    return gc.directions(model, tok, "response_avg", L)

def orth(v, u, c):
    w = v - c * u; return w / w.norm()

def angle(a, b):
    return float(torch.rad2deg(torch.arccos(torch.clamp(a @ b, -1, 1))))

def conditions(include_plac=True):
    out = [dict(name="NONE", kind="none", axis="none", dose=0.0, de=0.0, ds=0.0, plac=-1)]
    for d in DOSES:
        out.append(dict(name=f"E_A_{d}", kind="single", axis="econ", dose=d, de=-d, ds=0.0, plac=-1))
        out.append(dict(name=f"S_A_{d}", kind="single", axis="soc", dose=d, de=0.0, ds=-d, plac=-1))
    for n, (de, ds) in CELLS.items():
        out.append(dict(name=n, kind="cell", axis="both", dose=1.0, de=float(de), ds=float(ds), plac=-1))
    if include_plac:
        for ax in ["econ", "soc"]:
            for d in PLAC_DOSES: out.append(dict(name=f"PLAC_{ax}_{d}_0", kind="plac", axis=ax, dose=d, de=0.0, ds=0.0, plac=0))
            out.append(dict(name=f"PLAC_{ax}_1.0_1", kind="plac", axis=ax, dose=1.0, de=0.0, ds=0.0, plac=1))
    return out

def usable(conds, axes):
    """drop conditions whose axis did not reach the single-issue flip (PLAN section 4)"""
    axes = set(axes)
    return [c for c in conds if c["kind"] == "none" or (c["axis"] == "both" and axes == {"econ", "soc"}) or c["axis"] in axes]

class Pusher:
    """w = mu * own * (de * k50_e * e' + ds * k50_s * s'); placebo = d * k50_axis * mu * random direction orthogonal to both axes"""
    def __init__(self, C, oe, os_):
        self.mu, self.k50 = float(C["mu"]), {k: (None if v is None else float(v)) for k, v in C["k50"].items()}
        self.oe, self.os_ = oe, os_
        self.ep = orth(oe, os_, float(C["c_e"])); self.sp = orth(os_, self.ep, float(C["c_s"]))
    def w(self, cond, own):
        if cond["kind"] == "none": return torch.zeros_like(self.oe)
        if cond["kind"] == "plac": return cond["dose"] * self.k50[cond["axis"]] * self.mu * gc.placebo(self.oe, self.os_, cond["plac"])
        ke = (self.k50["econ"] or 0.0) * cond["de"]; ks = (self.k50["soc"] or 0.0) * cond["ds"]
        return self.mu * own * (ke * self.ep + ks * self.sp)

@torch.inference_mode()
def project_run(tok, model, steer, texts, W, layer, oe, os_, batch=48):
    """last-token projections on (oe, os_) at `layer` under per-row steering W; OOM halving"""
    E, S = np.zeros(len(texts)), np.zeros(len(texts)); b0 = 0
    while b0 < len(texts):
        idx = list(range(b0, min(b0 + batch, len(texts))))
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda"); steer.W = W[idx]
            with Capture(model, [layer]) as cap:
                model.model(**enc)
            h = cap.store[layer].cpu(); E[idx], S[idx] = (h @ oe).numpy(), (h @ os_).numpy(); b0 += len(idx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(2, batch // 2); log(f"OOM -> projection batch {batch}")
        finally:
            steer.W = None
    return E, S
